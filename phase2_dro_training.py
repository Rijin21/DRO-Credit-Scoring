import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def load_census_data(state_code='CA'):
    """Loads the Folktables Census Data."""
    print(f"📥 Downloading Census Data for {state_code}...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state_code], download=True)
    
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    feature_names = ACSIncome.features
    
    df = pd.DataFrame(features, columns=feature_names)
    df['Target_Income'] = label
    df['Race_Code'] = group
    return df

def train_dro_model(df):
    """
    Implements a custom Group Distributionally Robust Optimization (DRO) loop.
    Instead of ERM (Average Loss), we optimize for the Minimax (Worst-Group Loss).
    """
    print("\n🚀 Starting Group DRO Custom Training Loop...")
    
    # 1. Prepare Data
    X = df.drop(columns=['Target_Income', 'Race_Code'])
    y = df['Target_Income']
    groups = df['Race_Code'].astype(int)
    
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )
    
    # Group mapping
    race_mapping = {1: "White", 2: "Black", 6: "Asian", 8: "Other", 9: "Two or More"}
    valid_groups = [g for g in np.unique(group_train) if g in race_mapping]
    
    # 2. Initialize Uniform Sample Weights
    # Everyone starts with a weight of 1.0
    n_samples = len(y_train)
    sample_weights = np.ones(n_samples)
    
    # DRO Hyperparameters (FIXED)
    dro_iterations = 10    # Increased iterations for smoother convergence
    learning_rate = 0.05   # Lowered from 0.5 to prevent violent weight explosion
    trees_per_iteration = 15 # NEW: How many trees we add per loop
    
    final_model = None
    best_worst_acc = 0.0
    
    # Initialize the base model ONCE outside the loop
    model = xgb.XGBClassifier(
        n_estimators=trees_per_iteration, 
        max_depth=5,           # Restored capacity so it can actually learn
        min_child_weight=5,    
        reg_lambda=2.0,       
        learning_rate=0.1, 
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # 3. The Minimax DRO Loop
    for iteration in range(dro_iterations):
        print(f"\n--- DRO Iteration {iteration + 1}/{dro_iterations} ---")
        
        # THE FIX: Dynamic Boosting Intervention
        if final_model is None:
            # Train the first batch of trees on normal data
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # Add new trees to the EXISTING model using the new paranoid weights
            model.n_estimators += trees_per_iteration
            model.fit(X_train, y_train, sample_weight=sample_weights, xgb_model=final_model)
            
        final_model = model
        
        # Calculate loss for each specific demographic group
        train_preds_proba = model.predict_proba(X_train)
        
        group_losses = {}
        for g in valid_groups:
            mask = (group_train == g)
            if np.sum(mask) == 0: continue
            
            # We use Log Loss (Cross Entropy) to measure exactly how "wrong" the model is
            g_loss = log_loss(y_train[mask], train_preds_proba[mask])
            group_losses[g] = g_loss
            
        # Find the group the model is currently failing the most
        worst_group = max(group_losses, key=group_losses.get)
        print(f"Worst performing group this round: {race_mapping[worst_group]}")
        
        # 4. Exponentiated Gradient Weight Update (The Math of DRO)
        # Increase the weight of groups that have high loss, decrease for low loss
        for g in valid_groups:
            mask = (group_train == g)
            # Math: weight = weight * e^(learning_rate * loss)
            sample_weights[mask] *= np.exp(learning_rate * group_losses[g])
            
        # Normalize weights so they sum to the number of samples (stability)
        sample_weights = sample_weights / np.sum(sample_weights) * n_samples

    # 5. Final Evaluation on Test Set
    print("\n=======================================================")
    print("🛡️ DRO MODEL FINAL EVALUATION")
    print("=======================================================")
    
    test_preds = final_model.predict(X_test)
    global_acc = accuracy_score(y_test, test_preds)
    print(f"🌍 GLOBAL MODEL ACCURACY: {global_acc * 100:.2f}%")
    print("-------------------------------------------------------")
    
    for g in valid_groups:
        mask = (group_test == g)
        g_acc = accuracy_score(y_test[mask], test_preds[mask])
        print(f"  -> {race_mapping[g]:<12}: {g_acc * 100:.2f}% Accuracy")

if __name__ == "__main__":
    census_df = load_census_data(state_code='CA')
    train_dro_model(census_df)