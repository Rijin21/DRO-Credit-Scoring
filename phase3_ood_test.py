import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def load_census_data(state_code):
    """Loads Census Data for a specific state."""
    print(f"📥 Downloading Census Data for {state_code}...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state_code], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df['Target_Income'] = label
    df['Race_Code'] = group
    return df

def get_trained_baseline(df_train):
    """Trains the standard ERM XGBoost Model."""
    print("🚀 Training Standard Baseline Model (on Source State)...")
    X = df_train.drop(columns=['Target_Income', 'Race_Code'])
    y = df_train['Target_Income']
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def get_trained_dro(df_train):
    """Trains the DRO Model using Dynamic Boosting."""
    print("🛡️ Training DRO Model (on Source State)...")
    X = df_train.drop(columns=['Target_Income', 'Race_Code'])
    y = df_train['Target_Income']
    groups = df_train['Race_Code'].astype(int)
    
    valid_groups = [1, 2, 6, 8, 9]
    n_samples = len(y)
    sample_weights = np.ones(n_samples)
    
    dro_iterations = 10    
    learning_rate = 0.05   
    trees_per_iter = 15 
    
    final_model = None
    model = xgb.XGBClassifier(
        n_estimators=trees_per_iter, max_depth=5, min_child_weight=5, 
        reg_lambda=2.0, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    
    for iteration in range(dro_iterations):
        if final_model is None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.n_estimators += trees_per_iter
            model.fit(X, y, sample_weight=sample_weights, xgb_model=final_model)
            
        final_model = model
        train_preds_proba = model.predict_proba(X)
        
        group_losses = {}
        for g in valid_groups:
            mask = (groups == g)
            if np.sum(mask) > 0:
                group_losses[g] = log_loss(y[mask], train_preds_proba[mask])
                
        for g in valid_groups:
            mask = (groups == g)
            sample_weights[mask] *= np.exp(learning_rate * group_losses.get(g, 0))
        sample_weights = sample_weights / np.sum(sample_weights) * n_samples

    return final_model

def evaluate_on_new_domain(model, df_test, model_name, state_name):
    """Tests the model on a completely new state."""
    X_test = df_test.drop(columns=['Target_Income', 'Race_Code'])
    y_test = df_test['Target_Income']
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"   -> {model_name} Accuracy in {state_name}: {acc * 100:.2f}%")
    return acc

if __name__ == "__main__":
    print("\n=======================================================")
    print("🌪️ PHASE 3: OUT-OF-DISTRIBUTION (DOMAIN SHIFT) TEST")
    print("=======================================================\n")
    
    # 1. Load Data
    df_source = load_census_data('CA') # Train on California
    df_target = load_census_data('PR') # Deploy in Puerto Rico (EXTREME SHIFT)
    
    # 2. Train Both Models on California
    print("\n--- Training Phase (Source Domain: CA) ---")
    baseline_model = get_trained_baseline(df_source)
    dro_model = get_trained_dro(df_source)
    
    # 3. Simulate Domain Shift (Deploying to Texas)
    print("\n--- Deployment Phase (Target Domain: PR) ---")
    print("Deploying CA-trained models to Puerto Rico residents...\n")
    
    base_acc = evaluate_on_new_domain(baseline_model, df_target, "Standard Baseline", "Puerto Rico")
    dro_acc = evaluate_on_new_domain(dro_model, df_target, "Distributionally Robust", "Puerto Rico")
    
    print("\n=======================================================")
    print("📊 OOD SURVIVAL RESULTS")
    print("=======================================================")
    if dro_acc > base_acc:
        print("✅ SUCCESS! The DRO model survived the domain shift better than the Baseline.")
        print(f"   The DRO model outperformed the Baseline by {(dro_acc - base_acc) * 100:.2f}% in the new domain.")
    else:
        print("❌ The Baseline survived better. (This means Texas and California were too similar).")