import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def load_census_data(state_code='CA'):
    """
    Downloads the ACS Income dataset from US Census data via Folktables.
    Target: Predict if a person makes > $50k (Proxy for Credit/Loan Approval).
    """
    print(f"📥 Downloading Census Data for {state_code} (This might take a few seconds)...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state_code], download=True)
    
    # Extract features, target label, and the "protected group" (Race)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    
    # Convert to DataFrame for easier handling
    feature_names = ACSIncome.features
    df = pd.DataFrame(features, columns=feature_names)
    df['Target_Income'] = label
    df['Race_Code'] = group
    
    print(f"✅ Data loaded! Total records: {len(df)}")
    return df

def train_and_evaluate_baseline(df):
    """
    Trains a standard XGBoost model and exposes the 'Average Accuracy' trap.
    """
    print("\n🚀 Training Baseline XGBoost Model...")
    
    # 1. Prepare Data
    X = df.drop(columns=['Target_Income', 'Race_Code'])
    y = df['Target_Income']
    groups = df['Race_Code']
    
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )
    
    # 2. Train Standard Model (Minimizes Average Loss)
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluate Global (Average) Accuracy
    # Note: XGBoost Regressor outputs probabilities, so we round to 0 or 1 for classification
    test_preds = np.round(model.predict(X_test))
    global_acc = accuracy_score(y_test, test_preds)
    
    print("\n=======================================================")
    print(f"🌍 GLOBAL MODEL ACCURACY: {global_acc * 100:.2f}%")
    print("=======================================================")
    print("Looks great to a standard Data Scientist, right? Now let's dig deeper...\n")
    
    # 4. Expose the "Trap" (Subpopulation Accuracy)
    print("🔍 ACCURACY BY DEMOGRAPHIC GROUP:")
    
    # Group code mapping from the US Census Bureau
    race_mapping = {
        1: "White", 2: "Black", 6: "Asian", 8: "Other", 9: "Two or More"
    }
    
    group_accuracies = {}
    
    # Calculate accuracy for each specific demographic
    for group_code in np.unique(group_test):
        if group_code not in race_mapping:
            continue # Skip tiny micro-groups for cleaner output
            
        group_mask = (group_test == group_code)
        y_test_group = y_test[group_mask]
        preds_group = test_preds[group_mask]
        
        acc = accuracy_score(y_test_group, preds_group)
        group_accuracies[race_mapping[group_code]] = acc
        
        # Calculate group size to show imbalance
        group_size = len(y_test_group)
        percentage_of_data = (group_size / len(y_test)) * 100
        
        print(f"  -> {race_mapping[group_code]:<12}: {acc * 100:.2f}% Accuracy (Represents {percentage_of_data:.1f}% of data)")

    # Find the worst performing group
    worst_group = min(group_accuracies, key=group_accuracies.get)
    worst_acc = group_accuracies[worst_group]
    
    print("\n🚨 THE STANDARD ML TRAP EXPOSED 🚨")
    print(f"The model sacrifices the '{worst_group}' demographic (only {worst_acc * 100:.2f}% accuracy)")
    print("to artificially inflate the global accuracy on the majority groups.")

if __name__ == "__main__":
    census_df = load_census_data(state_code='CA')
    train_and_evaluate_baseline(census_df)