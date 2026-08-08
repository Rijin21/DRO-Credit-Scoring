import pickle
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data(state_code='CA'):
    print(f"Downloading Census Data for {state_code}...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state_code], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df['Target_Income'] = label
    df['Race_Code'] = group
    return df

def train_baseline(X_train, y_train):
    print("Training Baseline model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def train_dro(X_train, y_train, group_train):
    print("Training DRO model...")
    race_mapping = {1: "White", 2: "Black", 6: "Asian", 8: "Other", 9: "Two or More"}
    valid_groups = [g for g in np.unique(group_train) if g in race_mapping]
    
    n_samples = len(y_train)
    sample_weights = np.ones(n_samples)
    
    dro_iterations = 10
    learning_rate = 0.05
    trees_per_iteration = 15
    final_model = None

    model = xgb.XGBClassifier(
        n_estimators=trees_per_iteration,
        max_depth=5,
        min_child_weight=5,
        reg_lambda=2.0,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    for iteration in range(dro_iterations):
        print(f"  DRO Iteration {iteration + 1}/{dro_iterations}")
        if final_model is None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.n_estimators += trees_per_iteration
            model.fit(X_train, y_train, sample_weight=sample_weights, xgb_model=final_model)
        final_model = model

        train_preds_proba = model.predict_proba(X_train)
        group_losses = {}
        for g in valid_groups:
            mask = (group_train == g)
            if np.sum(mask) == 0:
                continue
            group_losses[g] = log_loss(y_train[mask], train_preds_proba[mask])

        for g in valid_groups:
            mask = (group_train == g)
            sample_weights[mask] *= np.exp(learning_rate * group_losses[g])
        sample_weights = sample_weights / np.sum(sample_weights) * n_samples

    return final_model

if __name__ == "__main__":
    # Load data
    df = load_data('CA')
    X = df.drop(columns=['Target_Income', 'Race_Code'])
    y = df['Target_Income']
    groups = df['Race_Code'].astype(int)

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )

    # Train and save baseline
    baseline_model = train_baseline(X_train, y_train)
    with open('baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline_model, f)
    print("Baseline model saved to baseline_model.pkl")

    # Train and save DRO
    dro_model = train_dro(X_train, y_train, group_train)
    with open('dro_model.pkl', 'wb') as f:
        pickle.dump(dro_model, f)
    print("DRO model saved to dro_model.pkl")

    # Save feature names for the app
    feature_names = list(X.columns)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Feature names saved to feature_names.pkl")

    print("\nAll models saved successfully!")