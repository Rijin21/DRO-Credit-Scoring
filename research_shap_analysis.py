import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_data():
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=['CA'], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    return df, label, group

if __name__ == "__main__":
    print("\n--- Generating SHAP Feature Importance ---")
    df, labels, groups = load_data()
    
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(df, labels, groups, test_size=0.2, random_state=42)
    
    # Train Baseline
    print("Training Baseline for SHAP...")
    baseline = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    baseline.fit(X_train, y_train)
    
    # Train A Quick DRO Model (Hardcoded to the best parameters from Phase 2)
    print("Training DRO for SHAP...")
    sample_weights = np.ones(len(y_train))
    dro_model = xgb.XGBClassifier(n_estimators=15, max_depth=5, min_child_weight=5, reg_lambda=2.0, learning_rate=0.1, random_state=42, n_jobs=-1)
    for _ in range(10):
        dro_model.fit(X_train, y_train, sample_weight=sample_weights, xgb_model=dro_model if _ > 0 else None)
        probs = dro_model.predict_proba(X_train)
        for g in [1, 2, 6, 8, 9]:
            mask = (g_train == g)
            from sklearn.metrics import log_loss
            if np.sum(mask) > 0:
                loss = log_loss(y_train[mask], probs[mask])
                sample_weights[mask] *= np.exp(0.05 * loss)
        sample_weights = sample_weights / np.sum(sample_weights) * len(y_train)

    print("\nCalculating SHAP values (this takes a moment)...")
    # We take a sample of 1000 to keep the script fast
    X_sample = X_test.sample(1000, random_state=42)
    
    # Baseline SHAP
    explainer_base = shap.TreeExplainer(baseline)
    shap_values_base = explainer_base(X_sample)
    
    # DRO SHAP
    explainer_dro = shap.TreeExplainer(dro_model)
    shap_values_dro = explainer_dro(X_sample)
    
    # Plot and Save
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_base, X_sample, show=False)
    plt.title("Baseline Model: Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_baseline.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_dro, X_sample, show=False)
    plt.title("DRO Model: Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_dro.png')
    plt.close()
    
    print("✅ SHAP plots saved as 'shap_baseline.png' and 'shap_dro.png'.")
    print("Compare the two images to see which fragile features the DRO model actively ignored!")