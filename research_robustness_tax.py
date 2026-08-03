import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_census_data(state_code='CA'):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state_code], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df['Target_Income'] = label
    df['Race_Code'] = group
    return df

def get_accuracy_metrics(model, X_test, y_test, group_test):
    preds = model.predict(X_test)
    global_acc = accuracy_score(y_test, preds)
    
    valid_groups = [1, 2, 6, 8, 9]
    group_accuracies = {}
    for g in valid_groups:
        mask = (group_test == g)
        if np.sum(mask) > 0:
            group_accuracies[g] = accuracy_score(y_test[mask], preds[mask])
            
    worst_group_acc = min(group_accuracies.values())
    return global_acc, worst_group_acc

def train_dro_with_lr(df, learning_rate):
    X = df.drop(columns=['Target_Income', 'Race_Code'])
    y = df['Target_Income']
    groups = df['Race_Code'].astype(int)
    
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, groups, test_size=0.2, random_state=42)
    
    valid_groups = [1, 2, 6, 8, 9]
    sample_weights = np.ones(len(y_train))
    
    final_model = None
    model = xgb.XGBClassifier(n_estimators=15, max_depth=5, min_child_weight=5, reg_lambda=2.0, learning_rate=0.1, random_state=42, n_jobs=-1)
    
    for _ in range(10): # 10 iterations of DRO
        if final_model is None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.n_estimators += 15
            model.fit(X_train, y_train, sample_weight=sample_weights, xgb_model=final_model)
            
        final_model = model
        train_preds_proba = model.predict_proba(X_train)
        
        group_losses = {g: log_loss(y_train[group_train == g], train_preds_proba[group_train == g]) for g in valid_groups if np.sum(group_train == g) > 0}
        for g in valid_groups:
            mask = (group_train == g)
            sample_weights[mask] *= np.exp(learning_rate * group_losses.get(g, 0))
        sample_weights = sample_weights / np.sum(sample_weights) * len(y_train)

    return get_accuracy_metrics(final_model, X_test, y_test, group_test)

if __name__ == "__main__":
    print("\n--- Running Robustness Tax Study ---")
    df = load_census_data('CA')
    
    # 1. Baseline
    print("Evaluating Baseline ERM...")
    X = df.drop(columns=['Target_Income', 'Race_Code'])
    y = df['Target_Income']
    X_train, X_test, y_train, y_test, _, group_test = train_test_split(X, y, df['Race_Code'], test_size=0.2, random_state=42)
    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
    base_global, base_worst = get_accuracy_metrics(base_model, X_test, y_test, group_test)
    
    results = [{'LR': 'Baseline', 'Global Acc': base_global, 'Worst-Group Acc': base_worst}]
    
    # 2. DRO across different Learning Rates
    lrs_to_test = [0.01, 0.05, 0.1, 0.25, 0.5]
    for lr in lrs_to_test:
        print(f"Evaluating DRO with Learning Rate: {lr}...")
        g_acc, w_acc = train_dro_with_lr(df, learning_rate=lr)
        results.append({'LR': f'DRO ({lr})', 'Global Acc': g_acc, 'Worst-Group Acc': w_acc})
        
    # 3. Plotting the Pareto Curve
    res_df = pd.DataFrame(results)
    print("\n--- Robustness Tax Trade-off Matrix ---")
    print(res_df.to_string(index=False))
    
    plt.figure(figsize=(8, 6))
    plt.scatter(res_df['Global Acc'], res_df['Worst-Group Acc'], color='blue', s=100)
    for i, row in res_df.iterrows():
        plt.annotate(row['LR'], (row['Global Acc'], row['Worst-Group Acc']), xytext=(5, 5), textcoords='offset points')
    
    plt.title('The Robustness Tax: Global vs. Worst-Group Accuracy')
    plt.xlabel('Global Accuracy (Majority Performance)')
    plt.ylabel('Worst-Group Accuracy (Minority Safety)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('robustness_tax_pareto.png')
    print("\n✅ Pareto curve saved as 'robustness_tax_pareto.png'")