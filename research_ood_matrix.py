import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import warnings

warnings.filterwarnings('ignore')

def get_data(state_code):
    source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    data = source.get_data(states=[state_code], download=True)
    features, label, group = ACSIncome.df_to_numpy(data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    return df, label, group

def evaluate_worst_group(model, X, y, groups):
    preds = model.predict(X)
    global_acc = accuracy_score(y, preds)
    group_accs = {g: accuracy_score(y[groups == g], preds[groups == g]) for g in [1, 2, 6, 8, 9] if np.sum(groups == g) > 0}
    return global_acc, min(group_accs.values())

def train_models(X_train, y_train, groups_train, seed):
    # Baseline
    base = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1).fit(X_train, y_train)
    
    # DRO
    weights = np.ones(len(y_train))
    dro = xgb.XGBClassifier(n_estimators=15, max_depth=5, min_child_weight=5, reg_lambda=2.0, learning_rate=0.1, random_state=seed, n_jobs=-1)
    for _ in range(10):
        dro.fit(X_train, y_train, sample_weight=weights, xgb_model=dro if _ > 0 else None)
        probs = dro.predict_proba(X_train)
        for g in [1, 2, 6, 8, 9]:
            mask = (groups_train == g)
            if np.sum(mask) > 0:
                weights[mask] *= np.exp(0.05 * log_loss(y_train[mask], probs[mask]))
        weights = weights / np.sum(weights) * len(y_train)
        
    return base, dro

if __name__ == "__main__":
    print("\n🌪️ THE OOD STRESS TEST MATRIX 🌪️\n")
    
    # Pairs: (Extreme Shift, Moderate Shift, Mild Shift)
    domain_pairs = [('CA', 'PR'), ('NY', 'AL'), ('WA', 'OR')]
    seeds = [42, 101] # Running 2 seeds to save time, feel free to add more!
    
    results = []
    
    for source, target in domain_pairs:
        print(f"\nDownloading data for Pair: {source} -> {target}")
        X_src, y_src, g_src = get_data(source)
        X_tgt, y_tgt, g_tgt = get_data(target)
        
        for seed in seeds:
            print(f"  -> Training on {source} (Seed {seed})...")
            base_model, dro_model = train_models(X_src, y_src, g_src, seed)
            
            print(f"  -> Testing on {target}...")
            # Evaluate Baseline
            base_glob, base_worst = evaluate_worst_group(base_model, X_tgt, y_tgt, g_tgt)
            # Evaluate DRO
            dro_glob, dro_worst = evaluate_worst_group(dro_model, X_tgt, y_tgt, g_tgt)
            
            results.append({
                'Source': source, 'Target': target, 'Seed': seed,
                'Base_Global': base_glob, 'DRO_Global': dro_glob,
                'Base_Worst_Group': base_worst, 'DRO_Worst_Group': dro_worst
            })
            
    res_df = pd.DataFrame(results)
    
    # Calculate the average collapse
    res_df['Global_Advantage'] = res_df['DRO_Global'] - res_df['Base_Global']
    res_df['Worst_Group_Advantage'] = res_df['DRO_Worst_Group'] - res_df['Base_Worst_Group']
    
    print("\n=======================================================")
    print("📊 FINAL OOD ROBUSTNESS MATRIX")
    print("=======================================================\n")
    print(res_df[['Source', 'Target', 'Seed', 'Worst_Group_Advantage']].to_string(index=False))
    
    print(f"\n✅ Matrix completed. Average DRO Worst-Group Advantage across all domains and seeds: {res_df['Worst_Group_Advantage'].mean()*100:.2f}%")