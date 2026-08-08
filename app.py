import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DRO Credit Scoring Engine",
    page_icon="🛡️",
    layout="wide"
)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('baseline_model.pkl', 'rb') as f:
        baseline = pickle.load(f)
    with open('dro_model.pkl', 'rb') as f:
        dro = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return baseline, dro, features

baseline_model, dro_model, feature_names = load_models()
# ── Disagreement Finder ───────────────────────────────────────────────────────
def find_disagreement(baseline, dro, feature_names, max_tries=2000):
    """Randomly sample realistic profiles until the two models disagree."""
    rng = np.random.default_rng()
    for _ in range(max_tries):
        candidate = {
            'AGEP': int(rng.integers(18, 90)),
            'COW': int(rng.integers(1, 7)),
            'SCHL': int(rng.choice([1, 16, 19, 21, 22, 24])),
            'MAR': 1,
            'OCCP': int(rng.integers(10, 9800)),
            'POBP': int(rng.integers(1, 554)),
            'RELP': int(rng.integers(0, 18)),
            'WKHP': int(rng.integers(1, 99)),
            'SEX': int(rng.integers(1, 3)),
            'RAC1P': int(rng.choice([1, 2, 6, 8, 9]))
        }
        cand_df = pd.DataFrame([candidate])[feature_names]
        b_pred = baseline.predict(cand_df)[0]
        d_pred = dro.predict(cand_df)[0]
        if b_pred != d_pred:
            return candidate
    return None

# Initialize session state to hold found profile
if 'found_profile' not in st.session_state:
    st.session_state.found_profile = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Distributionally Robust Credit Scoring Engine")
st.markdown("""
This demo exposes a critical flaw in standard ML and shows how **Group DRO** fixes it.
Toggle between models to see how they treat the same person differently.
""")

st.divider()

# ── Sidebar: Model Toggle ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["🔴 Standard Baseline (ERM)", "🟢 DRO Model (Robust)"],
    index=1
)

selected_model = baseline_model if "Baseline" in model_choice else dro_model
model_label = "Standard Baseline" if "Baseline" in model_choice else "DRO Model"

st.sidebar.divider()
st.sidebar.markdown("### 🔎 Disagreement Finder")
if st.sidebar.button("Find a profile where models disagree"):
    with st.spinner("Scanning profiles..."):
        result = find_disagreement(baseline_model, dro_model, feature_names)
        if result:
            st.session_state.found_profile = result
            st.sidebar.success("Found a disagreement! Values loaded below.")
        else:
            st.sidebar.warning("No disagreement found in 2000 tries. Try again.")
            
st.sidebar.divider()
st.sidebar.markdown("### 📊 OOD Benchmark Results")
st.sidebar.markdown("""
| Domain | Baseline | DRO |
|--------|----------|-----|
| California | 82.15% | 82.77% |
| Puerto Rico | 69.89% 🚨 | 73.30% ✅ |
""")

# ── Input Features ────────────────────────────────────────────────────────────
st.subheader("👤 Enter Individual Profile")

col1, col2, col3 = st.columns(3)

# Use found profile values as defaults if available
fp = st.session_state.found_profile or {}

with col1:
    AGEP = st.slider("Age", 18, 90, fp.get('AGEP', 35))
    WKHP = st.slider("Hours Worked Per Week", 1, 99, fp.get('WKHP', 40))
    schl_options = {
        1: "No schooling", 16: "High School Diploma",
        19: "Some College", 21: "Bachelor's Degree",
        22: "Master's Degree", 24: "Doctorate"
    }
    schl_keys = list(schl_options.keys())
    SCHL = st.selectbox("Education Level", schl_keys,
        index=schl_keys.index(fp.get('SCHL', 21)),
        format_func=lambda x: schl_options[x])

with col2:
    cow_options = {
        1: "Private for-profit", 2: "Private non-profit",
        3: "Local government", 4: "State government",
        5: "Federal government", 6: "Self-employed"
    }
    cow_keys = list(cow_options.keys())
    COW = st.selectbox("Class of Worker", cow_keys,
        index=cow_keys.index(fp.get('COW', 1)),
        format_func=lambda x: cow_options[x])
    OCCP = st.slider("Occupation Code", 10, 9800, fp.get('OCCP', 1000), step=10)
    POBP = st.slider("Place of Birth Code", 1, 554, fp.get('POBP', 6))

with col3:
    rac_options = {
        1: "White", 2: "Black / African American",
        6: "Asian", 8: "Other", 9: "Two or More Races"
    }
    rac_keys = list(rac_options.keys())
    RAC1P = st.selectbox("Race", rac_keys,
        index=rac_keys.index(fp.get('RAC1P', 1)),
        format_func=lambda x: rac_options[x])
    SEX = st.radio("Sex", [1, 2],
        index=[1, 2].index(fp.get('SEX', 1)),
        format_func=lambda x: "Male" if x == 1 else "Female")
    RELP = st.slider("Relationship to Household", 0, 17, fp.get('RELP', 0))

# ── Build Input Vector ────────────────────────────────────────────────────────
input_dict = {
    'AGEP': AGEP, 'COW': COW, 'SCHL': SCHL, 'MAR': 1,
    'OCCP': OCCP, 'POBP': POBP, 'RELP': RELP,
    'WKHP': WKHP, 'SEX': SEX, 'RAC1P': RAC1P
}
input_df = pd.DataFrame([input_dict])[feature_names]

# ── Prediction ────────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"🎯 Prediction — {model_label}")

pred = selected_model.predict(input_df)[0]
proba = selected_model.predict_proba(input_df)[0]
confidence = proba[1] if pred == 1 else proba[0]

col_pred, col_conf = st.columns(2)

with col_pred:
    if pred == 1:
        st.success("✅ Predicted Income: **Above $50,000**")
    else:
        st.error("❌ Predicted Income: **Below $50,000**")

with col_conf:
    st.metric("Confidence Score", f"{confidence * 100:.1f}%")

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Why did the model make this prediction? (SHAP)")

with st.spinner("Calculating SHAP values..."):
    explainer = shap.TreeExplainer(selected_model)
    # Modern SHAP API returns an Explanation object directly
    explanation = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(fig)
    plt.close()

# ── Side by Side Comparison ───────────────────────────────────────────────────
st.divider()
st.subheader("⚖️ Side-by-Side Model Comparison")

baseline_pred = baseline_model.predict(input_df)[0]
baseline_proba = baseline_model.predict_proba(input_df)[0]
dro_pred = dro_model.predict(input_df)[0]
dro_proba = dro_model.predict_proba(input_df)[0]

col_b, col_d = st.columns(2)

with col_b:
    st.markdown("### 🔴 Standard Baseline (ERM)")
    if baseline_pred == 1:
        st.success("Predicted: Above $50K")
    else:
        st.error("Predicted: Below $50K")
    st.metric("Confidence", f"{max(baseline_proba) * 100:.1f}%")

with col_d:
    st.markdown("### 🟢 DRO Model (Robust)")
    if dro_pred == 1:
        st.success("Predicted: Above $50K")
    else:
        st.error("Predicted: Below $50K")
    st.metric("Confidence", f"{max(dro_proba) * 100:.1f}%")

if baseline_pred != dro_pred:
    st.warning("⚠️ The two models **disagree** on this prediction — this is where DRO's minority protection is active.")
else:
    st.info("Both models agree on this prediction. Try changing the Race or Education inputs to see where they diverge.")

st.divider()
st.caption("Built with XGBoost + Group DRO | Data: US Census ACS 2018 | [GitHub](https://github.com/Rijin21/DRO-Credit-Scoring)")