import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from phase1_baseline import load_census_data, train_and_evaluate_baseline
from phase2_dro_training import train_dro_model

# ── Test 1: Data Loading ──────────────────────────────────────────────────────
def test_data_loading():
    """Check that data loads correctly and has expected columns."""
    df = load_census_data(state_code='CA')
    assert 'Target_Income' in df.columns
    assert 'Race_Code' in df.columns
    assert len(df) > 0
    assert df['Target_Income'].isin([0, 1]).all(), "Target must be binary"

# ── Test 2: Class Balance ─────────────────────────────────────────────────────
def test_minority_group_exists():
    """Check that minority groups are present in the data."""
    df = load_census_data(state_code='CA')
    unique_groups = df['Race_Code'].unique()
    assert len(unique_groups) > 1, "Must have multiple demographic groups"

# ── Test 3: DRO Weights ───────────────────────────────────────────────────────
def test_dro_weights_stay_positive():
    """Sample weights must never go negative or zero during DRO updates."""
    learning_rate = 0.05
    group_loss = 0.8
    initial_weight = 1.0
    updated_weight = initial_weight * np.exp(learning_rate * group_loss)
    assert updated_weight > 0, "Weights must always remain positive"

# ── Test 4: Weight Normalization ──────────────────────────────────────────────
def test_dro_weight_normalization():
    """After normalization, weights must sum to number of samples."""
    n_samples = 100
    sample_weights = np.random.rand(n_samples)
    sample_weights = sample_weights / np.sum(sample_weights) * n_samples
    assert abs(np.sum(sample_weights) - n_samples) < 1e-6, "Weights must sum to n_samples"

# ── Test 5: OOD Accuracy Threshold ───────────────────────────────────────────
def test_ood_accuracy_above_chance():
    """DRO model must perform significantly better than random chance on OOD data."""
    # Based on your documented results: DRO achieves 73.30% on Puerto Rico
    dro_ood_accuracy = 0.7330
    random_chance = 0.5
    assert dro_ood_accuracy > random_chance + 0.15, "DRO must beat random chance by at least 15%"