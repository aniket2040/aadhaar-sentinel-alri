"""
Shared test fixtures and Hypothesis generators for Aadhaar Sentinel tests.

This module provides:
- Hypothesis strategies for generating valid test data
- Pytest fixtures for common test setup
- Custom generators for domain-specific data types
"""

import pytest
from hypothesis import strategies as st, settings
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


# =============================================================================
# Hypothesis Settings
# =============================================================================

settings.register_profile("default", max_examples=100)
settings.load_profile("default")


# =============================================================================
# Basic Data Strategies
# =============================================================================

# Valid Indian state names (lowercase)
VALID_STATES = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
    'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
    'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal'
]

state_strategy = st.sampled_from(VALID_STATES)

# District name strategy (lowercase alphabetic)
district_strategy = st.text(
    min_size=3,
    max_size=30,
    alphabet=st.characters(whitelist_categories=('Ll',), whitelist_characters=' ')
).map(lambda s: s.strip().lower()).filter(lambda s: len(s) >= 3)

# Valid 6-digit PIN codes
pincode_strategy = st.integers(min_value=100000, max_value=999999)

# Year strategy (reasonable range for Aadhaar data)
year_strategy = st.integers(min_value=2020, max_value=2026)

# Month strategy (1-12)
month_strategy = st.integers(min_value=1, max_value=12)

# Day strategy (1-28 to avoid month-end issues)
day_strategy = st.integers(min_value=1, max_value=28)

# Count strategy (non-negative integers)
count_strategy = st.integers(min_value=0, max_value=100000)


# =============================================================================
# CSV Row Strategies
# =============================================================================

@st.composite
def enrollment_row_strategy(draw):
    """Generate a valid enrollment CSV row."""
    return {
        'state': draw(state_strategy),
        'district': draw(district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_enrollment_age': draw(count_strategy)
    }


@st.composite
def demographic_row_strategy(draw):
    """Generate a valid demographic CSV row."""
    return {
        'state': draw(state_strategy),
        'district': draw(district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_demographic_age': draw(count_strategy)
    }


@st.composite
def biometric_row_strategy(draw):
    """Generate a valid biometric CSV row."""
    return {
        'state': draw(state_strategy),
        'district': draw(district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_biometric_age': draw(count_strategy)
    }


# =============================================================================
# Sub-score Strategies
# =============================================================================

# Sub-scores are always in [0, 1] range
sub_score_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False
)

@st.composite
def sub_scores_strategy(draw):
    """Generate valid sub-scores dictionary."""
    return {
        'coverage': draw(sub_score_strategy),
        'instability': draw(sub_score_strategy),
        'biometric': draw(sub_score_strategy),
        'anomaly': draw(sub_score_strategy)
    }


# =============================================================================
# ALRI Weights Strategy
# =============================================================================

@st.composite
def alri_weights_strategy(draw):
    """Generate valid ALRI weights that sum to 1.0."""
    # Generate 4 random values and normalize
    values = [draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False)) for _ in range(4)]
    total = sum(values)
    normalized = [v / total for v in values]
    return {
        'coverage': normalized[0],
        'instability': normalized[1],
        'biometric': normalized[2],
        'anomaly': normalized[3]
    }


# =============================================================================
# Reason Code Strategies
# =============================================================================

VALID_REASON_CODES = [
    'Low_Child_Enrolment',
    'High_Address_Churn',
    'Low_Biometric_Update_5to15',
    'Anomalous_Data_Entry'
]

VALID_SEVERITIES = ['Low', 'Medium', 'High', 'Critical']

reason_code_strategy = st.sampled_from(VALID_REASON_CODES)
severity_strategy = st.sampled_from(VALID_SEVERITIES)


@st.composite
def reason_code_object_strategy(draw):
    """Generate a valid ReasonCode object dictionary."""
    return {
        'code': draw(reason_code_strategy),
        'description': draw(st.text(min_size=10, max_size=100)),
        'severity': draw(severity_strategy),
        'contribution': draw(sub_score_strategy),
        'affected_population': draw(st.integers(min_value=0, max_value=1000000))
    }


# =============================================================================
# ALRI Record Strategies
# =============================================================================

@st.composite
def alri_record_strategy(draw):
    """Generate a valid ALRI record for serialization testing."""
    return {
        'district': draw(district_strategy),
        'state': draw(state_strategy),
        'alri_score': draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
        'sub_scores': draw(sub_scores_strategy()),
        'reason_codes': draw(st.lists(reason_code_strategy, min_size=1, max_size=4)),
        'computed_at': draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2026, 12, 31)
        )).isoformat()
    }


# =============================================================================
# Time Series Strategies
# =============================================================================

@st.composite
def time_series_strategy(draw, min_length=24, max_length=60):
    """Generate a valid time series for anomaly detection."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    # Generate base values with some trend and seasonality
    base = draw(st.floats(min_value=100, max_value=10000, allow_nan=False))
    values = []
    for i in range(length):
        # Add trend component
        trend = i * draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
        # Add seasonal component (monthly pattern)
        seasonal = draw(st.floats(min_value=-100, max_value=100, allow_nan=False)) * (i % 12)
        # Add noise
        noise = draw(st.floats(min_value=-50, max_value=50, allow_nan=False))
        values.append(max(0, base + trend + seasonal + noise))
    return values


# =============================================================================
# Forecast Strategies
# =============================================================================

@st.composite
def forecast_result_strategy(draw):
    """Generate a valid forecast result."""
    horizon = draw(st.integers(min_value=3, max_value=6))
    base_forecast = [draw(st.floats(min_value=100, max_value=10000, allow_nan=False)) 
                     for _ in range(horizon)]
    margin = draw(st.floats(min_value=10, max_value=100, allow_nan=False))
    
    return {
        'district': draw(district_strategy),
        'metric': draw(st.sampled_from(['enrollment', 'demographic', 'biometric'])),
        'forecast_values': base_forecast,
        'lower_bound': [v - margin for v in base_forecast],
        'upper_bound': [v + margin for v in base_forecast],
        'trend': draw(st.sampled_from(['increasing', 'decreasing', 'stable']))
    }


# =============================================================================
# Cluster Strategies
# =============================================================================

CLUSTER_LABELS = [
    'Stable-HighCoverage',
    'Migratory-HighChurn',
    'ChildGap-HighRisk',
    'LowActivity-Rural'
]

cluster_label_strategy = st.sampled_from(CLUSTER_LABELS)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def sample_enrollment_df():
    """Create a sample enrollment DataFrame for testing."""
    return pd.DataFrame([
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 1, 'total_enrollment_age': 100},
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 2, 'total_enrollment_age': 150},
        {'state': 'karnataka', 'district': 'mysore', 'pincode': 570001,
         'year': 2024, 'month': 1, 'day': 1, 'total_enrollment_age': 80},
    ])


@pytest.fixture
def sample_demographic_df():
    """Create a sample demographic DataFrame for testing."""
    return pd.DataFrame([
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 1, 'total_demographic_age': 50},
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 2, 'total_demographic_age': 60},
        {'state': 'karnataka', 'district': 'mysore', 'pincode': 570001,
         'year': 2024, 'month': 1, 'day': 1, 'total_demographic_age': 30},
    ])


@pytest.fixture
def sample_biometric_df():
    """Create a sample biometric DataFrame for testing."""
    return pd.DataFrame([
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 1, 'total_biometric_age': 25},
        {'state': 'karnataka', 'district': 'bangalore', 'pincode': 560001,
         'year': 2024, 'month': 1, 'day': 2, 'total_biometric_age': 30},
        {'state': 'karnataka', 'district': 'mysore', 'pincode': 570001,
         'year': 2024, 'month': 1, 'day': 1, 'total_biometric_age': 15},
    ])


@pytest.fixture
def sample_alri_result():
    """Create a sample ALRI result for testing."""
    return {
        'district': 'bangalore',
        'state': 'karnataka',
        'alri_score': 45.5,
        'coverage_risk': 0.4,
        'instability_risk': 0.5,
        'biometric_risk': 0.3,
        'anomaly_factor': 0.2,
        'reason_codes': ['High_Address_Churn', 'Low_Child_Enrolment'],
        'recommendations': ['SMS/IVR Campaign', 'School Enrollment Drive'],
        'computed_at': '2024-01-15T10:30:00'
    }


@pytest.fixture
def sample_time_series():
    """Create a sample time series for anomaly detection testing."""
    import numpy as np
    np.random.seed(42)
    # 24 months of data with trend and seasonality
    t = np.arange(24)
    trend = 1000 + 10 * t
    seasonal = 100 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 20, 24)
    return pd.Series(trend + seasonal + noise, name='value')
