"""Property-based tests for ALRI scoring correctness.

Feature: aadhaar-sentinel, Properties 3-7: Scoring Properties
Validates: Requirements 2.3, 2.4, 3.3, 3.4, 4.3, 5.3, 6.1, 6.2
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.scoring.alri_calculator import ALRIWeights, ALRIResult, ALRICalculator
from tests.conftest import (
    sub_score_strategy,
    sub_scores_strategy,
    alri_weights_strategy,
    state_strategy,
    district_strategy,
    count_strategy,
)


# =============================================================================
# Custom Strategies for Scoring Tests
# =============================================================================

@st.composite
def valid_weights_strategy(draw):
    """Generate valid ALRI weights (non-negative, sum > 0)."""
    coverage = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    instability = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    biometric = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    anomaly = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    
    # Ensure at least one weight is positive
    total = coverage + instability + biometric + anomaly
    assume(total > 0)
    
    return ALRIWeights(
        coverage=coverage,
        instability=instability,
        biometric=biometric,
        anomaly=anomaly
    )


@st.composite
def district_enrollment_data_strategy(draw):
    """Generate district enrollment data for coverage risk testing."""
    num_records = draw(st.integers(min_value=3, max_value=20))
    
    # Generate enrollment values
    base_enrollment = draw(st.integers(min_value=100, max_value=10000))
    enrollments = [
        max(0, base_enrollment + draw(st.integers(min_value=-500, max_value=500)))
        for _ in range(num_records)
    ]
    
    return pd.DataFrame({
        'total_enrollment_age': enrollments
    })


@st.composite
def district_demographic_data_strategy(draw):
    """Generate district data for instability risk testing."""
    num_records = draw(st.integers(min_value=3, max_value=20))
    
    base_enrollment = draw(st.integers(min_value=1000, max_value=50000))
    base_demographic = draw(st.integers(min_value=10, max_value=1000))
    
    enrollments = [
        max(1, base_enrollment + draw(st.integers(min_value=-200, max_value=200)))
        for _ in range(num_records)
    ]
    demographics = [
        max(0, base_demographic + draw(st.integers(min_value=-100, max_value=100)))
        for _ in range(num_records)
    ]
    
    return pd.DataFrame({
        'total_enrollment_age': enrollments,
        'total_demographic_age': demographics
    })


@st.composite
def district_biometric_data_strategy(draw):
    """Generate district data for biometric risk testing."""
    num_records = draw(st.integers(min_value=3, max_value=20))
    
    base_biometric = draw(st.integers(min_value=50, max_value=5000))
    biometrics = [
        max(0, base_biometric + draw(st.integers(min_value=-200, max_value=200)))
        for _ in range(num_records)
    ]
    
    return pd.DataFrame({
        'total_biometric_age': biometrics
    })



class TestSubScoreRangeInvariant:
    """Property 3: Sub-score Range Invariant
    
    For any computed sub-score (Coverage_Risk, Instability_Risk, Biometric_Risk,
    Anomaly_Factor), the value SHALL be in the range [0, 1].
    
    Validates: Requirements 2.3, 3.3, 4.3, 5.3
    """
    
    @given(data=district_enrollment_data_strategy())
    @settings(max_examples=100)
    def test_coverage_risk_in_unit_range(self, data):
        """
        Feature: aadhaar-sentinel, Property 3: Sub-score Range Invariant
        
        For any district enrollment data, Coverage_Risk SHALL be in [0, 1].
        
        Validates: Requirements 2.3
        """
        calculator = ALRICalculator()
        coverage_risk = calculator.compute_coverage_risk(data)
        
        assert 0.0 <= coverage_risk <= 1.0, \
            f"Coverage risk {coverage_risk} not in [0, 1]"
    
    @given(data=district_demographic_data_strategy())
    @settings(max_examples=100)
    def test_instability_risk_in_unit_range(self, data):
        """
        Feature: aadhaar-sentinel, Property 3: Sub-score Range Invariant
        
        For any district demographic data, Instability_Risk SHALL be in [0, 1].
        
        Validates: Requirements 3.3
        """
        calculator = ALRICalculator()
        instability_risk = calculator.compute_instability_risk(data)
        
        assert 0.0 <= instability_risk <= 1.0, \
            f"Instability risk {instability_risk} not in [0, 1]"
    
    @given(data=district_biometric_data_strategy())
    @settings(max_examples=100)
    def test_biometric_risk_in_unit_range(self, data):
        """
        Feature: aadhaar-sentinel, Property 3: Sub-score Range Invariant
        
        For any district biometric data, Biometric_Risk SHALL be in [0, 1].
        
        Validates: Requirements 4.3
        """
        calculator = ALRICalculator()
        biometric_risk = calculator.compute_biometric_risk(data)
        
        assert 0.0 <= biometric_risk <= 1.0, \
            f"Biometric risk {biometric_risk} not in [0, 1]"
    
    @given(anomaly_score=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    @settings(max_examples=100)
    def test_anomaly_factor_in_unit_range(self, anomaly_score):
        """
        Feature: aadhaar-sentinel, Property 3: Sub-score Range Invariant
        
        For any input anomaly score, Anomaly_Factor SHALL be clipped to [0, 1].
        
        Validates: Requirements 5.3
        """
        calculator = ALRICalculator()
        # Pass empty DataFrame with pre-computed anomaly score
        anomaly_factor = calculator.compute_anomaly_factor(
            pd.DataFrame(), 
            anomaly_score=anomaly_score
        )
        
        assert 0.0 <= anomaly_factor <= 1.0, \
            f"Anomaly factor {anomaly_factor} not in [0, 1]"


class TestALRIScoreRangeInvariant:
    """Property 4: ALRI Score Range Invariant
    
    For any computed ALRI score, the value SHALL be in the range [0, 100].
    
    Validates: Requirements 6.2
    """
    
    @given(
        coverage=sub_score_strategy,
        instability=sub_score_strategy,
        biometric=sub_score_strategy,
        anomaly=sub_score_strategy
    )
    @settings(max_examples=100)
    def test_alri_score_in_hundred_range(self, coverage, instability, biometric, anomaly):
        """
        Feature: aadhaar-sentinel, Property 4: ALRI Score Range Invariant
        
        For any valid sub-scores in [0, 1], ALRI score SHALL be in [0, 100].
        
        Validates: Requirements 6.2
        """
        calculator = ALRICalculator()
        result = calculator.compute_alri_from_subscores(
            coverage_risk=coverage,
            instability_risk=instability,
            biometric_risk=biometric,
            anomaly_factor=anomaly
        )
        
        assert 0.0 <= result.alri_score <= 100.0, \
            f"ALRI score {result.alri_score} not in [0, 100]"
    
    @given(weights=valid_weights_strategy(), sub_scores=sub_scores_strategy())
    @settings(max_examples=100)
    def test_alri_score_with_custom_weights(self, weights, sub_scores):
        """
        Feature: aadhaar-sentinel, Property 4: ALRI Score Range Invariant
        
        For any valid weights and sub-scores, ALRI score SHALL be in [0, 100].
        
        Validates: Requirements 6.2
        """
        calculator = ALRICalculator(weights=weights)
        result = calculator.compute_alri_from_subscores(
            coverage_risk=sub_scores['coverage'],
            instability_risk=sub_scores['instability'],
            biometric_risk=sub_scores['biometric'],
            anomaly_factor=sub_scores['anomaly']
        )
        
        assert 0.0 <= result.alri_score <= 100.0, \
            f"ALRI score {result.alri_score} not in [0, 100]"



class TestALRIWeightedSumCorrectness:
    """Property 5: ALRI Weighted Sum Correctness
    
    For any set of valid sub-scores (C, D, B, A in [0,1]) and weights 
    (w1, w2, w3, w4 summing to 1.0), the ALRI score SHALL equal 
    (w1×C + w2×D + w3×B + w4×A) × 100.
    
    Validates: Requirements 6.1
    """
    
    @given(
        coverage=sub_score_strategy,
        instability=sub_score_strategy,
        biometric=sub_score_strategy,
        anomaly=sub_score_strategy
    )
    @settings(max_examples=100)
    def test_weighted_sum_with_default_weights(self, coverage, instability, biometric, anomaly):
        """
        Feature: aadhaar-sentinel, Property 5: ALRI Weighted Sum Correctness
        
        For default weights (0.30, 0.30, 0.30, 0.10), ALRI SHALL equal
        the weighted sum × 100.
        
        Validates: Requirements 6.1
        """
        calculator = ALRICalculator()
        result = calculator.compute_alri_from_subscores(
            coverage_risk=coverage,
            instability_risk=instability,
            biometric_risk=biometric,
            anomaly_factor=anomaly
        )
        
        # Calculate expected ALRI
        expected_weighted_sum = (
            0.30 * coverage +
            0.30 * instability +
            0.30 * biometric +
            0.10 * anomaly
        )
        expected_alri = expected_weighted_sum * 100
        
        assert abs(result.alri_score - expected_alri) < 1e-6, \
            f"ALRI {result.alri_score} != expected {expected_alri}"
    
    @given(weights=alri_weights_strategy(), sub_scores=sub_scores_strategy())
    @settings(max_examples=100)
    def test_weighted_sum_with_normalized_weights(self, weights, sub_scores):
        """
        Feature: aadhaar-sentinel, Property 5: ALRI Weighted Sum Correctness
        
        For any normalized weights (sum to 1.0), ALRI SHALL equal
        the weighted sum × 100.
        
        Validates: Requirements 6.1
        """
        # Create ALRIWeights from normalized weights dict
        alri_weights = ALRIWeights(
            coverage=weights['coverage'],
            instability=weights['instability'],
            biometric=weights['biometric'],
            anomaly=weights['anomaly']
        )
        
        calculator = ALRICalculator(weights=alri_weights)
        result = calculator.compute_alri_from_subscores(
            coverage_risk=sub_scores['coverage'],
            instability_risk=sub_scores['instability'],
            biometric_risk=sub_scores['biometric'],
            anomaly_factor=sub_scores['anomaly']
        )
        
        # Calculate expected ALRI with normalized weights
        weighted_sum = (
            weights['coverage'] * sub_scores['coverage'] +
            weights['instability'] * sub_scores['instability'] +
            weights['biometric'] * sub_scores['biometric'] +
            weights['anomaly'] * sub_scores['anomaly']
        )
        weight_total = sum(weights.values())
        expected_alri = (weighted_sum / weight_total) * 100
        
        assert abs(result.alri_score - expected_alri) < 1e-6, \
            f"ALRI {result.alri_score} != expected {expected_alri}"


class TestCoverageRiskMonotonicity:
    """Property 6: Coverage Risk Monotonicity
    
    For any two districts where district A has lower child enrollment proportion
    than district B, district A SHALL have a higher or equal Coverage_Risk score
    than district B.
    
    Validates: Requirements 2.4
    """
    
    @given(
        base_enrollment=st.integers(min_value=500, max_value=5000),
        low_factor=st.floats(min_value=0.1, max_value=0.5, allow_nan=False),
        high_factor=st.floats(min_value=0.6, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_lower_enrollment_yields_higher_risk(self, base_enrollment, low_factor, high_factor):
        """
        Feature: aadhaar-sentinel, Property 6: Coverage Risk Monotonicity
        
        For two districts with different enrollment levels relative to baseline,
        the district with lower enrollment SHALL have higher or equal risk.
        
        Validates: Requirements 2.4
        """
        assume(low_factor < high_factor)
        
        calculator = ALRICalculator()
        
        # District A: lower enrollment
        low_enrollment = int(base_enrollment * low_factor)
        data_low = pd.DataFrame({
            'total_enrollment_age': [low_enrollment] * 5
        })
        
        # District B: higher enrollment
        high_enrollment = int(base_enrollment * high_factor)
        data_high = pd.DataFrame({
            'total_enrollment_age': [high_enrollment] * 5
        })
        
        # Use same baseline for fair comparison
        baseline = base_enrollment
        
        risk_low = calculator.compute_coverage_risk(data_low, baseline_enrollment=baseline)
        risk_high = calculator.compute_coverage_risk(data_high, baseline_enrollment=baseline)
        
        # Lower enrollment should yield higher or equal risk
        assert risk_low >= risk_high - 0.01, \
            f"Lower enrollment risk {risk_low} should be >= higher enrollment risk {risk_high}"



class TestInstabilityRiskMonotonicity:
    """Property 7: Instability Risk Monotonicity
    
    For any two districts where district A has higher demographic update frequency
    than district B, district A SHALL have a higher or equal Data_Instability_Risk
    score than district B.
    
    Validates: Requirements 3.4
    """
    
    @given(
        base_enrollment=st.integers(min_value=5000, max_value=50000),
        low_update_rate=st.floats(min_value=0.001, max_value=0.02, allow_nan=False),
        high_update_rate=st.floats(min_value=0.03, max_value=0.1, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_higher_update_frequency_yields_higher_risk(
        self, base_enrollment, low_update_rate, high_update_rate
    ):
        """
        Feature: aadhaar-sentinel, Property 7: Instability Risk Monotonicity
        
        For two districts with different demographic update frequencies,
        the district with higher update frequency SHALL have higher or equal risk.
        
        Validates: Requirements 3.4
        """
        assume(low_update_rate < high_update_rate)
        
        calculator = ALRICalculator()
        
        # District A: higher update frequency
        high_updates = int(base_enrollment * high_update_rate)
        data_high = pd.DataFrame({
            'total_enrollment_age': [base_enrollment] * 5,
            'total_demographic_age': [high_updates] * 5
        })
        
        # District B: lower update frequency
        low_updates = int(base_enrollment * low_update_rate)
        data_low = pd.DataFrame({
            'total_enrollment_age': [base_enrollment] * 5,
            'total_demographic_age': [low_updates] * 5
        })
        
        risk_high = calculator.compute_instability_risk(data_high, baseline_enrollment=base_enrollment)
        risk_low = calculator.compute_instability_risk(data_low, baseline_enrollment=base_enrollment)
        
        # Higher update frequency should yield higher or equal risk
        assert risk_high >= risk_low - 0.01, \
            f"Higher update risk {risk_high} should be >= lower update risk {risk_low}"


class TestALRIResultDataclass:
    """Additional tests for ALRIResult dataclass validation."""
    
    @given(
        coverage=sub_score_strategy,
        instability=sub_score_strategy,
        biometric=sub_score_strategy,
        anomaly=sub_score_strategy,
        district=district_strategy,
        state=state_strategy
    )
    @settings(max_examples=100)
    def test_alri_result_stores_all_subscores(
        self, coverage, instability, biometric, anomaly, district, state
    ):
        """
        Feature: aadhaar-sentinel, Property 4: ALRI Score Range Invariant
        
        ALRIResult SHALL store all sub-scores correctly.
        
        Validates: Requirements 6.1, 6.2
        """
        calculator = ALRICalculator()
        result = calculator.compute_alri_from_subscores(
            coverage_risk=coverage,
            instability_risk=instability,
            biometric_risk=biometric,
            anomaly_factor=anomaly,
            district=district,
            state=state
        )
        
        # Verify all sub-scores are stored correctly
        assert abs(result.coverage_risk - coverage) < 1e-9
        assert abs(result.instability_risk - instability) < 1e-9
        assert abs(result.biometric_risk - biometric) < 1e-9
        assert abs(result.anomaly_factor - anomaly) < 1e-9
        assert result.district == district
        assert result.state == state
