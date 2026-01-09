"""Property-based tests for anomaly detection correctness.

Feature: aadhaar-sentinel, Properties 8-9: Anomaly Detection Properties
Validates: Requirements 5.1, 5.2
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.anomaly.detector import (
    STLAnomalyDetector,
    STLDecomposition,
    AnomalyResult,
)


# =============================================================================
# Custom Strategies for Anomaly Detection Tests
# =============================================================================

@st.composite
def time_series_strategy(draw, min_length=24, max_length=48):
    """
    Generate a valid time series for STL decomposition.
    
    The series has:
    - Sufficient length for STL (at least 2 * period)
    - Trend component
    - Seasonal component (period=12)
    - Random noise
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    # Base value
    base = draw(st.floats(min_value=500, max_value=5000, allow_nan=False))
    
    # Trend slope (can be positive, negative, or flat)
    trend_slope = draw(st.floats(min_value=-20, max_value=20, allow_nan=False))
    
    # Seasonal amplitude
    seasonal_amp = draw(st.floats(min_value=10, max_value=200, allow_nan=False))
    
    # Noise level
    noise_level = draw(st.floats(min_value=5, max_value=50, allow_nan=False))
    
    values = []
    for i in range(length):
        trend = trend_slope * i
        seasonal = seasonal_amp * np.sin(2 * np.pi * i / 12)
        noise = draw(st.floats(min_value=-noise_level, max_value=noise_level, allow_nan=False))
        value = base + trend + seasonal + noise
        values.append(max(1, value))  # Ensure positive values
    
    return pd.Series(values, name='value')


@st.composite
def time_series_with_anomaly_strategy(draw, min_length=24, max_length=48):
    """
    Generate a time series with at least one guaranteed anomaly.
    
    Injects a spike or drop that exceeds 3 standard deviations.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    # Base value and parameters
    base = draw(st.floats(min_value=500, max_value=2000, allow_nan=False))
    noise_level = draw(st.floats(min_value=5, max_value=30, allow_nan=False))
    
    values = []
    for i in range(length):
        trend = 2 * i
        seasonal = 50 * np.sin(2 * np.pi * i / 12)
        noise = draw(st.floats(min_value=-noise_level, max_value=noise_level, allow_nan=False))
        value = base + trend + seasonal + noise
        values.append(max(1, value))
    
    # Inject anomaly at a random position (not at edges)
    anomaly_idx = draw(st.integers(min_value=6, max_value=length - 6))
    
    # Calculate what would be a significant anomaly (> 3 sigma)
    std_estimate = noise_level * 1.5  # Rough estimate
    anomaly_magnitude = std_estimate * draw(st.floats(min_value=4.0, max_value=8.0, allow_nan=False))
    
    # Spike or drop
    is_spike = draw(st.booleans())
    if is_spike:
        values[anomaly_idx] += anomaly_magnitude
    else:
        values[anomaly_idx] = max(1, values[anomaly_idx] - anomaly_magnitude)
    
    return pd.Series(values, name='value'), anomaly_idx, is_spike


@st.composite
def zscore_strategy(draw):
    """Generate z-scores for testing anomaly classification."""
    return draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))


# =============================================================================
# Property 8: Anomaly Detection Threshold
# =============================================================================

class TestAnomalyDetectionThreshold:
    """Property 8: Anomaly Detection Threshold
    
    For any time-series residual value exceeding 3 standard deviations from
    the mean, the Anomaly_Detector SHALL flag it as an anomaly.
    
    Validates: Requirements 5.2
    """
    
    @given(zscore=st.floats(min_value=3.1, max_value=10.0, allow_nan=False))
    @settings(max_examples=100)
    def test_high_zscore_flagged_as_anomaly(self, zscore):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any z-score > 3.0, the point SHALL be classified as an anomaly.
        
        Validates: Requirements 5.2
        """
        detector = STLAnomalyDetector(zscore_threshold=3.0)
        
        # Test the classification logic directly
        anomaly_type = detector._classify_anomaly_type(zscore)
        anomaly_score = detector._zscore_to_anomaly_score(zscore)
        
        # High positive z-score should be flagged as spike
        assert anomaly_type == 'spike', \
            f"Z-score {zscore} should be classified as 'spike', got '{anomaly_type}'"
        
        # Anomaly score should be > 0.5 for values above threshold
        assert anomaly_score > 0.5, \
            f"Anomaly score {anomaly_score} should be > 0.5 for z-score {zscore}"
    
    @given(zscore=st.floats(min_value=-10.0, max_value=-3.1, allow_nan=False))
    @settings(max_examples=100)
    def test_low_zscore_flagged_as_anomaly(self, zscore):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any z-score < -3.0, the point SHALL be classified as an anomaly.
        
        Validates: Requirements 5.2
        """
        detector = STLAnomalyDetector(zscore_threshold=3.0)
        
        anomaly_type = detector._classify_anomaly_type(zscore)
        anomaly_score = detector._zscore_to_anomaly_score(zscore)
        
        # High negative z-score should be flagged as drop
        assert anomaly_type == 'drop', \
            f"Z-score {zscore} should be classified as 'drop', got '{anomaly_type}'"
        
        # Anomaly score should be > 0.5 for values above threshold
        assert anomaly_score > 0.5, \
            f"Anomaly score {anomaly_score} should be > 0.5 for z-score {zscore}"
    
    @given(zscore=st.floats(min_value=-2.9, max_value=2.9, allow_nan=False))
    @settings(max_examples=100)
    def test_normal_zscore_not_flagged(self, zscore):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any z-score within [-3.0, 3.0], the point SHALL NOT be classified
        as an anomaly.
        
        Validates: Requirements 5.2
        """
        detector = STLAnomalyDetector(zscore_threshold=3.0)
        
        anomaly_type = detector._classify_anomaly_type(zscore)
        
        # Normal z-score should not be flagged
        assert anomaly_type == 'normal', \
            f"Z-score {zscore} should be classified as 'normal', got '{anomaly_type}'"
    
    @given(
        threshold=st.floats(min_value=1.5, max_value=5.0, allow_nan=False),
        zscore_above=st.floats(min_value=0.1, max_value=5.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_configurable_threshold(self, threshold, zscore_above):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any configurable threshold T, z-scores > T SHALL be flagged.
        
        Validates: Requirements 5.2
        """
        detector = STLAnomalyDetector(zscore_threshold=threshold)
        
        # Z-score above threshold
        zscore = threshold + zscore_above
        anomaly_type = detector._classify_anomaly_type(zscore)
        
        assert anomaly_type == 'spike', \
            f"Z-score {zscore} (threshold={threshold}) should be 'spike'"
        
        # Z-score below threshold
        zscore_below = threshold - 0.1
        if zscore_below > 0:
            anomaly_type_below = detector._classify_anomaly_type(zscore_below)
            assert anomaly_type_below == 'normal', \
                f"Z-score {zscore_below} (threshold={threshold}) should be 'normal'"


# =============================================================================
# Property 9: STL Decomposition Reconstruction
# =============================================================================

class TestSTLDecompositionReconstruction:
    """Property 9: STL Decomposition Reconstruction
    
    For any time-series input, the sum of trend, seasonal, and residual
    components from STL decomposition SHALL reconstruct the original series
    (within floating-point tolerance).
    
    Validates: Requirements 5.1
    """
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_reconstruction_equals_original(self, series):
        """
        Feature: aadhaar-sentinel, Property 9: STL Decomposition Reconstruction
        
        For any valid time series, trend + seasonal + residual SHALL equal
        the original series within floating-point tolerance.
        
        Validates: Requirements 5.1
        """
        detector = STLAnomalyDetector(period=12)
        
        # Perform decomposition
        decomposition = detector.decompose(series)
        
        # Reconstruct
        reconstructed = decomposition.reconstruct()
        
        # Compare with original
        original = decomposition.original
        
        # Check reconstruction matches original within tolerance
        np.testing.assert_allclose(
            reconstructed.values,
            original.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Reconstructed series does not match original"
        )
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_decomposition_components_same_length(self, series):
        """
        Feature: aadhaar-sentinel, Property 9: STL Decomposition Reconstruction
        
        All decomposition components SHALL have the same length as the original.
        
        Validates: Requirements 5.1
        """
        detector = STLAnomalyDetector(period=12)
        decomposition = detector.decompose(series)
        
        original_len = len(decomposition.original)
        
        assert len(decomposition.trend) == original_len, \
            f"Trend length {len(decomposition.trend)} != original {original_len}"
        assert len(decomposition.seasonal) == original_len, \
            f"Seasonal length {len(decomposition.seasonal)} != original {original_len}"
        assert len(decomposition.residual) == original_len, \
            f"Residual length {len(decomposition.residual)} != original {original_len}"
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_residual_mean_near_zero(self, series):
        """
        Feature: aadhaar-sentinel, Property 9: STL Decomposition Reconstruction
        
        The residual component SHALL have a mean close to zero.
        
        Validates: Requirements 5.1
        """
        detector = STLAnomalyDetector(period=12)
        decomposition = detector.decompose(series)
        
        residual_mean = decomposition.residual.mean()
        
        # Residual mean should be close to zero (within reasonable tolerance)
        assert abs(residual_mean) < 50, \
            f"Residual mean {residual_mean} should be close to zero"


# =============================================================================
# Additional Anomaly Detection Properties
# =============================================================================

class TestAnomalyScoreRange:
    """Test that anomaly scores are always in [0, 1] range."""
    
    @given(zscore=zscore_strategy())
    @settings(max_examples=100)
    def test_anomaly_score_in_unit_range(self, zscore):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any z-score, the anomaly score SHALL be in [0, 1].
        
        Validates: Requirements 5.3
        """
        detector = STLAnomalyDetector()
        anomaly_score = detector._zscore_to_anomaly_score(zscore)
        
        assert 0.0 <= anomaly_score <= 1.0, \
            f"Anomaly score {anomaly_score} not in [0, 1] for z-score {zscore}"
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_anomaly_factor_in_unit_range(self, series):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        For any time series, the computed anomaly factor SHALL be in [0, 1].
        
        Validates: Requirements 5.3
        """
        detector = STLAnomalyDetector()
        anomaly_factor = detector.compute_anomaly_factor(time_series=series)
        
        assert 0.0 <= anomaly_factor <= 1.0, \
            f"Anomaly factor {anomaly_factor} not in [0, 1]"


class TestAnomalyResultConsistency:
    """Test consistency of AnomalyResult objects."""
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_anomaly_results_count_matches_series_length(self, series):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        The number of AnomalyResult objects SHALL equal the series length.
        
        Validates: Requirements 5.2
        """
        detector = STLAnomalyDetector()
        anomalies = detector.detect_anomalies(series)
        
        # Account for potential NaN removal in decomposition
        expected_length = len(series.dropna())
        
        assert len(anomalies) == expected_length, \
            f"Got {len(anomalies)} results, expected {expected_length}"
    
    @given(series=time_series_strategy(min_length=24, max_length=48))
    @settings(max_examples=100)
    def test_anomaly_type_consistency(self, series):
        """
        Feature: aadhaar-sentinel, Property 8: Anomaly Detection Threshold
        
        Anomaly type SHALL be consistent with z-score sign.
        
        Validates: Requirements 5.4
        """
        detector = STLAnomalyDetector()
        anomalies = detector.detect_anomalies(series)
        
        for result in anomalies:
            if result.is_anomaly:
                if result.residual_zscore > 0:
                    assert result.anomaly_type == 'spike', \
                        f"Positive z-score {result.residual_zscore} should be 'spike'"
                else:
                    assert result.anomaly_type == 'drop', \
                        f"Negative z-score {result.residual_zscore} should be 'drop'"
            else:
                assert result.anomaly_type == 'normal', \
                    f"Non-anomaly should be 'normal', got '{result.anomaly_type}'"
