"""Property-based tests for forecasting correctness.

Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
Validates: Requirements 9.2
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from datetime import datetime, timedelta

from src.forecasting.forecaster import TimeSeriesForecaster, ForecastResult
from tests.conftest import (
    district_strategy,
    state_strategy,
)


# =============================================================================
# Custom Strategies for Forecasting Tests
# =============================================================================

@st.composite
def time_series_data_strategy(draw, min_months=12, max_months=36):
    """
    Generate valid time-series data for forecasting.
    
    Creates a DataFrame with 'ds' (datetime) and 'y' (value) columns
    suitable for Prophet forecasting.
    """
    num_months = draw(st.integers(min_value=min_months, max_value=max_months))
    
    # Start date
    start_year = draw(st.integers(min_value=2020, max_value=2023))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_date = datetime(start_year, start_month, 1)
    
    # Generate dates
    dates = [start_date + timedelta(days=30 * i) for i in range(num_months)]
    
    # Generate values with trend and seasonality
    base_value = draw(st.floats(min_value=500, max_value=10000, allow_nan=False))
    trend_slope = draw(st.floats(min_value=-50, max_value=50, allow_nan=False))
    seasonal_amplitude = draw(st.floats(min_value=0, max_value=200, allow_nan=False))
    
    values = []
    for i in range(num_months):
        # Base + trend + seasonality + noise
        trend = trend_slope * i
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * i / 12)
        noise = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
        value = max(0, base_value + trend + seasonal + noise)
        values.append(value)
    
    return pd.DataFrame({
        'ds': dates,
        'y': values
    })


@st.composite
def forecast_result_strategy(draw):
    """Generate a valid ForecastResult for testing."""
    horizon = draw(st.integers(min_value=3, max_value=6))
    
    # Generate base forecast values
    base_value = draw(st.floats(min_value=100, max_value=10000, allow_nan=False))
    forecast_values = []
    for i in range(horizon):
        change = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))
        forecast_values.append(max(0, base_value + change))
    
    # Generate confidence intervals
    margin = draw(st.floats(min_value=10, max_value=500, allow_nan=False))
    lower_bound = [max(0, v - margin) for v in forecast_values]
    upper_bound = [v + margin for v in forecast_values]
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    forecast_dates = [start_date + timedelta(days=30 * i) for i in range(horizon)]
    
    return ForecastResult(
        district=draw(district_strategy),
        metric=draw(st.sampled_from(['enrollment', 'demographic', 'biometric'])),
        forecast_values=forecast_values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        forecast_dates=forecast_dates,
        trend=draw(st.sampled_from(['increasing', 'decreasing', 'stable']))
    )


class TestForecastConfidenceInterval:
    """Property 14: Forecast Confidence Interval
    
    For any forecast result, the lower_bound values SHALL be less than or equal
    to forecast_values, and forecast_values SHALL be less than or equal to
    upper_bound values.
    
    Validates: Requirements 9.2
    """
    
    @given(data=time_series_data_strategy(min_months=12, max_months=24))
    @settings(max_examples=5, deadline=120000)  # 60 second deadline for Prophet fitting
    def test_confidence_interval_bounds(self, data):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        For any valid time-series data, the forecast confidence intervals
        SHALL satisfy: lower_bound <= forecast_values <= upper_bound.
        
        Validates: Requirements 9.2
        """
        forecaster = TimeSeriesForecaster(horizon_months=3)
        
        try:
            forecaster.fit(data, district='test_district', metric='enrollment')
            result = forecaster.predict()
            
            # Verify confidence interval property
            for i in range(len(result.forecast_values)):
                lower = result.lower_bound[i]
                forecast = result.forecast_values[i]
                upper = result.upper_bound[i]
                
                assert lower <= forecast, \
                    f"Lower bound {lower} > forecast {forecast} at index {i}"
                assert forecast <= upper, \
                    f"Forecast {forecast} > upper bound {upper} at index {i}"
                
        except ValueError as e:
            # Skip if insufficient data (this is expected behavior)
            if "Insufficient data" in str(e):
                assume(False)
            raise
    
    @given(result=forecast_result_strategy())
    @settings(max_examples=50)
    def test_generated_forecast_result_bounds(self, result):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        For any generated ForecastResult, confidence intervals SHALL be valid.
        
        Validates: Requirements 9.2
        """
        # Verify confidence interval property for generated results
        for i in range(len(result.forecast_values)):
            lower = result.lower_bound[i]
            forecast = result.forecast_values[i]
            upper = result.upper_bound[i]
            
            assert lower <= forecast, \
                f"Lower bound {lower} > forecast {forecast} at index {i}"
            assert forecast <= upper, \
                f"Forecast {forecast} > upper bound {upper} at index {i}"


class TestForecastNonNegativity:
    """Additional property: Forecast values should be non-negative.
    
    Since we're forecasting counts (enrollments, updates), values should
    never be negative.
    """
    
    @given(data=time_series_data_strategy(min_months=12, max_months=24))
    @settings(max_examples=5, deadline=120000)
    def test_forecast_values_non_negative(self, data):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        For any forecast, all values (including bounds) SHALL be non-negative.
        
        Validates: Requirements 9.2
        """
        forecaster = TimeSeriesForecaster(horizon_months=3)
        
        try:
            forecaster.fit(data, district='test_district', metric='enrollment')
            result = forecaster.predict()
            
            # Verify non-negativity
            for i in range(len(result.forecast_values)):
                assert result.forecast_values[i] >= 0, \
                    f"Forecast value {result.forecast_values[i]} is negative at index {i}"
                assert result.lower_bound[i] >= 0, \
                    f"Lower bound {result.lower_bound[i]} is negative at index {i}"
                assert result.upper_bound[i] >= 0, \
                    f"Upper bound {result.upper_bound[i]} is negative at index {i}"
                
        except ValueError as e:
            if "Insufficient data" in str(e):
                assume(False)
            raise


class TestForecastTrendDetection:
    """Test trend detection functionality."""
    
    @given(data=time_series_data_strategy(min_months=12, max_months=24))
    @settings(max_examples=5, deadline=120000)
    def test_trend_is_valid_value(self, data):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        For any forecast, trend SHALL be one of 'increasing', 'decreasing', 'stable'.
        
        Validates: Requirements 9.3
        """
        forecaster = TimeSeriesForecaster(horizon_months=3)
        
        try:
            forecaster.fit(data, district='test_district', metric='enrollment')
            result = forecaster.predict()
            
            valid_trends = ['increasing', 'decreasing', 'stable']
            assert result.trend in valid_trends, \
                f"Trend '{result.trend}' not in valid values {valid_trends}"
                
        except ValueError as e:
            if "Insufficient data" in str(e):
                assume(False)
            raise
    
    @given(data=time_series_data_strategy(min_months=12, max_months=24))
    @settings(max_examples=5, deadline=120000)
    def test_detect_declining_trend_consistency(self, data):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        detect_declining_trend() SHALL return True iff trend == 'decreasing'.
        
        Validates: Requirements 9.3
        """
        forecaster = TimeSeriesForecaster(horizon_months=3)
        
        try:
            forecaster.fit(data, district='test_district', metric='enrollment')
            result = forecaster.predict()
            
            is_declining = forecaster.detect_declining_trend(result)
            
            if result.trend == 'decreasing':
                assert is_declining, \
                    "detect_declining_trend should return True for decreasing trend"
            else:
                assert not is_declining, \
                    f"detect_declining_trend should return False for {result.trend} trend"
                
        except ValueError as e:
            if "Insufficient data" in str(e):
                assume(False)
            raise


class TestForecastHorizon:
    """Test forecast horizon configuration."""
    
    @given(
        data=time_series_data_strategy(min_months=12, max_months=24),
        horizon=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=5, deadline=120000)
    def test_forecast_length_matches_horizon(self, data, horizon):
        """
        Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
        
        Forecast length SHALL match the configured horizon_months.
        
        Validates: Requirements 9.1
        """
        forecaster = TimeSeriesForecaster(horizon_months=horizon)
        
        try:
            forecaster.fit(data, district='test_district', metric='enrollment')
            result = forecaster.predict()
            
            assert len(result.forecast_values) == horizon, \
                f"Forecast length {len(result.forecast_values)} != horizon {horizon}"
            assert len(result.lower_bound) == horizon, \
                f"Lower bound length {len(result.lower_bound)} != horizon {horizon}"
            assert len(result.upper_bound) == horizon, \
                f"Upper bound length {len(result.upper_bound)} != horizon {horizon}"
            assert len(result.forecast_dates) == horizon, \
                f"Dates length {len(result.forecast_dates)} != horizon {horizon}"
                
        except ValueError as e:
            if "Insufficient data" in str(e):
                assume(False)
            raise
