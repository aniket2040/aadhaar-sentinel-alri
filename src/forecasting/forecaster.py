"""
Time-series forecasting module for Aadhaar Sentinel.

This module provides forecasting capabilities using Prophet for predicting
future enrollment, demographic, and biometric update volumes.

Feature: aadhaar-sentinel
Requirements: 9.1, 9.2, 9.3, 9.4
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Configure logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


@dataclass
class ForecastResult:
    """
    Result of a time-series forecast.
    
    Attributes:
        district: District name for the forecast
        metric: Type of metric ('enrollment', 'demographic', 'biometric')
        forecast_values: Predicted values for future periods
        lower_bound: Lower confidence interval values
        upper_bound: Upper confidence interval values
        forecast_dates: Dates corresponding to forecast values
        trend: Overall trend direction ('increasing', 'decreasing', 'stable')
    """
    district: str
    metric: str  # 'enrollment', 'demographic', 'biometric'
    forecast_values: List[float] = field(default_factory=list)
    lower_bound: List[float] = field(default_factory=list)
    upper_bound: List[float] = field(default_factory=list)
    forecast_dates: List[datetime] = field(default_factory=list)
    trend: str = 'stable'  # 'increasing', 'decreasing', 'stable'


class TimeSeriesForecaster:
    """
    Forecasts future volumes using Prophet.
    
    This class provides time-series forecasting capabilities for enrollment,
    demographic updates, and biometric updates. It uses Facebook Prophet
    for interpretable forecasts with confidence intervals.
    
    Feature: aadhaar-sentinel, Property 14: Forecast Confidence Interval
    Validates: Requirements 9.1, 9.2, 9.3, 9.4
    """
    
    def __init__(self, model_type: str = 'prophet', horizon_months: int = 6):
        """
        Initialize the forecaster.
        
        Args:
            model_type: Type of model to use ('prophet' supported)
            horizon_months: Number of months to forecast (3-6 recommended)
        """
        self.model_type = model_type
        self.horizon_months = max(3, min(horizon_months, 12))  # Clamp to 3-12
        self.model: Optional[Prophet] = None
        self._fitted = False
        self._district: str = ''
        self._metric: str = ''
        self._last_date: Optional[datetime] = None

    def fit(self, time_series: pd.DataFrame, district: str = '', metric: str = 'enrollment') -> None:
        """
        Fit the forecasting model to historical data.
        
        The input DataFrame should have columns:
        - 'ds': datetime column with dates
        - 'y': numeric column with values to forecast
        
        Or alternatively:
        - 'year', 'month': for date construction
        - A value column matching the metric name
        
        Args:
            time_series: DataFrame with historical time-series data
            district: District name for the forecast
            metric: Type of metric being forecast
            
        Raises:
            ValueError: If insufficient data points (< 12 months)
        """
        self._district = district
        self._metric = metric
        
        # Prepare data for Prophet
        df = self._prepare_data(time_series, metric)
        
        if len(df) < 12:
            raise ValueError(
                f"Insufficient data points for forecasting. "
                f"Need at least 12 months, got {len(df)}."
            )
        
        # Store last date for forecast generation
        self._last_date = df['ds'].max()
        
        # Initialize and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,  # 95% confidence interval
            changepoint_prior_scale=0.05  # Conservative changepoint detection
        )
        
        # Suppress Prophet's verbose output
        self.model.fit(df)
        self._fitted = True
    
    def _prepare_data(self, time_series: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Prepare data for Prophet format.
        
        Args:
            time_series: Input DataFrame
            metric: Metric name for value column lookup
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        df = time_series.copy()
        
        # Check if already in Prophet format
        if 'ds' in df.columns and 'y' in df.columns:
            return df[['ds', 'y']].dropna()
        
        # Convert year/month to datetime
        if 'year' in df.columns and 'month' in df.columns:
            df['ds'] = pd.to_datetime(
                df[['year', 'month']].assign(day=1)
            )
        elif 'date' in df.columns:
            df['ds'] = pd.to_datetime(df['date'])
        else:
            raise ValueError(
                "DataFrame must have 'ds' column, 'year'/'month' columns, "
                "or 'date' column"
            )
        
        # Find value column
        value_col = None
        possible_cols = [
            'y', 'value', metric,
            f'total_{metric}', f'total_{metric}_age',
            f'{metric}_count', 'count', 'total'
        ]
        
        for col in possible_cols:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None:
            raise ValueError(
                f"Could not find value column. Tried: {possible_cols}"
            )
        
        df['y'] = df[value_col].astype(float)
        
        # Aggregate by month if multiple entries per month
        df = df.groupby('ds')['y'].sum().reset_index()
        
        return df[['ds', 'y']].sort_values('ds').dropna()

    def predict(self) -> ForecastResult:
        """
        Generate forecast with confidence intervals.
        
        Returns:
            ForecastResult with predicted values, confidence intervals,
            dates, and trend direction.
            
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=self.horizon_months,
            freq='MS'  # Month start frequency
        )
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract only future predictions (beyond training data)
        future_forecast = forecast[forecast['ds'] > self._last_date].copy()
        
        # Ensure non-negative values (counts can't be negative)
        forecast_values = future_forecast['yhat'].clip(lower=0).tolist()
        lower_bound = future_forecast['yhat_lower'].clip(lower=0).tolist()
        upper_bound = future_forecast['yhat_upper'].clip(lower=0).tolist()
        forecast_dates = future_forecast['ds'].tolist()
        
        # Determine trend
        trend = self._determine_trend(forecast_values)
        
        return ForecastResult(
            district=self._district,
            metric=self._metric,
            forecast_values=forecast_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            forecast_dates=forecast_dates,
            trend=trend
        )
    
    def _determine_trend(self, values: List[float]) -> str:
        """
        Determine the overall trend direction from forecast values.
        
        Args:
            values: List of forecast values
            
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2)
        
        # Normalize slope by mean value to get relative change
        mean_value = np.mean(y)
        if mean_value > 0:
            relative_slope = slope / mean_value
        else:
            relative_slope = 0
        
        # Threshold for trend detection (5% change per period)
        threshold = 0.05
        
        if relative_slope > threshold:
            return 'increasing'
        elif relative_slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def detect_declining_trend(self, forecast: Optional[ForecastResult] = None) -> bool:
        """
        Flag if forecast shows declining trend for proactive intervention.
        
        Args:
            forecast: Optional ForecastResult to analyze. If None, generates
                     a new forecast from the fitted model.
                     
        Returns:
            True if trend is declining, False otherwise
            
        Raises:
            RuntimeError: If model not fitted and no forecast provided
        """
        if forecast is None:
            if not self._fitted:
                raise RuntimeError(
                    "Model must be fitted or forecast must be provided."
                )
            forecast = self.predict()
        
        return forecast.trend == 'decreasing'
    
    def fit_predict(
        self,
        time_series: pd.DataFrame,
        district: str = '',
        metric: str = 'enrollment'
    ) -> ForecastResult:
        """
        Convenience method to fit and predict in one call.
        
        Args:
            time_series: DataFrame with historical time-series data
            district: District name for the forecast
            metric: Type of metric being forecast
            
        Returns:
            ForecastResult with predictions
        """
        self.fit(time_series, district, metric)
        return self.predict()
