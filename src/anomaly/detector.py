"""
STL Anomaly Detector Module

Detects anomalies in time-series data using STL (Seasonal-Trend decomposition using Loess).
The detector decomposes time-series into trend, seasonal, and residual components,
then flags residuals exceeding a configurable z-score threshold as anomalies.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection for a single data point.
    
    Attributes:
        is_anomaly: Whether this point is flagged as an anomaly
        anomaly_score: Score between 0 (normal) and 1 (extreme anomaly)
        anomaly_type: Type of anomaly ('spike', 'drop', 'normal')
        residual_zscore: Z-score of the residual component
        timestamp: Timestamp or index of the data point
    """
    is_anomaly: bool
    anomaly_score: float  # 0-1
    anomaly_type: str  # 'spike', 'drop', 'normal'
    residual_zscore: float
    timestamp: Optional[datetime] = None
    index: Optional[int] = None
    
    def __post_init__(self):
        """Validate anomaly score is in [0, 1]."""
        if not 0 <= self.anomaly_score <= 1:
            raise ValueError(f"anomaly_score must be in [0, 1], got {self.anomaly_score}")


@dataclass
class STLDecomposition:
    """
    Result of STL decomposition.
    
    Attributes:
        trend: Trend component of the time series
        seasonal: Seasonal component of the time series
        residual: Residual component of the time series
        original: Original time series
    """
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    original: pd.Series
    
    def reconstruct(self) -> pd.Series:
        """
        Reconstruct the original series from components.
        
        Property 9: STL Decomposition Reconstruction
        For any time-series input, the sum of trend, seasonal, and residual
        components SHALL reconstruct the original series (within floating-point tolerance).
        
        Returns:
            Reconstructed series (trend + seasonal + residual)
        """
        return self.trend + self.seasonal + self.residual


class STLAnomalyDetector:
    """
    Detects anomalies using STL decomposition.
    
    The detector applies STL decomposition to separate trend, seasonal, and
    residual components, then flags residuals exceeding a z-score threshold
    as anomalies.
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    
    Attributes:
        zscore_threshold: Z-score threshold for anomaly detection (default: 3.0)
        period: Seasonal period for STL decomposition (default: 12 for monthly data)
    """
    
    def __init__(
        self,
        zscore_threshold: float = 3.0,
        period: int = 12,
        robust: bool = True
    ):
        """
        Initialize the STL Anomaly Detector.
        
        Args:
            zscore_threshold: Z-score threshold for flagging anomalies (default: 3.0)
            period: Seasonal period for STL decomposition (default: 12 for monthly)
            robust: Whether to use robust STL fitting (default: True)
        """
        if zscore_threshold <= 0:
            raise ValueError("zscore_threshold must be positive")
        if period < 2:
            raise ValueError("period must be at least 2")
            
        self.zscore_threshold = zscore_threshold
        self.period = period
        self.robust = robust
        self._last_decomposition: Optional[STLDecomposition] = None
    
    def decompose(self, time_series: pd.Series) -> STLDecomposition:
        """
        Apply STL decomposition to extract trend, seasonal, and residual components.
        
        Requirement 5.1: Apply STL_Decomposition to separate trend, seasonal,
        and residual components.
        
        Args:
            time_series: Time series data as pandas Series
            
        Returns:
            STLDecomposition containing trend, seasonal, and residual components
            
        Raises:
            ValueError: If time series is too short for decomposition
        """
        if len(time_series) < 2 * self.period:
            raise ValueError(
                f"Time series too short for STL decomposition. "
                f"Need at least {2 * self.period} points, got {len(time_series)}"
            )
        
        # Ensure series has no NaN values
        series_clean = time_series.dropna()
        if len(series_clean) < 2 * self.period:
            raise ValueError(
                f"Time series has too many NaN values. "
                f"Need at least {2 * self.period} non-NaN points"
            )
        
        # Apply STL decomposition
        stl = STL(
            series_clean,
            period=self.period,
            robust=self.robust
        )
        result = stl.fit()
        
        decomposition = STLDecomposition(
            trend=result.trend,
            seasonal=result.seasonal,
            residual=result.resid,
            original=series_clean
        )
        
        self._last_decomposition = decomposition
        return decomposition
    
    def _compute_zscore(self, residuals: pd.Series) -> pd.Series:
        """
        Compute z-scores for residual values.
        
        Args:
            residuals: Residual component from STL decomposition
            
        Returns:
            Series of z-scores
        """
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        
        if std_residual == 0 or np.isnan(std_residual):
            # No variance in residuals - no anomalies
            return pd.Series(np.zeros(len(residuals)), index=residuals.index)
        
        return (residuals - mean_residual) / std_residual
    
    def _classify_anomaly_type(self, zscore: float) -> str:
        """
        Classify anomaly type based on z-score sign.
        
        Args:
            zscore: Z-score of the residual
            
        Returns:
            'spike' for positive anomaly, 'drop' for negative, 'normal' otherwise
        """
        if abs(zscore) <= self.zscore_threshold:
            return 'normal'
        elif zscore > 0:
            return 'spike'
        else:
            return 'drop'
    
    def _zscore_to_anomaly_score(self, zscore: float) -> float:
        """
        Convert z-score to anomaly score in [0, 1].
        
        Requirement 5.3: Generate an Anomaly_Factor score between 0 (normal)
        and 1 (extreme).
        
        Args:
            zscore: Z-score of the residual
            
        Returns:
            Anomaly score in [0, 1]
        """
        abs_zscore = abs(zscore)
        
        if abs_zscore <= self.zscore_threshold:
            # Below threshold: scale linearly from 0 to 0.5
            return (abs_zscore / self.zscore_threshold) * 0.5
        else:
            # Above threshold: scale from 0.5 to 1.0
            # Use sigmoid-like scaling for extreme values
            excess = abs_zscore - self.zscore_threshold
            # At 6 sigma (3 above threshold), score approaches 1.0
            score = 0.5 + 0.5 * (1 - np.exp(-excess / self.zscore_threshold))
            return min(score, 1.0)
    
    def detect_anomalies(self, time_series: pd.Series) -> List[AnomalyResult]:
        """
        Identify anomalies where residual exceeds threshold.
        
        Requirement 5.2: Flag residuals exceeding 3 standard deviations from mean.
        Requirement 5.4: Identify sudden spikes or drops in update volumes.
        
        Args:
            time_series: Time series data as pandas Series
            
        Returns:
            List of AnomalyResult for each data point
        """
        # Perform STL decomposition
        decomposition = self.decompose(time_series)
        
        # Compute z-scores for residuals
        zscores = self._compute_zscore(decomposition.residual)
        
        results = []
        for i, (idx, zscore) in enumerate(zscores.items()):
            abs_zscore = abs(zscore)
            is_anomaly = abs_zscore > self.zscore_threshold
            anomaly_type = self._classify_anomaly_type(zscore)
            anomaly_score = self._zscore_to_anomaly_score(zscore)
            
            # Try to get timestamp if index is datetime
            timestamp = None
            if isinstance(idx, (datetime, pd.Timestamp)):
                timestamp = idx
            
            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_type=anomaly_type,
                residual_zscore=float(zscore),
                timestamp=timestamp,
                index=i
            ))
        
        return results
    
    def compute_anomaly_factor(
        self,
        anomalies: Optional[List[AnomalyResult]] = None,
        time_series: Optional[pd.Series] = None
    ) -> float:
        """
        Convert anomaly results to a single [0, 1] factor score.
        
        Requirement 5.3: Generate an Anomaly_Factor score between 0 (normal)
        and 1 (extreme).
        
        The factor considers:
        - Proportion of anomalous points
        - Maximum anomaly severity
        - Recent anomaly weighting (more recent = higher impact)
        
        Args:
            anomalies: List of AnomalyResult (if None, uses time_series)
            time_series: Time series to analyze (if anomalies not provided)
            
        Returns:
            Anomaly factor score in [0, 1]
        """
        if anomalies is None:
            if time_series is None:
                return 0.0
            try:
                anomalies = self.detect_anomalies(time_series)
            except ValueError:
                # Time series too short for decomposition
                return 0.0
        
        if not anomalies:
            return 0.0
        
        # Count anomalies
        anomaly_count = sum(1 for a in anomalies if a.is_anomaly)
        total_count = len(anomalies)
        
        if anomaly_count == 0:
            return 0.0
        
        # Proportion of anomalous points (weight: 30%)
        proportion_factor = anomaly_count / total_count
        
        # Maximum anomaly score (weight: 40%)
        max_score = max(a.anomaly_score for a in anomalies)
        
        # Recent anomaly weighting (weight: 30%)
        # Give more weight to recent anomalies
        recent_weight = 0.0
        if anomaly_count > 0:
            # Weight recent points more heavily
            weights = np.linspace(0.5, 1.5, total_count)
            weighted_scores = [
                a.anomaly_score * w 
                for a, w in zip(anomalies, weights) 
                if a.is_anomaly
            ]
            if weighted_scores:
                recent_weight = np.mean(weighted_scores)
        
        # Combine factors
        anomaly_factor = (
            0.3 * proportion_factor +
            0.4 * max_score +
            0.3 * recent_weight
        )
        
        return float(np.clip(anomaly_factor, 0.0, 1.0))
    
    def get_last_decomposition(self) -> Optional[STLDecomposition]:
        """Return the last computed STL decomposition."""
        return self._last_decomposition
    
    def detect_spikes_and_drops(
        self,
        time_series: pd.Series
    ) -> Tuple[List[int], List[int]]:
        """
        Identify indices of spikes and drops in the time series.
        
        Requirement 5.4: Identify sudden spikes or drops in update volumes.
        
        Args:
            time_series: Time series data
            
        Returns:
            Tuple of (spike_indices, drop_indices)
        """
        anomalies = self.detect_anomalies(time_series)
        
        spike_indices = [
            a.index for a in anomalies 
            if a.is_anomaly and a.anomaly_type == 'spike'
        ]
        drop_indices = [
            a.index for a in anomalies 
            if a.is_anomaly and a.anomaly_type == 'drop'
        ]
        
        return spike_indices, drop_indices
