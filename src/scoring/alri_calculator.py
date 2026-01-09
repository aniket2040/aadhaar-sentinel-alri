"""
ALRI Calculator Module

Computes the Aadhaar Lifecycle Risk Index (ALRI) from aggregated district data.
The ALRI is a composite score (0-100) combining four sub-scores:
- Coverage Risk: Measures enrollment coverage gaps
- Data Instability Risk: Measures demographic update frequency
- Biometric Compliance Risk: Measures biometric update compliance
- Anomaly Factor: Detects unusual patterns in data

Requirements: 2.1-2.4, 3.1-3.4, 4.1-4.4, 6.1-6.3
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class ALRIWeights:
    """
    Configurable weights for ALRI sub-score aggregation.
    
    Default weights sum to 1.0:
    - coverage: 0.30 (30%)
    - instability: 0.30 (30%)
    - biometric: 0.30 (30%)
    - anomaly: 0.10 (10%)
    
    Requirements: 6.1, 6.3
    """
    coverage: float = 0.30
    instability: float = 0.30
    biometric: float = 0.30
    anomaly: float = 0.10
    
    def __post_init__(self):
        """Validate that weights are non-negative."""
        if any(w < 0 for w in [self.coverage, self.instability, self.biometric, self.anomaly]):
            raise ValueError("All weights must be non-negative")
    
    def total(self) -> float:
        """Return the sum of all weights."""
        return self.coverage + self.instability + self.biometric + self.anomaly


@dataclass
class ALRIResult:
    """
    Result of ALRI computation for a district.
    
    Contains the composite ALRI score (0-100) and all sub-scores (0-1).
    
    Requirements: 6.1, 6.2
    """
    district: str
    state: str
    alri_score: float  # 0-100
    coverage_risk: float  # 0-1
    instability_risk: float  # 0-1
    biometric_risk: float  # 0-1
    anomaly_factor: float  # 0-1
    reason_codes: List[Any] = field(default_factory=list)
    recommendations: List[Any] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate score ranges."""
        if not 0 <= self.alri_score <= 100:
            raise ValueError(f"ALRI score must be in [0, 100], got {self.alri_score}")
        for name, score in [
            ('coverage_risk', self.coverage_risk),
            ('instability_risk', self.instability_risk),
            ('biometric_risk', self.biometric_risk),
            ('anomaly_factor', self.anomaly_factor)
        ]:
            if not 0 <= score <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {score}")


class ALRICalculator:
    """
    Computes ALRI scores from aggregated district data.
    
    The calculator combines four sub-scores using configurable weights:
    ALRI = (w1×Coverage + w2×Instability + w3×Biometric + w4×Anomaly) × 100
    
    Requirements: 2.1-2.4, 3.1-3.4, 4.1-4.4, 6.1-6.3
    """
    
    def __init__(self, weights: Optional[ALRIWeights] = None):
        """
        Initialize the ALRI Calculator.
        
        Args:
            weights: Custom weights for sub-score aggregation. 
                     Uses default weights if not provided.
        """
        self.weights = weights or ALRIWeights()
        self._computation_log: List[Dict[str, Any]] = []
    
    def _log_computation(self, step: str, details: Dict[str, Any]) -> None:
        """Log computation step for audit trail (Requirement 6.4)."""
        self._computation_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        })
    
    def _zscore_normalize(self, value: float, mean: float, std: float) -> float:
        """
        Apply z-score normalization and clip to [0, 1].
        
        Args:
            value: The value to normalize
            mean: Population mean
            std: Population standard deviation
            
        Returns:
            Normalized value clipped to [0, 1]
        """
        if std == 0 or np.isnan(std):
            return 0.5  # Default to middle if no variance
        
        z_score = (value - mean) / std
        # Convert z-score to [0, 1] using sigmoid-like transformation
        # z-score of -3 maps to ~0, z-score of +3 maps to ~1
        normalized = 1 / (1 + np.exp(-z_score))
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _scale_to_unit(self, value: float, min_val: float, max_val: float) -> float:
        """
        Scale a value to [0, 1] range using min-max scaling.
        
        Args:
            value: The value to scale
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Scaled value in [0, 1]
        """
        if max_val == min_val:
            return 0.5
        scaled = (value - min_val) / (max_val - min_val)
        return float(np.clip(scaled, 0.0, 1.0))
    
    def compute_coverage_risk(
        self,
        district_data: pd.DataFrame,
        baseline_enrollment: Optional[float] = None
    ) -> float:
        """
        Compute Coverage Risk sub-score (0-1).
        
        Measures enrollment coverage gaps relative to district baseline.
        Higher risk indicates lower enrollment coverage.
        
        Requirements: 2.1, 2.2, 2.3, 2.4
        
        Args:
            district_data: DataFrame with enrollment data for the district
            baseline_enrollment: Expected baseline enrollment (optional)
            
        Returns:
            Coverage risk score in [0, 1]
        """
        if district_data.empty:
            self._log_computation('coverage_risk', {'status': 'empty_data', 'result': 0.5})
            return 0.5
        
        # Get enrollment column
        enrollment_col = None
        for col in ['total_enrollment_age', 'total_enrollments', 'enrollment']:
            if col in district_data.columns:
                enrollment_col = col
                break
        
        if enrollment_col is None:
            self._log_computation('coverage_risk', {'status': 'no_enrollment_column', 'result': 0.5})
            return 0.5
        
        enrollments = district_data[enrollment_col].values
        
        # Calculate current enrollment rate
        current_enrollment = np.mean(enrollments)
        
        # Use provided baseline or compute from data
        if baseline_enrollment is None:
            baseline_enrollment = np.median(enrollments) if len(enrollments) > 0 else 1.0
        
        if baseline_enrollment <= 0:
            baseline_enrollment = 1.0
        
        # Calculate enrollment rate relative to baseline
        enrollment_rate = current_enrollment / baseline_enrollment
        
        # Detect month-on-month decline (Requirement 2.2)
        decline_factor = 0.0
        if len(enrollments) >= 2:
            # Calculate trend using simple linear regression
            x = np.arange(len(enrollments))
            std_x = np.std(x)
            std_enrollments = np.std(enrollments)
            # Only compute correlation if both have non-zero variance
            if std_x > 0 and std_enrollments > 0:
                correlation = np.corrcoef(x, enrollments)[0, 1]
                if not np.isnan(correlation):
                    slope = correlation * std_enrollments / std_x
                    if slope < 0:
                        # Negative slope indicates decline
                        decline_factor = min(abs(slope) / baseline_enrollment, 0.3)
        
        # Lower enrollment rate = higher risk (Requirement 2.4)
        # Invert the rate so low coverage yields high risk
        if enrollment_rate >= 1.0:
            base_risk = 0.0
        else:
            base_risk = 1.0 - enrollment_rate
        
        # Combine base risk with decline factor
        coverage_risk = min(base_risk + decline_factor, 1.0)
        
        # Apply z-score normalization and clip (Requirement 2.3)
        mean_enrollment = np.mean(enrollments)
        std_enrollment = np.std(enrollments)
        
        if std_enrollment > 0:
            # Normalize using z-score approach
            z_normalized = self._zscore_normalize(current_enrollment, mean_enrollment, std_enrollment)
            # Invert: lower enrollment = higher risk
            coverage_risk = 1.0 - z_normalized
        
        result = float(np.clip(coverage_risk, 0.0, 1.0))
        
        self._log_computation('coverage_risk', {
            'current_enrollment': current_enrollment,
            'baseline': baseline_enrollment,
            'enrollment_rate': enrollment_rate,
            'decline_factor': decline_factor,
            'result': result
        })
        
        return result

    
    def compute_instability_risk(
        self,
        district_data: pd.DataFrame,
        baseline_enrollment: Optional[float] = None
    ) -> float:
        """
        Compute Data Instability Risk sub-score (0-1).
        
        Measures demographic update frequency and volatility.
        Higher risk indicates more frequent demographic changes.
        
        Requirements: 3.1, 3.2, 3.3, 3.4
        
        Args:
            district_data: DataFrame with demographic update data
            baseline_enrollment: Baseline enrollment for rate calculation
            
        Returns:
            Instability risk score in [0, 1]
        """
        if district_data.empty:
            self._log_computation('instability_risk', {'status': 'empty_data', 'result': 0.5})
            return 0.5
        
        # Get demographic update column
        demographic_col = None
        for col in ['total_demographic_age', 'total_demographic_updates', 'demographic_updates']:
            if col in district_data.columns:
                demographic_col = col
                break
        
        # Get enrollment column for rate calculation
        enrollment_col = None
        for col in ['total_enrollment_age', 'total_enrollments', 'enrollment']:
            if col in district_data.columns:
                enrollment_col = col
                break
        
        if demographic_col is None:
            self._log_computation('instability_risk', {'status': 'no_demographic_column', 'result': 0.5})
            return 0.5
        
        demographic_updates = district_data[demographic_col].values
        
        # Calculate demographic update rate per 1000 enrollments (Requirement 3.1)
        if enrollment_col and baseline_enrollment is None:
            enrollments = district_data[enrollment_col].values
            baseline_enrollment = np.sum(enrollments) if len(enrollments) > 0 else 1000.0
        
        if baseline_enrollment is None or baseline_enrollment <= 0:
            baseline_enrollment = 1000.0
        
        total_updates = np.sum(demographic_updates)
        update_rate_per_1000 = (total_updates / baseline_enrollment) * 1000
        
        # Calculate rolling volatility (Requirement 3.2)
        volatility = 0.0
        if len(demographic_updates) >= 3:
            # Use rolling standard deviation
            rolling_std = pd.Series(demographic_updates).rolling(window=3, min_periods=2).std()
            volatility = rolling_std.mean() if not rolling_std.isna().all() else 0.0
            if np.isnan(volatility):
                volatility = 0.0
        
        # Normalize volatility relative to mean
        mean_updates = np.mean(demographic_updates)
        if mean_updates > 0:
            normalized_volatility = volatility / mean_updates
        else:
            normalized_volatility = 0.0
        
        # Combine update rate and volatility into risk score
        # Higher update rate = higher risk (Requirement 3.4)
        # Scale update rate: assume 50 updates per 1000 enrollments is moderate
        rate_risk = self._scale_to_unit(update_rate_per_1000, 0, 100)
        
        # Scale volatility: assume coefficient of variation of 0.5 is moderate
        volatility_risk = self._scale_to_unit(normalized_volatility, 0, 1.0)
        
        # Combine: 70% rate, 30% volatility
        instability_risk = 0.7 * rate_risk + 0.3 * volatility_risk
        
        result = float(np.clip(instability_risk, 0.0, 1.0))
        
        self._log_computation('instability_risk', {
            'total_updates': total_updates,
            'baseline_enrollment': baseline_enrollment,
            'update_rate_per_1000': update_rate_per_1000,
            'volatility': volatility,
            'rate_risk': rate_risk,
            'volatility_risk': volatility_risk,
            'result': result
        })
        
        return result
    
    def compute_biometric_risk(
        self,
        district_data: pd.DataFrame,
        expected_biometric_updates: Optional[float] = None
    ) -> float:
        """
        Compute Biometric Compliance Risk sub-score (0-1).
        
        Measures biometric update rates relative to expected volumes.
        Higher risk indicates lower biometric compliance.
        
        Requirements: 4.1, 4.2, 4.3, 4.4
        
        Args:
            district_data: DataFrame with biometric update data
            expected_biometric_updates: Expected biometric update volume
            
        Returns:
            Biometric risk score in [0, 1]
        """
        if district_data.empty:
            self._log_computation('biometric_risk', {'status': 'empty_data', 'result': 0.5})
            return 0.5
        
        # Get biometric update column
        biometric_col = None
        for col in ['total_biometric_age', 'total_biometric_updates', 'biometric_updates']:
            if col in district_data.columns:
                biometric_col = col
                break
        
        if biometric_col is None:
            self._log_computation('biometric_risk', {'status': 'no_biometric_column', 'result': 0.5})
            return 0.5
        
        biometric_updates = district_data[biometric_col].values
        actual_updates = np.sum(biometric_updates)
        
        # Calculate expected updates if not provided (Requirement 4.1)
        if expected_biometric_updates is None:
            # Use median as expected baseline
            expected_biometric_updates = np.median(biometric_updates) * len(biometric_updates)
            if expected_biometric_updates <= 0:
                expected_biometric_updates = 1.0
        
        # Calculate compliance rate
        compliance_rate = actual_updates / expected_biometric_updates if expected_biometric_updates > 0 else 0.0
        
        # Detect declining trend (Requirement 4.2)
        decline_penalty = 0.0
        if len(biometric_updates) >= 2:
            x = np.arange(len(biometric_updates))
            if np.std(biometric_updates) > 0 and np.std(x) > 0:
                correlation = np.corrcoef(x, biometric_updates)[0, 1]
                if not np.isnan(correlation) and correlation < -0.3:
                    # Significant negative correlation indicates decline
                    decline_penalty = min(abs(correlation) * 0.3, 0.3)
        
        # Higher missing updates = higher risk (Requirement 4.3)
        if compliance_rate >= 1.0:
            base_risk = 0.0
        else:
            base_risk = 1.0 - compliance_rate
        
        # Apply threshold check (Requirement 4.4)
        # Flag if updates fall below 70% of expected
        threshold_penalty = 0.0
        if compliance_rate < 0.7:
            threshold_penalty = 0.2
        
        biometric_risk = min(base_risk + decline_penalty + threshold_penalty, 1.0)
        
        result = float(np.clip(biometric_risk, 0.0, 1.0))
        
        self._log_computation('biometric_risk', {
            'actual_updates': actual_updates,
            'expected_updates': expected_biometric_updates,
            'compliance_rate': compliance_rate,
            'decline_penalty': decline_penalty,
            'threshold_penalty': threshold_penalty,
            'result': result
        })
        
        return result

    
    def compute_anomaly_factor(
        self,
        district_data: pd.DataFrame,
        anomaly_score: Optional[float] = None
    ) -> float:
        """
        Compute Anomaly Factor sub-score (0-1).
        
        This method accepts a pre-computed anomaly score from the STLAnomalyDetector
        or computes a simple anomaly factor based on data variance.
        
        Requirements: 5.3
        
        Args:
            district_data: DataFrame with time-series data
            anomaly_score: Pre-computed anomaly score from detector
            
        Returns:
            Anomaly factor score in [0, 1]
        """
        # If anomaly score is provided, use it directly
        if anomaly_score is not None:
            result = float(np.clip(anomaly_score, 0.0, 1.0))
            self._log_computation('anomaly_factor', {
                'source': 'provided',
                'result': result
            })
            return result
        
        if district_data.empty:
            self._log_computation('anomaly_factor', {'status': 'empty_data', 'result': 0.0})
            return 0.0
        
        # Simple anomaly detection based on z-scores
        # Look for any numeric column with time-series data
        numeric_cols = district_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self._log_computation('anomaly_factor', {'status': 'no_numeric_columns', 'result': 0.0})
            return 0.0
        
        max_anomaly_score = 0.0
        
        for col in numeric_cols:
            values = district_data[col].dropna().values
            if len(values) < 3:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                # Count values exceeding 3 standard deviations
                anomaly_count = np.sum(z_scores > 3)
                anomaly_ratio = anomaly_count / len(values)
                
                # Also consider max z-score
                max_z = np.max(z_scores)
                z_score_factor = min(max_z / 6.0, 1.0)  # 6 sigma = max risk
                
                col_anomaly_score = 0.5 * anomaly_ratio + 0.5 * z_score_factor
                max_anomaly_score = max(max_anomaly_score, col_anomaly_score)
        
        result = float(np.clip(max_anomaly_score, 0.0, 1.0))
        
        self._log_computation('anomaly_factor', {
            'source': 'computed',
            'result': result
        })
        
        return result
    
    def compute_alri(
        self,
        district_data: pd.DataFrame,
        district: str,
        state: str,
        baseline_enrollment: Optional[float] = None,
        expected_biometric_updates: Optional[float] = None,
        anomaly_score: Optional[float] = None
    ) -> ALRIResult:
        """
        Compute composite ALRI score (0-100) with all components.
        
        Combines sub-scores using weighted formula:
        ALRI = (w1×Coverage + w2×Instability + w3×Biometric + w4×Anomaly) × 100
        
        Requirements: 6.1, 6.2
        
        Args:
            district_data: DataFrame with all district data
            district: District name
            state: State name
            baseline_enrollment: Baseline enrollment for normalization
            expected_biometric_updates: Expected biometric update volume
            anomaly_score: Pre-computed anomaly score
            
        Returns:
            ALRIResult with composite score and all sub-scores
        """
        # Compute all sub-scores
        coverage_risk = self.compute_coverage_risk(district_data, baseline_enrollment)
        instability_risk = self.compute_instability_risk(district_data, baseline_enrollment)
        biometric_risk = self.compute_biometric_risk(district_data, expected_biometric_updates)
        anomaly_factor = self.compute_anomaly_factor(district_data, anomaly_score)
        
        # Compute weighted ALRI score (Requirement 6.1)
        weighted_sum = (
            self.weights.coverage * coverage_risk +
            self.weights.instability * instability_risk +
            self.weights.biometric * biometric_risk +
            self.weights.anomaly * anomaly_factor
        )
        
        # Normalize by total weights and scale to 0-100
        weight_total = self.weights.total()
        if weight_total > 0:
            alri_score = (weighted_sum / weight_total) * 100
        else:
            alri_score = 0.0
        
        # Ensure output in [0, 100] range (Requirement 6.2)
        alri_score = float(np.clip(alri_score, 0.0, 100.0))
        
        self._log_computation('alri_score', {
            'district': district,
            'state': state,
            'coverage_risk': coverage_risk,
            'instability_risk': instability_risk,
            'biometric_risk': biometric_risk,
            'anomaly_factor': anomaly_factor,
            'weighted_sum': weighted_sum,
            'weight_total': weight_total,
            'alri_score': alri_score
        })
        
        return ALRIResult(
            district=district,
            state=state,
            alri_score=alri_score,
            coverage_risk=coverage_risk,
            instability_risk=instability_risk,
            biometric_risk=biometric_risk,
            anomaly_factor=anomaly_factor,
            computed_at=datetime.now()
        )
    
    def compute_alri_from_subscores(
        self,
        coverage_risk: float,
        instability_risk: float,
        biometric_risk: float,
        anomaly_factor: float,
        district: str = "unknown",
        state: str = "unknown"
    ) -> ALRIResult:
        """
        Compute ALRI score directly from pre-computed sub-scores.
        
        Useful for testing and when sub-scores are computed separately.
        
        Args:
            coverage_risk: Coverage risk sub-score [0, 1]
            instability_risk: Instability risk sub-score [0, 1]
            biometric_risk: Biometric risk sub-score [0, 1]
            anomaly_factor: Anomaly factor sub-score [0, 1]
            district: District name
            state: State name
            
        Returns:
            ALRIResult with composite score
        """
        # Validate sub-scores are in [0, 1]
        for name, score in [
            ('coverage_risk', coverage_risk),
            ('instability_risk', instability_risk),
            ('biometric_risk', biometric_risk),
            ('anomaly_factor', anomaly_factor)
        ]:
            if not 0 <= score <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {score}")
        
        # Compute weighted ALRI score
        weighted_sum = (
            self.weights.coverage * coverage_risk +
            self.weights.instability * instability_risk +
            self.weights.biometric * biometric_risk +
            self.weights.anomaly * anomaly_factor
        )
        
        weight_total = self.weights.total()
        if weight_total > 0:
            alri_score = (weighted_sum / weight_total) * 100
        else:
            alri_score = 0.0
        
        alri_score = float(np.clip(alri_score, 0.0, 100.0))
        
        return ALRIResult(
            district=district,
            state=state,
            alri_score=alri_score,
            coverage_risk=coverage_risk,
            instability_risk=instability_risk,
            biometric_risk=biometric_risk,
            anomaly_factor=anomaly_factor,
            computed_at=datetime.now()
        )
    
    def get_computation_log(self) -> List[Dict[str, Any]]:
        """Return the computation log for audit purposes (Requirement 6.4)."""
        return self._computation_log.copy()
    
    def clear_computation_log(self) -> None:
        """Clear the computation log."""
        self._computation_log.clear()
