"""
Reason Code Generator Module

Generates explainable reason codes from ALRI sub-scores to help field officers
understand why a district has high risk and take appropriate action.

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.scoring.alri_calculator import ALRIResult


class Severity(Enum):
    """
    Severity levels for reason codes.
    
    Requirements: 7.4
    """
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class ReasonCode:
    """
    Explainable reason code indicating a risk driver.
    
    Requirements: 7.2, 7.4
    
    Attributes:
        code: Machine-readable code (e.g., 'Low_Child_Enrolment')
        description: Human-readable description of the risk
        severity: Severity level (Low/Medium/High/Critical)
        contribution: How much this factor contributed to ALRI (0-1)
        affected_population: Estimated people affected (optional)
    """
    code: str
    description: str
    severity: Severity
    contribution: float
    affected_population: int = 0
    
    def __post_init__(self):
        """Validate contribution is in valid range."""
        if not 0 <= self.contribution <= 1:
            raise ValueError(f"Contribution must be in [0, 1], got {self.contribution}")


class ReasonCodeGenerator:
    """
    Generates explainable reason codes from ALRI sub-scores.
    
    Maps sub-scores to human-readable labels and ranks them by contribution.
    
    Requirements: 7.1, 7.2, 7.3, 7.4
    """
    
    # Mapping from sub-score type to reason code and description
    REASON_CODE_MAP = {
        'coverage': {
            'code': 'Low_Child_Enrolment',
            'description': 'Low enrollment coverage, especially for children in 0-5 and 5-17 age bands'
        },
        'instability': {
            'code': 'High_Address_Churn',
            'description': 'High frequency of demographic updates indicating population mobility'
        },
        'biometric': {
            'code': 'Low_Biometric_Update_5to15',
            'description': 'Low biometric update rates for 5 and 15-year cohorts'
        },
        'anomaly': {
            'code': 'Anomalous_Data_Entry',
            'description': 'Unusual spikes or drops in update patterns detected'
        }
    }
    
    # Severity thresholds for sub-scores
    SEVERITY_THRESHOLDS = {
        'critical': 0.75,
        'high': 0.50,
        'medium': 0.25,
        'low': 0.0
    }
    
    def __init__(self, alri_threshold: float = 0.0):
        """
        Initialize the ReasonCodeGenerator.
        
        Args:
            alri_threshold: Minimum ALRI score to generate reason codes (default: 0.0)
        """
        self.alri_threshold = alri_threshold
    
    def determine_severity(self, score: float) -> Severity:
        """
        Map a sub-score to severity level.
        
        Requirements: 7.4
        
        Args:
            score: Sub-score value in [0, 1]
            
        Returns:
            Severity level based on score thresholds
        """
        if score >= self.SEVERITY_THRESHOLDS['critical']:
            return Severity.CRITICAL
        elif score >= self.SEVERITY_THRESHOLDS['high']:
            return Severity.HIGH
        elif score >= self.SEVERITY_THRESHOLDS['medium']:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def generate(self, alri_result: 'ALRIResult') -> List[ReasonCode]:
        """
        Generate ranked reason codes from ALRI components.
        
        Requirements: 7.1, 7.2, 7.3
        
        Args:
            alri_result: ALRIResult containing sub-scores
            
        Returns:
            List of ReasonCode objects ranked by contribution (highest first)
        """
        # Check if ALRI exceeds threshold (Requirement 7.1)
        if alri_result.alri_score < self.alri_threshold:
            return []
        
        # Extract sub-scores with their types
        sub_scores = [
            ('coverage', alri_result.coverage_risk),
            ('instability', alri_result.instability_risk),
            ('biometric', alri_result.biometric_risk),
            ('anomaly', alri_result.anomaly_factor)
        ]
        
        # Calculate total contribution for normalization
        total_score = sum(score for _, score in sub_scores)
        
        # Generate reason codes for each sub-score
        reason_codes = []
        for score_type, score_value in sub_scores:
            # Calculate contribution as proportion of total
            if total_score > 0:
                contribution = score_value / total_score
            else:
                contribution = 0.25  # Equal contribution if all zeros
            
            code_info = self.REASON_CODE_MAP[score_type]
            severity = self.determine_severity(score_value)
            
            reason_code = ReasonCode(
                code=code_info['code'],
                description=code_info['description'],
                severity=severity,
                contribution=contribution,
                affected_population=0  # Can be populated with actual data
            )
            reason_codes.append(reason_code)
        
        # Sort by contribution magnitude (highest first) - Requirement 7.3
        reason_codes.sort(key=lambda rc: rc.contribution, reverse=True)
        
        return reason_codes
    
    def get_top_contributors(
        self, 
        alri_result: 'ALRIResult', 
        n: int = 2
    ) -> List[ReasonCode]:
        """
        Get the top N contributing reason codes.
        
        Requirements: 7.1
        
        Args:
            alri_result: ALRIResult containing sub-scores
            n: Number of top contributors to return
            
        Returns:
            List of top N ReasonCode objects by contribution
        """
        all_codes = self.generate(alri_result)
        return all_codes[:n]
