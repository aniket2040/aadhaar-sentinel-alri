"""
Recommendation Engine Module

Maps reason codes to actionable interventions for at-risk districts.
Prioritizes low-cost, high-impact interventions.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

from src.explainability.reason_codes import ReasonCode


class CostLevel(Enum):
    """Cost level for interventions."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class Intervention:
    """
    Recommended operational action for an at-risk district.
    
    Requirements: 8.2, 8.3
    
    Attributes:
        action: Short action name (e.g., 'Mobile Van Deployment')
        description: Detailed description of the intervention
        estimated_cost: Cost level (Low/Medium/High)
        estimated_impact: Estimated people affected
        priority: Priority level (1 = highest)
    """
    action: str
    description: str
    estimated_cost: CostLevel
    estimated_impact: int
    priority: int
    
    def __post_init__(self):
        """Validate priority is positive."""
        if self.priority < 1:
            raise ValueError(f"Priority must be >= 1, got {self.priority}")


class RecommendationEngine:
    """
    Maps reason codes to actionable interventions.
    
    Provides 1-3 recommended interventions per reason code,
    prioritized by cost-effectiveness (low-cost, high-impact first).
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    
    # Mapping from reason codes to interventions
    # Each reason code maps to 1-3 interventions (Requirement 8.1)
    INTERVENTION_MAP: Dict[str, List[Intervention]] = {
        'Low_Child_Enrolment': [
            Intervention(
                action='School Enrollment Drive',
                description='Partner with local schools to conduct enrollment camps during school hours, targeting children aged 5-17',
                estimated_cost=CostLevel.LOW,
                estimated_impact=5000,
                priority=1
            ),
            Intervention(
                action='Mobile Van Deployment',
                description='Deploy mobile enrollment vans to underserved areas with low child enrollment coverage',
                estimated_cost=CostLevel.MEDIUM,
                estimated_impact=3000,
                priority=2
            ),
            Intervention(
                action='Anganwadi Partnership',
                description='Collaborate with Anganwadi centers for 0-5 age group enrollment drives',
                estimated_cost=CostLevel.LOW,
                estimated_impact=2000,
                priority=3
            ),
        ],
        'High_Address_Churn': [
            Intervention(
                action='SMS/IVR Campaign',
                description='Send SMS and IVR reminders to residents about address update procedures and nearest centers',
                estimated_cost=CostLevel.LOW,
                estimated_impact=10000,
                priority=1
            ),
            Intervention(
                action='Additional Update Kiosks',
                description='Install additional self-service kiosks in high-churn areas for convenient address updates',
                estimated_cost=CostLevel.MEDIUM,
                estimated_impact=5000,
                priority=2
            ),
            Intervention(
                action='Community Awareness Program',
                description='Conduct community meetings to educate residents about importance of keeping Aadhaar updated',
                estimated_cost=CostLevel.LOW,
                estimated_impact=3000,
                priority=3
            ),
        ],
        'Low_Biometric_Update_5to15': [
            Intervention(
                action='Free Biometric Camp',
                description='Organize free biometric update camps targeting 5 and 15-year cohorts requiring mandatory updates',
                estimated_cost=CostLevel.LOW,
                estimated_impact=4000,
                priority=1
            ),
            Intervention(
                action='School Biometric Drive',
                description='Conduct biometric update drives in schools for students in 5-15 age group',
                estimated_cost=CostLevel.LOW,
                estimated_impact=6000,
                priority=2
            ),
            Intervention(
                action='Parent Notification Campaign',
                description='Send notifications to parents about mandatory biometric updates for their children',
                estimated_cost=CostLevel.LOW,
                estimated_impact=8000,
                priority=3
            ),
        ],
        'Anomalous_Data_Entry': [
            Intervention(
                action='Data Quality Audit',
                description='Conduct detailed audit of enrollment centers showing anomalous patterns to identify data quality issues',
                estimated_cost=CostLevel.MEDIUM,
                estimated_impact=1000,
                priority=1
            ),
            Intervention(
                action='Operator Training',
                description='Provide refresher training to enrollment operators on data entry best practices',
                estimated_cost=CostLevel.LOW,
                estimated_impact=500,
                priority=2
            ),
        ],
    }
    
    def __init__(self):
        """Initialize the RecommendationEngine."""
        pass
    
    def recommend(self, reason_codes: List[ReasonCode]) -> List[Intervention]:
        """
        Generate prioritized intervention list from reason codes.
        
        Requirements: 8.1, 8.4
        
        Args:
            reason_codes: List of ReasonCode objects from ReasonCodeGenerator
            
        Returns:
            List of Intervention objects ordered by priority (low-cost, high-impact first)
        """
        if not reason_codes:
            return []
        
        all_interventions: List[Intervention] = []
        
        # Map each reason code to its interventions (Requirement 8.1)
        for rc in reason_codes:
            if rc.code in self.INTERVENTION_MAP:
                interventions = self.INTERVENTION_MAP[rc.code]
                all_interventions.extend(interventions)
        
        # Remove duplicates while preserving order
        seen_actions = set()
        unique_interventions = []
        for intervention in all_interventions:
            if intervention.action not in seen_actions:
                seen_actions.add(intervention.action)
                unique_interventions.append(intervention)
        
        # Sort by priority: low-cost, high-impact first (Requirement 8.4)
        # Priority order: 1) Cost (Low < Medium < High), 2) Impact (higher first), 3) Priority number
        def sort_key(intervention: Intervention):
            cost_order = {CostLevel.LOW: 0, CostLevel.MEDIUM: 1, CostLevel.HIGH: 2}
            return (
                cost_order[intervention.estimated_cost],
                -intervention.estimated_impact,  # Negative for descending order
                intervention.priority
            )
        
        unique_interventions.sort(key=sort_key)
        
        return unique_interventions
    
    def recommend_for_code(self, reason_code: str) -> List[Intervention]:
        """
        Get interventions for a specific reason code.
        
        Requirements: 8.1
        
        Args:
            reason_code: The reason code string (e.g., 'Low_Child_Enrolment')
            
        Returns:
            List of 1-3 Intervention objects for the given code
        """
        if reason_code not in self.INTERVENTION_MAP:
            return []
        
        return self.INTERVENTION_MAP[reason_code].copy()
    
    def get_top_interventions(
        self, 
        reason_codes: List[ReasonCode], 
        n: int = 3
    ) -> List[Intervention]:
        """
        Get the top N prioritized interventions.
        
        Args:
            reason_codes: List of ReasonCode objects
            n: Number of top interventions to return
            
        Returns:
            List of top N Intervention objects
        """
        all_interventions = self.recommend(reason_codes)
        return all_interventions[:n]
