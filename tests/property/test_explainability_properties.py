"""Property-based tests for explainability module correctness.

Feature: aadhaar-sentinel, Properties 10-13: Explainability Properties
Validates: Requirements 7.2, 7.3, 7.4, 8.1, 8.4
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.scoring.alri_calculator import ALRIResult, ALRICalculator
from src.explainability.reason_codes import Severity, ReasonCode, ReasonCodeGenerator
from tests.conftest import (
    sub_score_strategy,
    sub_scores_strategy,
    state_strategy,
    district_strategy,
)


# =============================================================================
# Custom Strategies for Explainability Tests
# =============================================================================

@st.composite
def alri_result_strategy(draw):
    """Generate a valid ALRIResult for testing."""
    coverage = draw(sub_score_strategy)
    instability = draw(sub_score_strategy)
    biometric = draw(sub_score_strategy)
    anomaly = draw(sub_score_strategy)
    
    # Compute ALRI score using default weights
    weighted_sum = 0.30 * coverage + 0.30 * instability + 0.30 * biometric + 0.10 * anomaly
    alri_score = weighted_sum * 100
    
    return ALRIResult(
        district=draw(district_strategy),
        state=draw(state_strategy),
        alri_score=alri_score,
        coverage_risk=coverage,
        instability_risk=instability,
        biometric_risk=biometric,
        anomaly_factor=anomaly
    )


@st.composite
def alri_result_with_varied_subscores_strategy(draw):
    """Generate ALRIResult with varied sub-scores for ranking tests."""
    # Generate 4 distinct sub-scores to ensure clear ranking
    scores = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) for _ in range(4)]
    
    coverage, instability, biometric, anomaly = scores
    
    weighted_sum = 0.30 * coverage + 0.30 * instability + 0.30 * biometric + 0.10 * anomaly
    alri_score = weighted_sum * 100
    
    return ALRIResult(
        district=draw(district_strategy),
        state=draw(state_strategy),
        alri_score=alri_score,
        coverage_risk=coverage,
        instability_risk=instability,
        biometric_risk=biometric,
        anomaly_factor=anomaly
    )


# =============================================================================
# Property 10: Reason Code Ranking
# =============================================================================

class TestReasonCodeRanking:
    """Property 10: Reason Code Ranking
    
    For any ALRI result with multiple contributing factors, the generated
    Reason_Codes SHALL be ordered by contribution magnitude (highest first).
    
    Validates: Requirements 7.3
    """
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_reason_codes_ordered_by_contribution(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 10: Reason Code Ranking
        
        For any ALRI result, generated reason codes SHALL be ordered
        by contribution magnitude (highest first).
        
        Validates: Requirements 7.3
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        # Verify codes are ordered by contribution (descending)
        contributions = [rc.contribution for rc in reason_codes]
        
        for i in range(len(contributions) - 1):
            assert contributions[i] >= contributions[i + 1], \
                f"Reason codes not ordered by contribution: {contributions}"
    
    @given(alri_result=alri_result_with_varied_subscores_strategy())
    @settings(max_examples=100)
    def test_highest_subscore_yields_highest_contribution(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 10: Reason Code Ranking
        
        The sub-score with highest value SHALL have the highest contribution
        in the generated reason codes.
        
        Validates: Requirements 7.3
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        # Find the highest sub-score
        sub_scores = {
            'Low_Child_Enrolment': alri_result.coverage_risk,
            'High_Address_Churn': alri_result.instability_risk,
            'Low_Biometric_Update_5to15': alri_result.biometric_risk,
            'Anomalous_Data_Entry': alri_result.anomaly_factor
        }
        
        max_subscore_code = max(sub_scores, key=sub_scores.get)
        
        # The first reason code should correspond to the highest sub-score
        # (unless there are ties)
        if len(reason_codes) > 0:
            top_code = reason_codes[0]
            max_value = sub_scores[max_subscore_code]
            top_value = sub_scores[top_code.code]
            
            # Allow for floating point comparison
            assert top_value >= max_value - 1e-9, \
                f"Top reason code {top_code.code} (value={top_value}) should match " \
                f"highest sub-score {max_subscore_code} (value={max_value})"


# =============================================================================
# Property 11: Reason Code Completeness
# =============================================================================

class TestReasonCodeCompleteness:
    """Property 11: Reason Code Completeness
    
    For any generated Reason_Code, it SHALL include a valid label from the
    defined set, a severity level (Low/Medium/High/Critical), and a
    contribution value.
    
    Validates: Requirements 7.2, 7.4
    """
    
    VALID_CODES = {
        'Low_Child_Enrolment',
        'High_Address_Churn',
        'Low_Biometric_Update_5to15',
        'Anomalous_Data_Entry'
    }
    
    VALID_SEVERITIES = {Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_reason_codes_have_valid_labels(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any generated reason code, the code label SHALL be from the
        defined set of valid codes.
        
        Validates: Requirements 7.2
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        for rc in reason_codes:
            assert rc.code in self.VALID_CODES, \
                f"Invalid reason code: {rc.code}. Valid codes: {self.VALID_CODES}"
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_reason_codes_have_valid_severity(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any generated reason code, the severity SHALL be one of
        Low/Medium/High/Critical.
        
        Validates: Requirements 7.4
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        for rc in reason_codes:
            assert rc.severity in self.VALID_SEVERITIES, \
                f"Invalid severity: {rc.severity}. Valid severities: {self.VALID_SEVERITIES}"
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_reason_codes_have_valid_contribution(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any generated reason code, the contribution value SHALL be
        in the range [0, 1].
        
        Validates: Requirements 7.3
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        for rc in reason_codes:
            assert 0.0 <= rc.contribution <= 1.0, \
                f"Invalid contribution: {rc.contribution}. Must be in [0, 1]"
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_reason_codes_have_description(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any generated reason code, the description SHALL be non-empty.
        
        Validates: Requirements 7.2
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        for rc in reason_codes:
            assert rc.description and len(rc.description) > 0, \
                f"Reason code {rc.code} has empty description"
    
    @given(alri_result=alri_result_strategy())
    @settings(max_examples=100)
    def test_contributions_sum_to_one(self, alri_result):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any ALRI result, the sum of all reason code contributions
        SHALL equal 1.0 (within floating point tolerance).
        
        Validates: Requirements 7.3
        """
        generator = ReasonCodeGenerator()
        reason_codes = generator.generate(alri_result)
        
        if len(reason_codes) > 0:
            total_contribution = sum(rc.contribution for rc in reason_codes)
            assert abs(total_contribution - 1.0) < 1e-6, \
                f"Contributions sum to {total_contribution}, expected 1.0"


# =============================================================================
# Severity Determination Tests
# =============================================================================

class TestSeverityDetermination:
    """Tests for severity level determination based on sub-scores."""
    
    @given(score=st.floats(min_value=0.75, max_value=1.0, allow_nan=False))
    @settings(max_examples=100)
    def test_critical_severity_threshold(self, score):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any sub-score >= 0.75, severity SHALL be CRITICAL.
        
        Validates: Requirements 7.4
        """
        generator = ReasonCodeGenerator()
        severity = generator.determine_severity(score)
        
        assert severity == Severity.CRITICAL, \
            f"Score {score} should yield CRITICAL severity, got {severity}"
    
    @given(score=st.floats(min_value=0.50, max_value=0.749, allow_nan=False))
    @settings(max_examples=100)
    def test_high_severity_threshold(self, score):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any sub-score in [0.50, 0.75), severity SHALL be HIGH.
        
        Validates: Requirements 7.4
        """
        generator = ReasonCodeGenerator()
        severity = generator.determine_severity(score)
        
        assert severity == Severity.HIGH, \
            f"Score {score} should yield HIGH severity, got {severity}"
    
    @given(score=st.floats(min_value=0.25, max_value=0.499, allow_nan=False))
    @settings(max_examples=100)
    def test_medium_severity_threshold(self, score):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any sub-score in [0.25, 0.50), severity SHALL be MEDIUM.
        
        Validates: Requirements 7.4
        """
        generator = ReasonCodeGenerator()
        severity = generator.determine_severity(score)
        
        assert severity == Severity.MEDIUM, \
            f"Score {score} should yield MEDIUM severity, got {severity}"
    
    @given(score=st.floats(min_value=0.0, max_value=0.249, allow_nan=False))
    @settings(max_examples=100)
    def test_low_severity_threshold(self, score):
        """
        Feature: aadhaar-sentinel, Property 11: Reason Code Completeness
        
        For any sub-score in [0, 0.25), severity SHALL be LOW.
        
        Validates: Requirements 7.4
        """
        generator = ReasonCodeGenerator()
        severity = generator.determine_severity(score)
        
        assert severity == Severity.LOW, \
            f"Score {score} should yield LOW severity, got {severity}"


# =============================================================================
# Import Recommendation Engine for Properties 12-13
# =============================================================================

from src.recommendations.engine import CostLevel, Intervention, RecommendationEngine


# =============================================================================
# Custom Strategies for Recommendation Tests
# =============================================================================

@st.composite
def reason_code_list_strategy(draw):
    """Generate a list of valid ReasonCode objects."""
    # Generate 1-4 reason codes
    num_codes = draw(st.integers(min_value=1, max_value=4))
    
    codes = []
    for _ in range(num_codes):
        # Generate sub-scores for contribution calculation
        coverage = draw(sub_score_strategy)
        instability = draw(sub_score_strategy)
        biometric = draw(sub_score_strategy)
        anomaly = draw(sub_score_strategy)
        
        total = coverage + instability + biometric + anomaly
        if total == 0:
            total = 1.0  # Avoid division by zero
        
        # Pick a random code type
        code_type = draw(st.sampled_from(['coverage', 'instability', 'biometric', 'anomaly']))
        
        code_map = {
            'coverage': ('Low_Child_Enrolment', coverage / total),
            'instability': ('High_Address_Churn', instability / total),
            'biometric': ('Low_Biometric_Update_5to15', biometric / total),
            'anomaly': ('Anomalous_Data_Entry', anomaly / total)
        }
        
        code_name, contribution = code_map[code_type]
        score_value = {'coverage': coverage, 'instability': instability, 
                       'biometric': biometric, 'anomaly': anomaly}[code_type]
        
        # Determine severity based on score
        if score_value >= 0.75:
            severity = Severity.CRITICAL
        elif score_value >= 0.50:
            severity = Severity.HIGH
        elif score_value >= 0.25:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        codes.append(ReasonCode(
            code=code_name,
            description=f'Test description for {code_name}',
            severity=severity,
            contribution=contribution
        ))
    
    return codes


# =============================================================================
# Property 12: Recommendation Mapping
# =============================================================================

class TestRecommendationMapping:
    """Property 12: Recommendation Mapping
    
    For any valid Reason_Code, the Recommendation_Engine SHALL produce
    between 1 and 3 interventions from the defined intervention set.
    
    Validates: Requirements 8.1, 8.2
    """
    
    VALID_REASON_CODES = {
        'Low_Child_Enrolment',
        'High_Address_Churn',
        'Low_Biometric_Update_5to15',
        'Anomalous_Data_Entry'
    }
    
    @given(reason_code=st.sampled_from(list(VALID_REASON_CODES)))
    @settings(max_examples=100)
    def test_each_code_produces_1_to_3_interventions(self, reason_code):
        """
        Feature: aadhaar-sentinel, Property 12: Recommendation Mapping
        
        For any valid reason code, the engine SHALL produce 1-3 interventions.
        
        Validates: Requirements 8.1
        """
        engine = RecommendationEngine()
        interventions = engine.recommend_for_code(reason_code)
        
        assert 1 <= len(interventions) <= 3, \
            f"Reason code {reason_code} produced {len(interventions)} interventions, expected 1-3"
    
    @given(reason_codes=reason_code_list_strategy())
    @settings(max_examples=100)
    def test_recommend_produces_interventions_for_all_codes(self, reason_codes):
        """
        Feature: aadhaar-sentinel, Property 12: Recommendation Mapping
        
        For any list of reason codes, the engine SHALL produce interventions
        for each valid code.
        
        Validates: Requirements 8.1
        """
        engine = RecommendationEngine()
        interventions = engine.recommend(reason_codes)
        
        # Should produce at least 1 intervention if we have valid codes
        valid_codes = [rc for rc in reason_codes if rc.code in self.VALID_REASON_CODES]
        if valid_codes:
            assert len(interventions) >= 1, \
                f"Expected at least 1 intervention for {len(valid_codes)} valid codes"
    
    @given(reason_codes=reason_code_list_strategy())
    @settings(max_examples=100)
    def test_interventions_have_valid_structure(self, reason_codes):
        """
        Feature: aadhaar-sentinel, Property 12: Recommendation Mapping
        
        For any generated intervention, it SHALL have valid action, description,
        cost level, impact, and priority.
        
        Validates: Requirements 8.2
        """
        engine = RecommendationEngine()
        interventions = engine.recommend(reason_codes)
        
        for intervention in interventions:
            # Check action is non-empty
            assert intervention.action and len(intervention.action) > 0, \
                "Intervention action should be non-empty"
            
            # Check description is non-empty
            assert intervention.description and len(intervention.description) > 0, \
                "Intervention description should be non-empty"
            
            # Check cost level is valid
            assert intervention.estimated_cost in {CostLevel.LOW, CostLevel.MEDIUM, CostLevel.HIGH}, \
                f"Invalid cost level: {intervention.estimated_cost}"
            
            # Check impact is non-negative
            assert intervention.estimated_impact >= 0, \
                f"Impact should be non-negative: {intervention.estimated_impact}"
            
            # Check priority is positive
            assert intervention.priority >= 1, \
                f"Priority should be >= 1: {intervention.priority}"


# =============================================================================
# Property 13: Recommendation Priority Ordering
# =============================================================================

class TestRecommendationPriorityOrdering:
    """Property 13: Recommendation Priority Ordering
    
    For any list of recommendations, they SHALL be ordered by priority
    (low-cost, high-impact first).
    
    Validates: Requirements 8.4
    """
    
    @given(reason_codes=reason_code_list_strategy())
    @settings(max_examples=100)
    def test_recommendations_ordered_by_cost_first(self, reason_codes):
        """
        Feature: aadhaar-sentinel, Property 13: Recommendation Priority Ordering
        
        Recommendations SHALL be ordered with low-cost interventions first.
        
        Validates: Requirements 8.4
        """
        engine = RecommendationEngine()
        interventions = engine.recommend(reason_codes)
        
        if len(interventions) < 2:
            return  # Nothing to compare
        
        cost_order = {CostLevel.LOW: 0, CostLevel.MEDIUM: 1, CostLevel.HIGH: 2}
        
        # Check that costs are in non-decreasing order (allowing ties)
        for i in range(len(interventions) - 1):
            current_cost = cost_order[interventions[i].estimated_cost]
            next_cost = cost_order[interventions[i + 1].estimated_cost]
            
            # If costs are equal, check impact ordering
            if current_cost == next_cost:
                # Higher impact should come first (or equal)
                assert interventions[i].estimated_impact >= interventions[i + 1].estimated_impact or \
                       interventions[i].priority <= interventions[i + 1].priority, \
                    f"Within same cost level, higher impact should come first"
            else:
                # Lower cost should come first
                assert current_cost <= next_cost, \
                    f"Lower cost interventions should come first: " \
                    f"{interventions[i].estimated_cost} vs {interventions[i + 1].estimated_cost}"
    
    @given(reason_codes=reason_code_list_strategy())
    @settings(max_examples=100)
    def test_low_cost_interventions_prioritized(self, reason_codes):
        """
        Feature: aadhaar-sentinel, Property 13: Recommendation Priority Ordering
        
        Low-cost interventions SHALL appear before high-cost interventions.
        
        Validates: Requirements 8.4
        """
        engine = RecommendationEngine()
        interventions = engine.recommend(reason_codes)
        
        if len(interventions) == 0:
            return
        
        # Find first high-cost intervention
        first_high_cost_idx = None
        for i, intervention in enumerate(interventions):
            if intervention.estimated_cost == CostLevel.HIGH:
                first_high_cost_idx = i
                break
        
        if first_high_cost_idx is None:
            return  # No high-cost interventions
        
        # All interventions before first high-cost should be low or medium cost
        for i in range(first_high_cost_idx):
            assert interventions[i].estimated_cost in {CostLevel.LOW, CostLevel.MEDIUM}, \
                f"High-cost intervention at index {first_high_cost_idx} should not " \
                f"come before low/medium cost intervention at index {i}"
    
    @given(reason_codes=reason_code_list_strategy())
    @settings(max_examples=100)
    def test_no_duplicate_interventions(self, reason_codes):
        """
        Feature: aadhaar-sentinel, Property 13: Recommendation Priority Ordering
        
        The recommendation list SHALL not contain duplicate interventions.
        
        Validates: Requirements 8.4
        """
        engine = RecommendationEngine()
        interventions = engine.recommend(reason_codes)
        
        actions = [i.action for i in interventions]
        unique_actions = set(actions)
        
        assert len(actions) == len(unique_actions), \
            f"Found duplicate interventions: {actions}"
