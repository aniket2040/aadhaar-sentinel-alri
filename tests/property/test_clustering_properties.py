"""Property-based tests for district clustering correctness.

Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
Validates: Requirements 10.2, 10.3
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.clustering.segmentation import DistrictClusterer, ClusterProfile
from tests.conftest import (
    district_strategy,
    sub_score_strategy,
    CLUSTER_LABELS,
)


# Suppress sklearn convergence warnings for tests with edge case data
# These warnings occur when KMeans finds fewer distinct clusters than requested
# due to duplicate/identical data points - expected behavior in property tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters.*:sklearn.exceptions.ConvergenceWarning"
)


# =============================================================================
# Custom Strategies for Clustering Tests
# =============================================================================

@st.composite
def district_features_strategy(draw, min_districts=4, max_districts=20):
    """Generate a DataFrame of district features for clustering.
    
    Args:
        draw: Hypothesis draw function
        min_districts: Minimum number of districts to generate
        max_districts: Maximum number of districts to generate
    
    Returns:
        DataFrame with district names and numeric feature columns
    """
    num_districts = draw(st.integers(min_value=min_districts, max_value=max_districts))
    
    # Generate unique district names
    districts = [f"district_{i}" for i in range(num_districts)]
    
    # Generate feature values for each district with some variation
    # Use different base values to ensure distinct clusters can form
    coverage_risks = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                      for _ in range(num_districts)]
    instability_risks = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                         for _ in range(num_districts)]
    biometric_risks = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                       for _ in range(num_districts)]
    anomaly_factors = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                       for _ in range(num_districts)]
    
    return pd.DataFrame({
        'district': districts,
        'coverage_risk': coverage_risks,
        'instability_risk': instability_risks,
        'biometric_risk': biometric_risks,
        'anomaly_factor': anomaly_factors
    })


@st.composite
def small_district_features_strategy(draw):
    """Generate a small DataFrame for edge case testing (1-3 districts)."""
    num_districts = draw(st.integers(min_value=1, max_value=3))
    
    districts = [f"district_{i}" for i in range(num_districts)]
    coverage_risks = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                      for _ in range(num_districts)]
    instability_risks = [draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) 
                         for _ in range(num_districts)]
    
    return pd.DataFrame({
        'district': districts,
        'coverage_risk': coverage_risks,
        'instability_risk': instability_risks
    })


@st.composite
def n_clusters_strategy(draw, max_clusters=8):
    """Generate a valid number of clusters."""
    return draw(st.integers(min_value=2, max_value=max_clusters))


class TestClusterAssignmentCompleteness:
    """Property 15: Cluster Assignment Completeness
    
    For any set of districts passed to the Clustering_Module, every district
    SHALL be assigned to exactly one cluster with a valid label.
    
    Validates: Requirements 10.2, 10.3
    """
    
    @given(district_features=district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_districts_assigned_to_exactly_one_cluster(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        For any set of districts, every district SHALL be assigned to exactly
        one cluster.
        
        Validates: Requirements 10.2
        """
        clusterer = DistrictClusterer(n_clusters=4)
        profiles = clusterer.fit_predict(district_features)
        
        # Collect all districts from all clusters
        all_assigned_districts = []
        for profile in profiles:
            all_assigned_districts.extend(profile.districts)
        
        # Get original district list
        original_districts = district_features['district'].tolist()
        
        # Every district should appear exactly once
        assert len(all_assigned_districts) == len(original_districts), \
            f"Expected {len(original_districts)} assignments, got {len(all_assigned_districts)}"
        
        assert set(all_assigned_districts) == set(original_districts), \
            f"Assigned districts don't match original districts"
    
    @given(district_features=district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_clusters_have_valid_labels(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        For any clustering result, every cluster SHALL have a valid label.
        
        Validates: Requirements 10.3
        """
        clusterer = DistrictClusterer(n_clusters=4)
        profiles = clusterer.fit_predict(district_features)
        
        for profile in profiles:
            # Label should be non-empty string
            assert isinstance(profile.label, str), \
                f"Cluster {profile.cluster_id} has non-string label: {profile.label}"
            assert len(profile.label) > 0, \
                f"Cluster {profile.cluster_id} has empty label"
            
            # Label should be from predefined set or follow pattern
            valid_labels = list(CLUSTER_LABELS) + [f'Cluster-{i}' for i in range(20)]
            assert profile.label in valid_labels, \
                f"Cluster {profile.cluster_id} has invalid label: {profile.label}"
    
    @given(
        district_features=district_features_strategy(min_districts=10, max_districts=20),
        n_clusters=n_clusters_strategy(max_clusters=6)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_configurable_cluster_count(self, district_features, n_clusters):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        For any configurable n_clusters, the clustering SHALL produce at most
        n_clusters clusters (may be fewer if fewer districts or duplicate points).
        
        Validates: Requirements 10.4
        """
        clusterer = DistrictClusterer(n_clusters=n_clusters)
        profiles = clusterer.fit_predict(district_features)
        
        num_districts = len(district_features)
        expected_max_clusters = min(n_clusters, num_districts)
        
        # Should have at most expected_max_clusters (may be fewer due to duplicate points)
        assert len(profiles) <= expected_max_clusters, \
            f"Expected at most {expected_max_clusters} clusters, got {len(profiles)}"
        
        # At least one cluster should exist
        assert len(profiles) >= 1, "Should have at least one cluster"
        
        # All returned clusters should have at least one district
        for profile in profiles:
            assert len(profile.districts) > 0, \
                f"Cluster {profile.cluster_id} has no districts"
    
    @given(district_features=district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_cluster_profiles_have_characteristics(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        For any clustering result, every cluster profile SHALL include
        characteristic metrics.
        
        Validates: Requirements 10.3
        """
        clusterer = DistrictClusterer(n_clusters=4)
        profiles = clusterer.fit_predict(district_features)
        
        # Get feature columns (excluding 'district')
        feature_cols = [col for col in district_features.columns if col != 'district']
        
        for profile in profiles:
            # Characteristics should be a dict
            assert isinstance(profile.characteristics, dict), \
                f"Cluster {profile.cluster_id} characteristics is not a dict"
            
            # Should have characteristics for each feature
            for col in feature_cols:
                assert col in profile.characteristics, \
                    f"Cluster {profile.cluster_id} missing characteristic: {col}"
                
                # Characteristic values should be valid floats (not NaN)
                value = profile.characteristics[col]
                assert isinstance(value, (int, float)), \
                    f"Characteristic {col} has invalid type: {type(value)}"
                assert not np.isnan(value), \
                    f"Characteristic {col} is NaN"
    
    @given(district_features=small_district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_handles_fewer_districts_than_clusters(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        When there are fewer districts than requested clusters, the clustering
        SHALL still assign all districts and produce valid profiles.
        
        Validates: Requirements 10.2, 10.3
        """
        # Request more clusters than districts
        clusterer = DistrictClusterer(n_clusters=10)
        profiles = clusterer.fit_predict(district_features)
        
        num_districts = len(district_features)
        
        # Should have at most num_districts clusters
        assert len(profiles) <= num_districts, \
            f"Expected at most {num_districts} clusters, got {len(profiles)}"
        
        # All districts should still be assigned
        all_assigned = []
        for profile in profiles:
            all_assigned.extend(profile.districts)
        
        assert len(all_assigned) == num_districts, \
            f"Expected {num_districts} assignments, got {len(all_assigned)}"
    
    @given(district_features=district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_no_duplicate_district_assignments(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        For any clustering result, no district SHALL appear in more than one cluster.
        
        Validates: Requirements 10.2
        """
        clusterer = DistrictClusterer(n_clusters=4)
        profiles = clusterer.fit_predict(district_features)
        
        # Collect all districts
        all_districts = []
        for profile in profiles:
            all_districts.extend(profile.districts)
        
        # Check for duplicates
        assert len(all_districts) == len(set(all_districts)), \
            "Some districts appear in multiple clusters"
    
    @given(district_features=district_features_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_cluster_ids_are_sequential(self, district_features):
        """
        Feature: aadhaar-sentinel, Property 15: Cluster Assignment Completeness
        
        Cluster IDs SHALL be sequential integers starting from 0.
        
        Validates: Requirements 10.3
        """
        clusterer = DistrictClusterer(n_clusters=4)
        profiles = clusterer.fit_predict(district_features)
        
        cluster_ids = sorted([p.cluster_id for p in profiles])
        expected_ids = list(range(len(profiles)))
        
        assert cluster_ids == expected_ids, \
            f"Cluster IDs {cluster_ids} are not sequential from 0"
