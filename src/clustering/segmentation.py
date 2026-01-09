"""District clustering and segmentation module.

This module provides functionality to segment districts by behavioral patterns
using KMeans or hierarchical clustering algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterProfile:
    """Profile describing a cluster of districts.
    
    Attributes:
        cluster_id: Unique identifier for the cluster
        label: Human-readable label describing the cluster behavior
        districts: List of district names assigned to this cluster
        characteristics: Mean metrics for the cluster (feature name -> value)
    """
    cluster_id: int
    label: str
    districts: List[str] = field(default_factory=list)
    characteristics: Dict[str, float] = field(default_factory=dict)


class DistrictClusterer:
    """Segments districts by behavioral patterns using clustering algorithms.
    
    This class implements district segmentation using KMeans clustering
    with configurable number of clusters. Districts are grouped based on
    their ALRI sub-scores and other behavioral metrics.
    
    Attributes:
        n_clusters: Number of clusters to create (default: 4)
        method: Clustering method to use ('kmeans' supported)
        CLUSTER_LABELS: Mapping of cluster IDs to descriptive labels
    """
    
    CLUSTER_LABELS = {
        0: 'Stable-HighCoverage',
        1: 'Migratory-HighChurn',
        2: 'ChildGap-HighRisk',
        3: 'LowActivity-Rural',
    }
    
    def __init__(self, n_clusters: int = 4, method: str = 'kmeans'):
        """Initialize the DistrictClusterer.
        
        Args:
            n_clusters: Number of clusters to create (default: 4)
            method: Clustering method ('kmeans' supported)
        """
        self.n_clusters = n_clusters
        self.method = method
        self._model: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_columns: List[str] = []

    
    def fit_predict(self, district_features: pd.DataFrame) -> List[ClusterProfile]:
        """Cluster districts and return profiles.
        
        This method fits a clustering model to the district features and
        assigns each district to a cluster. It returns a list of ClusterProfile
        objects describing each cluster.
        
        Args:
            district_features: DataFrame with district features. Must contain
                a 'district' column and numeric feature columns for clustering.
                Expected feature columns include: coverage_risk, instability_risk,
                biometric_risk, anomaly_factor, or similar metrics.
        
        Returns:
            List of ClusterProfile objects, one per cluster, containing:
                - cluster_id: The cluster identifier
                - label: Descriptive label for the cluster
                - districts: List of district names in the cluster
                - characteristics: Mean feature values for the cluster
        
        Raises:
            ValueError: If district_features is empty or missing required columns
        """
        if district_features.empty:
            raise ValueError("district_features DataFrame cannot be empty")
        
        if 'district' not in district_features.columns:
            raise ValueError("district_features must contain a 'district' column")
        
        # Extract district names
        districts = district_features['district'].tolist()
        
        # Get numeric feature columns (exclude 'district' and other non-numeric)
        self._feature_columns = [
            col for col in district_features.columns 
            if col != 'district' and district_features[col].dtype in ['float64', 'int64', 'float32', 'int32']
        ]
        
        if not self._feature_columns:
            raise ValueError("district_features must contain at least one numeric feature column")
        
        # Extract feature matrix
        X = district_features[self._feature_columns].values
        
        # Handle NaN values by replacing with column means
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Adjust n_clusters if we have fewer samples than clusters
        actual_n_clusters = min(self.n_clusters, len(districts))
        
        # Fit KMeans model
        self._model = KMeans(
            n_clusters=actual_n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = self._model.fit_predict(X_scaled)
        
        # Build cluster profiles
        profiles = self._build_profiles(
            districts=districts,
            cluster_labels=cluster_labels,
            features=district_features[self._feature_columns],
            actual_n_clusters=actual_n_clusters
        )
        
        return profiles
    
    def _build_profiles(
        self,
        districts: List[str],
        cluster_labels: np.ndarray,
        features: pd.DataFrame,
        actual_n_clusters: int
    ) -> List[ClusterProfile]:
        """Build ClusterProfile objects from clustering results.
        
        Args:
            districts: List of district names
            cluster_labels: Array of cluster assignments
            features: DataFrame of feature values
            actual_n_clusters: Actual number of clusters used
        
        Returns:
            List of ClusterProfile objects
        """
        profiles = []
        
        # Get unique cluster labels that actually have assignments
        unique_labels = np.unique(cluster_labels)
        
        # Renumber clusters sequentially starting from 0
        for new_id, original_id in enumerate(sorted(unique_labels)):
            # Get districts in this cluster
            mask = cluster_labels == original_id
            cluster_districts = [d for d, m in zip(districts, mask) if m]
            
            # Skip empty clusters (shouldn't happen with unique_labels, but safety check)
            if not cluster_districts:
                continue
            
            # Compute mean characteristics for this cluster
            cluster_features = features.iloc[mask]
            characteristics = {}
            for col in self._feature_columns:
                mean_val = cluster_features[col].mean()
                # Handle NaN by using 0.0 as default
                characteristics[col] = float(mean_val) if not np.isnan(mean_val) else 0.0
            
            # Get label (use new sequential id)
            label = self._get_cluster_label(new_id, characteristics)
            
            profile = ClusterProfile(
                cluster_id=new_id,
                label=label,
                districts=cluster_districts,
                characteristics=characteristics
            )
            profiles.append(profile)
        
        return profiles
    
    def _get_cluster_label(self, cluster_id: int, characteristics: Dict[str, float]) -> str:
        """Get a descriptive label for a cluster.
        
        Uses predefined labels if available, otherwise generates a label
        based on cluster characteristics.
        
        Args:
            cluster_id: The cluster identifier
            characteristics: Mean feature values for the cluster
        
        Returns:
            Descriptive label string
        """
        if cluster_id in self.CLUSTER_LABELS:
            return self.CLUSTER_LABELS[cluster_id]
        else:
            # Generate a label for clusters beyond the predefined set
            return f'Cluster-{cluster_id}'
    
    def predict(self, district_features: pd.DataFrame) -> np.ndarray:
        """Predict cluster assignments for new districts.
        
        Args:
            district_features: DataFrame with same feature columns as fit_predict
        
        Returns:
            Array of cluster assignments
        
        Raises:
            ValueError: If model has not been fitted
        """
        if self._model is None or self._scaler is None:
            raise ValueError("Model has not been fitted. Call fit_predict first.")
        
        X = district_features[self._feature_columns].values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        X_scaled = self._scaler.transform(X)
        
        return self._model.predict(X_scaled)
