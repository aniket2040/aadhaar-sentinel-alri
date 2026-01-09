"""Monthly aggregation module for Aadhaar Sentinel ETL pipeline.

This module provides the MonthlyAggregator class for aggregating daily data
to monthly district-level summaries.
"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MonthlyAggregator:
    """Aggregates daily data to monthly district-level summaries.
    
    Groups records by state, district, year, and month to produce
    monthly aggregates for analysis.
    """
    
    def aggregate_by_district_month(
        self, 
        df: pd.DataFrame, 
        value_column: str
    ) -> pd.DataFrame:
        """Group by state, district, year, month and sum totals.
        
        Args:
            df: DataFrame with daily records containing state, district,
                year, month, day, and a value column
            value_column: Name of the column to aggregate (e.g., 
                'total_enrollment_age', 'total_demographic_age')
        
        Returns:
            DataFrame with monthly aggregates containing:
            - state, district, year, month
            - total (sum of value_column)
            - record_count (number of daily records)
            - pincode_count (number of unique pincodes)
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'state', 'district', 'year', 'month', 
                'total', 'record_count', 'pincode_count'
            ])
        
        # Validate required columns
        required_cols = ['state', 'district', 'year', 'month', value_column]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Group by state, district, year, month
        grouped = df.groupby(['state', 'district', 'year', 'month'])
        
        # Compute aggregates
        result = grouped.agg(
            total=(value_column, 'sum'),
            record_count=(value_column, 'count'),
            pincode_count=('pincode', 'nunique') if 'pincode' in df.columns 
                          else (value_column, 'count')
        ).reset_index()
        
        logger.info(f"Aggregated {len(df)} daily records into "
                   f"{len(result)} monthly aggregates")
        
        return result
    
    def compute_baselines(
        self, 
        df: pd.DataFrame,
        value_column: str = 'total'
    ) -> pd.DataFrame:
        """Compute district-level baseline metrics for normalization.
        
        Calculates mean, std, min, max for each district across all
        available months to establish baseline metrics.
        
        Args:
            df: DataFrame with monthly aggregates (output of 
                aggregate_by_district_month)
            value_column: Column to compute baselines for
        
        Returns:
            DataFrame with baseline metrics per district:
            - state, district
            - baseline_mean, baseline_std, baseline_min, baseline_max
            - month_count (number of months of data)
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'state', 'district', 'baseline_mean', 'baseline_std',
                'baseline_min', 'baseline_max', 'month_count'
            ])
        
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found in DataFrame")
        
        # Group by state and district
        grouped = df.groupby(['state', 'district'])
        
        # Compute baseline statistics
        result = grouped.agg(
            baseline_mean=(value_column, 'mean'),
            baseline_std=(value_column, 'std'),
            baseline_min=(value_column, 'min'),
            baseline_max=(value_column, 'max'),
            month_count=(value_column, 'count')
        ).reset_index()
        
        # Fill NaN std (when only 1 data point) with 0
        result['baseline_std'] = result['baseline_std'].fillna(0)
        
        logger.info(f"Computed baselines for {len(result)} districts")
        
        return result
