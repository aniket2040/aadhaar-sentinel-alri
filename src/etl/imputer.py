"""Missing value handling module for Aadhaar Sentinel ETL pipeline.

This module provides the MissingValueHandler class for handling missing
values with documented imputation rules.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImputationLog:
    """Record of imputation actions for audit trail."""
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def add(self, column: str, row_index: int, original_value: Any, 
            imputed_value: Any, strategy: str):
        """Add an imputation record."""
        self.records.append({
            'timestamp': datetime.now().isoformat(),
            'column': column,
            'row_index': row_index,
            'original_value': str(original_value),
            'imputed_value': str(imputed_value),
            'strategy': strategy
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert log to DataFrame."""
        return pd.DataFrame(self.records)
    
    def __len__(self) -> int:
        return len(self.records)


class MissingValueHandler:
    """Handles missing values with documented rules.
    
    Supports median imputation strategy and logs all affected records
    with timestamps for audit trails.
    """
    
    SUPPORTED_STRATEGIES = ['median', 'mean', 'zero']
    
    def __init__(self):
        self.imputation_log = ImputationLog()
    
    def impute(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        strategy: str = 'median'
    ) -> pd.DataFrame:
        """Apply imputation and log affected records.
        
        Args:
            df: DataFrame with potential missing values
            columns: List of columns to impute. If None, imputes all
                    numeric columns with missing values.
            strategy: Imputation strategy ('median', 'mean', 'zero')
        
        Returns:
            DataFrame with imputed values
        
        Raises:
            ValueError: If unsupported strategy is specified
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )
        
        df = df.copy()
        
        # Determine columns to impute
        if columns is None:
            # Find numeric columns with missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if df[col].isna().any()]
        
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            # Find rows with missing values
            missing_mask = df[column].isna()
            missing_indices = df.index[missing_mask].tolist()
            
            if not missing_indices:
                continue
            
            # Calculate imputation value
            if strategy == 'median':
                impute_value = df[column].median()
            elif strategy == 'mean':
                impute_value = df[column].mean()
            elif strategy == 'zero':
                impute_value = 0
            
            # Handle case where all values are NaN
            if pd.isna(impute_value):
                impute_value = 0
                logger.warning(
                    f"All values in '{column}' are NaN, using 0 for imputation"
                )
            
            # Log each imputation
            for idx in missing_indices:
                self.imputation_log.add(
                    column=column,
                    row_index=idx,
                    original_value=df.loc[idx, column],
                    imputed_value=impute_value,
                    strategy=strategy
                )
            
            # Apply imputation
            df.loc[missing_mask, column] = impute_value
            
            logger.info(
                f"Imputed {len(missing_indices)} missing values in '{column}' "
                f"using {strategy} strategy (value: {impute_value})"
            )
        
        return df
    
    def get_imputation_log(self) -> pd.DataFrame:
        """Get the imputation log as a DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, column, row_index,
            original_value, imputed_value, strategy
        """
        return self.imputation_log.to_dataframe()
    
    def get_affected_count(self) -> int:
        """Get the total number of imputed values.
        
        Returns:
            Count of imputed values
        """
        return len(self.imputation_log)
    
    def clear_log(self):
        """Clear the imputation log."""
        self.imputation_log = ImputationLog()
