"""Data loader module for Aadhaar Sentinel ETL pipeline.

This module provides the DataLoader class for loading and validating
enrollment, demographic, and biometric CSV data files.
"""

import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when CSV schema validation fails."""
    pass


class DataLoader:
    """Loads and validates CSV data files for Aadhaar Sentinel.
    
    Handles loading of enrollment, demographic, and biometric CSV files
    with schema validation and type coercion.
    """
    
    # Schema definitions for each CSV type
    ENROLLMENT_SCHEMA: Dict[str, Any] = {
        'state': str,
        'district': str,
        'pincode': int,
        'year': int,
        'month': int,
        'day': int,
        'total_enrollment_age': int
    }
    
    DEMOGRAPHIC_SCHEMA: Dict[str, Any] = {
        'state': str,
        'district': str,
        'pincode': int,
        'year': int,
        'month': int,
        'day': int,
        'total_demographic_age': int
    }
    
    BIOMETRIC_SCHEMA: Dict[str, Any] = {
        'state': str,
        'district': str,
        'pincode': int,
        'year': int,
        'month': int,
        'day': int,
        'total_biometric_age': int
    }

    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any], 
                         data_type: str) -> None:
        """Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            schema: Expected column names and types
            data_type: Type of data for error messages
            
        Raises:
            SchemaValidationError: If required columns are missing
        """
        required_columns = set(schema.keys())
        actual_columns = set(df.columns)
        missing_columns = required_columns - actual_columns
        
        if missing_columns:
            raise SchemaValidationError(
                f"Missing required columns for {data_type} data: {missing_columns}"
            )
    
    def _coerce_types(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Apply type coercion to DataFrame columns.
        
        Args:
            df: DataFrame to coerce
            schema: Expected column types
            
        Returns:
            DataFrame with coerced types
        """
        df = df.copy()
        
        for column, dtype in schema.items():
            if column not in df.columns:
                continue
                
            try:
                if dtype == str:
                    df[column] = df[column].astype(str).str.lower().str.strip()
                elif dtype == int:
                    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"Type coercion applied to column {column}: {e}")
                
        return df
    
    def load_enrollment(self, filepath: str) -> pd.DataFrame:
        """Load enrollment CSV with schema validation.
        
        Args:
            filepath: Path to enrollment CSV file
            
        Returns:
            Validated and type-coerced DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SchemaValidationError: If schema validation fails
            ValueError: If CSV format is invalid
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {e}")
        
        self._validate_schema(df, self.ENROLLMENT_SCHEMA, 'enrollment')
        df = self._coerce_types(df, self.ENROLLMENT_SCHEMA)
        
        logger.info(f"Loaded {len(df)} enrollment records from {filepath}")
        return df
    
    def load_demographic(self, filepath: str) -> pd.DataFrame:
        """Load demographic CSV with schema validation.
        
        Args:
            filepath: Path to demographic CSV file
            
        Returns:
            Validated and type-coerced DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SchemaValidationError: If schema validation fails
            ValueError: If CSV format is invalid
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {e}")
        
        self._validate_schema(df, self.DEMOGRAPHIC_SCHEMA, 'demographic')
        df = self._coerce_types(df, self.DEMOGRAPHIC_SCHEMA)
        
        logger.info(f"Loaded {len(df)} demographic records from {filepath}")
        return df
    
    def load_biometric(self, filepath: str) -> pd.DataFrame:
        """Load biometric CSV with schema validation.
        
        Args:
            filepath: Path to biometric CSV file
            
        Returns:
            Validated and type-coerced DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SchemaValidationError: If schema validation fails
            ValueError: If CSV format is invalid
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {e}")
        
        self._validate_schema(df, self.BIOMETRIC_SCHEMA, 'biometric')
        df = self._coerce_types(df, self.BIOMETRIC_SCHEMA)
        
        logger.info(f"Loaded {len(df)} biometric records from {filepath}")
        return df
