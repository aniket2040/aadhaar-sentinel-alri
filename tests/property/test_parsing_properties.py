"""Property-based tests for CSV parsing correctness.

Feature: aadhaar-sentinel, Property 1: CSV Parsing Correctness
Validates: Requirements 1.1, 1.2, 1.3
"""

import pytest
import pandas as pd
import tempfile
import os
from hypothesis import given, settings
from hypothesis import strategies as st

from src.etl.data_loader import DataLoader

# Import strategies from conftest
from tests.conftest import (
    state_strategy,
    pincode_strategy,
    year_strategy,
    month_strategy,
    day_strategy,
    count_strategy,
)

# ASCII-only district strategy for CSV compatibility
# Exclude values that pandas might interpret as null/nan
ascii_district_strategy = st.text(
    min_size=3,
    max_size=30,
    alphabet='abcdefghijklmnopqrstuvwxyz '
).map(lambda s: s.strip().lower()).filter(
    lambda s: len(s) >= 3 and s not in ('null', 'nan', 'none', 'na', 'n/a')
)


@st.composite
def enrollment_row_ascii(draw):
    """Generate a valid enrollment CSV row with ASCII-only district names."""
    return {
        'state': draw(state_strategy),
        'district': draw(ascii_district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_enrollment_age': draw(count_strategy)
    }


@st.composite
def demographic_row_ascii(draw):
    """Generate a valid demographic CSV row with ASCII-only district names."""
    return {
        'state': draw(state_strategy),
        'district': draw(ascii_district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_demographic_age': draw(count_strategy)
    }


@st.composite
def biometric_row_ascii(draw):
    """Generate a valid biometric CSV row with ASCII-only district names."""
    return {
        'state': draw(state_strategy),
        'district': draw(ascii_district_strategy),
        'pincode': draw(pincode_strategy),
        'year': draw(year_strategy),
        'month': draw(month_strategy),
        'day': draw(day_strategy),
        'total_biometric_age': draw(count_strategy)
    }


class TestCSVParsingProperties:
    """Property tests for CSV parsing correctness.
    
    Property 1: CSV Parsing Correctness
    For any valid CSV row containing state, district, pincode, year, month, day,
    and count fields, parsing the row SHALL extract all fields with correct types
    and values matching the original data.
    
    Validates: Requirements 1.1, 1.2, 1.3
    """
    
    @given(row=enrollment_row_ascii())
    @settings(max_examples=100)
    def test_enrollment_parsing_preserves_values(self, row):
        """
        Feature: aadhaar-sentinel, Property 1: CSV Parsing Correctness
        
        For any valid enrollment CSV row, parsing SHALL extract all fields
        with correct types and values matching the original data.
        
        Validates: Requirements 1.1
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                          delete=False, encoding='utf-8') as f:
            f.write('state,district,pincode,year,month,day,total_enrollment_age\n')
            f.write(f"{row['state']},{row['district']},{row['pincode']},"
                   f"{row['year']},{row['month']},{row['day']},"
                   f"{row['total_enrollment_age']}\n")
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load_enrollment(temp_path)
            
            assert len(df) == 1
            parsed_row = df.iloc[0]
            
            # Check string fields (normalized to lowercase)
            assert parsed_row['state'] == row['state'].lower().strip()
            assert parsed_row['district'] == row['district'].lower().strip()
            
            # Check integer fields
            assert parsed_row['pincode'] == row['pincode']
            assert parsed_row['year'] == row['year']
            assert parsed_row['month'] == row['month']
            assert parsed_row['day'] == row['day']
            assert parsed_row['total_enrollment_age'] == row['total_enrollment_age']
            
        finally:
            os.unlink(temp_path)
    
    @given(row=demographic_row_ascii())
    @settings(max_examples=100)
    def test_demographic_parsing_preserves_values(self, row):
        """
        Feature: aadhaar-sentinel, Property 1: CSV Parsing Correctness
        
        For any valid demographic CSV row, parsing SHALL extract all fields
        with correct types and values matching the original data.
        
        Validates: Requirements 1.2
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                          delete=False, encoding='utf-8') as f:
            f.write('state,district,pincode,year,month,day,total_demographic_age\n')
            f.write(f"{row['state']},{row['district']},{row['pincode']},"
                   f"{row['year']},{row['month']},{row['day']},"
                   f"{row['total_demographic_age']}\n")
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load_demographic(temp_path)
            
            assert len(df) == 1
            parsed_row = df.iloc[0]
            
            assert parsed_row['state'] == row['state'].lower().strip()
            assert parsed_row['district'] == row['district'].lower().strip()
            assert parsed_row['pincode'] == row['pincode']
            assert parsed_row['year'] == row['year']
            assert parsed_row['month'] == row['month']
            assert parsed_row['day'] == row['day']
            assert parsed_row['total_demographic_age'] == row['total_demographic_age']
            
        finally:
            os.unlink(temp_path)
    
    @given(row=biometric_row_ascii())
    @settings(max_examples=100)
    def test_biometric_parsing_preserves_values(self, row):
        """
        Feature: aadhaar-sentinel, Property 1: CSV Parsing Correctness
        
        For any valid biometric CSV row, parsing SHALL extract all fields
        with correct types and values matching the original data.
        
        Validates: Requirements 1.3
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                          delete=False, encoding='utf-8') as f:
            f.write('state,district,pincode,year,month,day,total_biometric_age\n')
            f.write(f"{row['state']},{row['district']},{row['pincode']},"
                   f"{row['year']},{row['month']},{row['day']},"
                   f"{row['total_biometric_age']}\n")
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load_biometric(temp_path)
            
            assert len(df) == 1
            parsed_row = df.iloc[0]
            
            assert parsed_row['state'] == row['state'].lower().strip()
            assert parsed_row['district'] == row['district'].lower().strip()
            assert parsed_row['pincode'] == row['pincode']
            assert parsed_row['year'] == row['year']
            assert parsed_row['month'] == row['month']
            assert parsed_row['day'] == row['day']
            assert parsed_row['total_biometric_age'] == row['total_biometric_age']
            
        finally:
            os.unlink(temp_path)
    
    @given(rows=st.lists(enrollment_row_ascii(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_multiple_rows_parsing(self, rows):
        """
        Feature: aadhaar-sentinel, Property 1: CSV Parsing Correctness
        
        For any list of valid CSV rows, parsing SHALL extract all rows
        with correct count matching input.
        
        Validates: Requirements 1.1, 1.2, 1.3
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                          delete=False, encoding='utf-8') as f:
            f.write('state,district,pincode,year,month,day,total_enrollment_age\n')
            for row in rows:
                f.write(f"{row['state']},{row['district']},{row['pincode']},"
                       f"{row['year']},{row['month']},{row['day']},"
                       f"{row['total_enrollment_age']}\n")
            temp_path = f.name
        
        try:
            loader = DataLoader()
            df = loader.load_enrollment(temp_path)
            
            # Verify row count matches
            assert len(df) == len(rows)
            
        finally:
            os.unlink(temp_path)
