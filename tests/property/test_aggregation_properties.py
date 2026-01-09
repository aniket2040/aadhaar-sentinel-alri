"""Property-based tests for aggregation correctness.

Feature: aadhaar-sentinel, Property 2: Aggregation Sum Invariant
Validates: Requirements 1.4
"""

import pytest
import pandas as pd
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.etl.aggregator import MonthlyAggregator

# Import strategies from conftest
from tests.conftest import (
    state_strategy,
    pincode_strategy,
    year_strategy,
    month_strategy,
    day_strategy,
    count_strategy,
)

# ASCII-only district strategy for consistency
ascii_district_strategy = st.text(
    min_size=3,
    max_size=30,
    alphabet='abcdefghijklmnopqrstuvwxyz '
).map(lambda s: s.strip().lower()).filter(lambda s: len(s) >= 3)


@st.composite
def daily_records_strategy(draw, min_records=1, max_records=20):
    """Generate a list of daily records for a single district-month."""
    state = draw(state_strategy)
    district = draw(ascii_district_strategy)
    year = draw(year_strategy)
    month = draw(month_strategy)
    
    num_records = draw(st.integers(min_value=min_records, max_value=max_records))
    
    records = []
    for _ in range(num_records):
        records.append({
            'state': state,
            'district': district,
            'pincode': draw(pincode_strategy),
            'year': year,
            'month': month,
            'day': draw(day_strategy),
            'total_enrollment_age': draw(count_strategy)
        })
    
    return records


@st.composite
def multi_district_records_strategy(draw, min_districts=1, max_districts=5):
    """Generate records for multiple districts."""
    num_districts = draw(st.integers(min_value=min_districts, max_value=max_districts))
    
    all_records = []
    for _ in range(num_districts):
        records = draw(daily_records_strategy())
        all_records.extend(records)
    
    return all_records


class TestAggregationProperties:
    """Property tests for aggregation sum invariant.
    
    Property 2: Aggregation Sum Invariant
    For any set of daily records for a district-month combination, the monthly
    aggregate total SHALL equal the sum of all daily values for that district-month.
    
    Validates: Requirements 1.4
    """
    
    @given(records=daily_records_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_single_district_sum_invariant(self, records):
        """
        Feature: aadhaar-sentinel, Property 2: Aggregation Sum Invariant
        
        For any set of daily records for a single district-month, the monthly
        aggregate total SHALL equal the sum of all daily values.
        
        Validates: Requirements 1.4
        """
        df = pd.DataFrame(records)
        aggregator = MonthlyAggregator()
        
        result = aggregator.aggregate_by_district_month(df, 'total_enrollment_age')
        
        # Calculate expected sum
        expected_sum = sum(r['total_enrollment_age'] for r in records)
        
        # Should have exactly one aggregated row
        assert len(result) == 1
        
        # The total should equal the sum of all daily values
        assert result.iloc[0]['total'] == expected_sum
        
        # Record count should match number of input records
        assert result.iloc[0]['record_count'] == len(records)
    
    @given(records=multi_district_records_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multi_district_sum_invariant(self, records):
        """
        Feature: aadhaar-sentinel, Property 2: Aggregation Sum Invariant
        
        For any set of daily records across multiple districts, each district-month
        aggregate total SHALL equal the sum of daily values for that district-month.
        
        Validates: Requirements 1.4
        """
        df = pd.DataFrame(records)
        aggregator = MonthlyAggregator()
        
        result = aggregator.aggregate_by_district_month(df, 'total_enrollment_age')
        
        # For each aggregated row, verify the sum matches
        for _, row in result.iterrows():
            # Filter original records for this district-month
            mask = (
                (df['state'] == row['state']) &
                (df['district'] == row['district']) &
                (df['year'] == row['year']) &
                (df['month'] == row['month'])
            )
            expected_sum = df.loc[mask, 'total_enrollment_age'].sum()
            expected_count = mask.sum()
            
            assert row['total'] == expected_sum
            assert row['record_count'] == expected_count
    
    @given(records=daily_records_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_total_sum_preserved(self, records):
        """
        Feature: aadhaar-sentinel, Property 2: Aggregation Sum Invariant
        
        The total sum across all aggregated rows SHALL equal the total sum
        of all input records.
        
        Validates: Requirements 1.4
        """
        df = pd.DataFrame(records)
        aggregator = MonthlyAggregator()
        
        result = aggregator.aggregate_by_district_month(df, 'total_enrollment_age')
        
        # Total sum should be preserved
        input_total = df['total_enrollment_age'].sum()
        output_total = result['total'].sum()
        
        assert input_total == output_total
    
    @given(records=multi_district_records_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_record_count_preserved(self, records):
        """
        Feature: aadhaar-sentinel, Property 2: Aggregation Sum Invariant
        
        The total record count across all aggregated rows SHALL equal the
        number of input records.
        
        Validates: Requirements 1.4
        """
        df = pd.DataFrame(records)
        aggregator = MonthlyAggregator()
        
        result = aggregator.aggregate_by_district_month(df, 'total_enrollment_age')
        
        # Total record count should be preserved
        assert result['record_count'].sum() == len(records)
