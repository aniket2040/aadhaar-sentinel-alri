"""
Property-based tests for ALRI Storage Module.

This module tests:
- Property 16: ALRI Record Serialization Round-Trip
- Property 17: Query Filter Correctness

Requirements: 13.3, 13.4
"""

import os
import tempfile
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.storage.serializer import (
    ALRIRecord,
    ALRISerializer,
    ALRIStorage,
    SerializationError
)
from tests.conftest import (
    alri_record_strategy,
    district_strategy,
    state_strategy,
    sub_scores_strategy,
    reason_code_strategy
)


# =============================================================================
# Property 16: ALRI Record Serialization Round-Trip
# Feature: aadhaar-sentinel, Property 16: ALRI Record Serialization Round-Trip
# Validates: Requirements 13.4
# =============================================================================

@settings(max_examples=100)
@given(record_data=alri_record_strategy())
def test_property_16_serialization_round_trip(record_data):
    """
    Property 16: ALRI Record Serialization Round-Trip
    
    For any valid ALRIRecord, serializing to JSON then deserializing 
    SHALL produce an equivalent record with identical field values.
    
    **Validates: Requirements 13.4**
    """
    # Create ALRIRecord from generated data
    original_record = ALRIRecord(
        district=record_data['district'],
        state=record_data['state'],
        alri_score=record_data['alri_score'],
        sub_scores=record_data['sub_scores'],
        reason_codes=record_data['reason_codes'],
        computed_at=record_data['computed_at']
    )
    
    # Serialize and deserialize
    serializer = ALRISerializer()
    json_str = serializer.serialize(original_record)
    deserialized_record = serializer.deserialize(json_str)
    
    # Verify round-trip produces equivalent record
    assert deserialized_record.district == original_record.district
    assert deserialized_record.state == original_record.state
    assert abs(deserialized_record.alri_score - original_record.alri_score) < 1e-9
    assert deserialized_record.reason_codes == original_record.reason_codes
    assert deserialized_record.computed_at == original_record.computed_at
    
    # Verify sub_scores match
    for key in original_record.sub_scores:
        assert key in deserialized_record.sub_scores
        assert abs(deserialized_record.sub_scores[key] - original_record.sub_scores[key]) < 1e-9


# =============================================================================
# Property 17: Query Filter Correctness
# Feature: aadhaar-sentinel, Property 17: Query Filter Correctness
# Validates: Requirements 13.3
# =============================================================================

@st.composite
def records_with_filters_strategy(draw):
    """Generate a list of records and valid filter parameters."""
    # Generate multiple records
    num_records = draw(st.integers(min_value=1, max_value=20))
    records_data = [draw(alri_record_strategy()) for _ in range(num_records)]
    
    # Create ALRIRecord objects
    records = [
        ALRIRecord(
            district=r['district'],
            state=r['state'],
            alri_score=r['alri_score'],
            sub_scores=r['sub_scores'],
            reason_codes=r['reason_codes'],
            computed_at=r['computed_at']
        )
        for r in records_data
    ]
    
    # Generate filter parameters (may or may not match any records)
    filter_district = draw(st.one_of(
        st.none(),
        st.sampled_from([r.district for r in records]) if records else st.none()
    ))
    
    filter_min_score = draw(st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
    ))
    
    # Generate date range from existing computed_at values
    if records:
        dates = sorted([r.computed_at for r in records])
        filter_date_range = draw(st.one_of(
            st.none(),
            st.just((dates[0], dates[-1]))
        ))
    else:
        filter_date_range = None
    
    return {
        'records': records,
        'district': filter_district,
        'min_score': filter_min_score,
        'date_range': filter_date_range
    }


@settings(max_examples=100)
@given(test_data=records_with_filters_strategy())
def test_property_17_query_filter_correctness(test_data):
    """
    Property 17: Query Filter Correctness
    
    For any query with district, date_range, or min_score filters, 
    all returned records SHALL satisfy all specified filter conditions.
    
    **Validates: Requirements 13.3**
    """
    records = test_data['records']
    district_filter = test_data['district']
    min_score_filter = test_data['min_score']
    date_range_filter = test_data['date_range']
    
    # Create storage and load records
    storage = ALRIStorage()
    storage._records = records  # Directly set records for testing
    
    # Execute query
    results = storage.query(
        district=district_filter,
        date_range=date_range_filter,
        min_score=min_score_filter
    )
    
    # Verify all returned records satisfy ALL filter conditions
    for record in results:
        # Check district filter
        if district_filter is not None:
            assert record.district.lower() == district_filter.lower(), \
                f"Record district '{record.district}' does not match filter '{district_filter}'"
        
        # Check date range filter
        if date_range_filter is not None:
            start_date, end_date = date_range_filter
            assert start_date <= record.computed_at <= end_date, \
                f"Record date '{record.computed_at}' not in range [{start_date}, {end_date}]"
        
        # Check min_score filter
        if min_score_filter is not None:
            assert record.alri_score >= min_score_filter, \
                f"Record score {record.alri_score} is less than min_score {min_score_filter}"


@settings(max_examples=100)
@given(test_data=records_with_filters_strategy())
def test_property_17_query_returns_subset(test_data):
    """
    Property 17 (additional): Query results are always a subset of stored records.
    
    For any query, the returned records SHALL be a subset of the stored records.
    
    **Validates: Requirements 13.3**
    """
    records = test_data['records']
    district_filter = test_data['district']
    min_score_filter = test_data['min_score']
    date_range_filter = test_data['date_range']
    
    # Create storage and load records
    storage = ALRIStorage()
    storage._records = records
    
    # Execute query
    results = storage.query(
        district=district_filter,
        date_range=date_range_filter,
        min_score=min_score_filter
    )
    
    # Verify results are a subset of original records
    assert len(results) <= len(records), \
        f"Query returned {len(results)} records but only {len(records)} were stored"
    
    for result in results:
        assert result in records, \
            f"Query returned a record not in the original set"
