# Design Document

## Overview

Aadhaar Sentinel is a Python-based decision-support platform that computes the Aadhaar Lifecycle Risk Index (ALRI) from three datasets: enrollment, demographic updates, and biometric updates. The system aggregates data at district/PIN level, computes four sub-scores (Coverage Risk, Data Instability Risk, Biometric Compliance Risk, Anomaly Factor), combines them into a composite ALRI score (0-100), generates explainable reason codes, and recommends prioritized interventions.

The platform consists of:
- ETL pipeline for data ingestion and monthly aggregation
- Risk scoring engine with configurable weights
- Anomaly detection using STL decomposition
- Time-series forecasting (Prophet/SARIMA)
- District clustering for behavioral segmentation
- Interactive Streamlit dashboard with heatmaps and alerts
- Automated PDF report generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Aadhaar Sentinel Platform                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ enrollment   │    │ demographic  │    │ biometric    │   Data Sources    │
│  │    .csv      │    │    .csv      │    │    .csv      │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ETL Pipeline                                   │   │
│  │  • Data loading & validation                                          │   │
│  │  • Monthly aggregation by district                                    │   │
│  │  • Missing value imputation                                           │   │
│  │  • Versioned dataset creation                                         │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     ALRI Calculator                                   │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐         │   │
│  │  │ Coverage   │ │ Instability│ │ Biometric  │ │ Anomaly    │         │   │
│  │  │ Risk (C)   │ │ Risk (D)   │ │ Risk (B)   │ │ Factor (A) │         │   │
│  │  │   0.30     │ │   0.30     │ │   0.30     │ │   0.10     │         │   │
│  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘         │   │
│  │        └──────────────┴──────────────┴──────────────┘                │   │
│  │                             ▼                                         │   │
│  │              ALRI = 0.30C + 0.30D + 0.30B + 0.10A                     │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Reason Code    │  │ Forecasting    │  │ Clustering     │                 │
│  │ Generator      │  │ Module         │  │ Module         │                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  Recommendation Engine                                │   │
│  │  • Maps reason codes to interventions                                 │   │
│  │  • Prioritizes low-cost, high-impact actions                          │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Presentation Layer                                │   │
│  │  ┌────────────────────┐    ┌────────────────────┐                     │   │
│  │  │ Streamlit Dashboard│    │ PDF Report Generator│                    │   │
│  │  │ • District heatmap │    │ • Top-10 at-risk    │                    │   │
│  │  │ • Time-series      │    │ • Recommendations   │                    │   │
│  │  │ • Alerts panel     │    │ • Trend analysis    │                    │   │
│  │  └────────────────────┘    └────────────────────┘                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. ETL Pipeline (`src/etl/`)

```python
# src/etl/data_loader.py
class DataLoader:
    """Loads and validates CSV data files."""
    
    def load_enrollment(self, filepath: str) -> pd.DataFrame:
        """Load enrollment CSV with schema validation."""
        pass
    
    def load_demographic(self, filepath: str) -> pd.DataFrame:
        """Load demographic CSV with schema validation."""
        pass
    
    def load_biometric(self, filepath: str) -> pd.DataFrame:
        """Load biometric CSV with schema validation."""
        pass

# src/etl/aggregator.py
class MonthlyAggregator:
    """Aggregates daily data to monthly district-level summaries."""
    
    def aggregate_by_district_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group by state, district, year, month and sum totals."""
        pass
    
    def compute_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute district-level baseline metrics for normalization."""
        pass

# src/etl/imputer.py
class MissingValueHandler:
    """Handles missing values with documented rules."""
    
    def impute(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Apply imputation and log affected records."""
        pass
```

### 2. ALRI Calculator (`src/scoring/`)

```python
# src/scoring/alri_calculator.py
@dataclass
class ALRIWeights:
    coverage: float = 0.30
    instability: float = 0.30
    biometric: float = 0.30
    anomaly: float = 0.10

@dataclass
class ALRIResult:
    district: str
    state: str
    alri_score: float  # 0-100
    coverage_risk: float  # 0-1
    instability_risk: float  # 0-1
    biometric_risk: float  # 0-1
    anomaly_factor: float  # 0-1
    reason_codes: List[ReasonCode]
    recommendations: List[Intervention]
    computed_at: datetime

class ALRICalculator:
    """Computes ALRI scores from aggregated data."""
    
    def __init__(self, weights: ALRIWeights = None):
        self.weights = weights or ALRIWeights()
    
    def compute_coverage_risk(self, district_data: pd.DataFrame) -> float:
        """Compute coverage risk sub-score (0-1)."""
        pass
    
    def compute_instability_risk(self, district_data: pd.DataFrame) -> float:
        """Compute data instability risk sub-score (0-1)."""
        pass
    
    def compute_biometric_risk(self, district_data: pd.DataFrame) -> float:
        """Compute biometric compliance risk sub-score (0-1)."""
        pass
    
    def compute_anomaly_factor(self, district_data: pd.DataFrame) -> float:
        """Compute anomaly factor sub-score (0-1)."""
        pass
    
    def compute_alri(self, district_data: pd.DataFrame) -> ALRIResult:
        """Compute composite ALRI score (0-100) with all components."""
        pass
```

### 3. Anomaly Detector (`src/anomaly/`)

```python
# src/anomaly/detector.py
@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float  # 0-1
    anomaly_type: str  # 'spike', 'drop', 'normal'
    residual_zscore: float
    timestamp: datetime

class STLAnomalyDetector:
    """Detects anomalies using STL decomposition."""
    
    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold
    
    def decompose(self, time_series: pd.Series) -> Dict[str, pd.Series]:
        """Apply STL decomposition to extract trend, seasonal, residual."""
        pass
    
    def detect_anomalies(self, time_series: pd.Series) -> List[AnomalyResult]:
        """Identify anomalies where residual > threshold."""
        pass
    
    def compute_anomaly_factor(self, anomalies: List[AnomalyResult]) -> float:
        """Convert anomaly results to 0-1 factor score."""
        pass
```

### 4. Reason Code Generator (`src/explainability/`)

```python
# src/explainability/reason_codes.py
class Severity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class ReasonCode:
    code: str  # e.g., 'Low_Child_Enrolment'
    description: str
    severity: Severity
    contribution: float  # How much this factor contributed to ALRI
    affected_population: int  # Estimated people affected

class ReasonCodeGenerator:
    """Generates explainable reason codes from sub-scores."""
    
    REASON_CODE_MAP = {
        'coverage': 'Low_Child_Enrolment',
        'instability': 'High_Address_Churn',
        'biometric': 'Low_Biometric_Update_5to15',
        'anomaly': 'Anomalous_Data_Entry'
    }
    
    def generate(self, alri_result: ALRIResult) -> List[ReasonCode]:
        """Generate ranked reason codes from ALRI components."""
        pass
    
    def determine_severity(self, score: float) -> Severity:
        """Map score to severity level."""
        pass
```

### 5. Recommendation Engine (`src/recommendations/`)

```python
# src/recommendations/engine.py
@dataclass
class Intervention:
    action: str  # e.g., 'Mobile Van Deployment'
    description: str
    estimated_cost: str  # 'Low', 'Medium', 'High'
    estimated_impact: int  # People affected
    priority: int  # 1 = highest

class RecommendationEngine:
    """Maps reason codes to actionable interventions."""
    
    INTERVENTION_MAP = {
        'Low_Child_Enrolment': [
            Intervention('School Enrollment Drive', ...),
            Intervention('Mobile Van Deployment', ...),
        ],
        'High_Address_Churn': [
            Intervention('SMS/IVR Campaign', ...),
            Intervention('Additional Update Kiosks', ...),
        ],
        'Low_Biometric_Update_5to15': [
            Intervention('Free Biometric Camp', ...),
            Intervention('School Biometric Drive', ...),
        ],
        'Anomalous_Data_Entry': [
            Intervention('Data Quality Audit', ...),
        ],
    }
    
    def recommend(self, reason_codes: List[ReasonCode]) -> List[Intervention]:
        """Generate prioritized intervention list."""
        pass
```

### 6. Forecasting Module (`src/forecasting/`)

```python
# src/forecasting/forecaster.py
@dataclass
class ForecastResult:
    district: str
    metric: str  # 'enrollment', 'demographic', 'biometric'
    forecast_values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    forecast_dates: List[datetime]
    trend: str  # 'increasing', 'decreasing', 'stable'

class TimeSeriesForecaster:
    """Forecasts future volumes using Prophet or SARIMA."""
    
    def __init__(self, model_type: str = 'prophet', horizon_months: int = 6):
        self.model_type = model_type
        self.horizon_months = horizon_months
    
    def fit(self, time_series: pd.DataFrame) -> None:
        """Fit forecasting model to historical data."""
        pass
    
    def predict(self) -> ForecastResult:
        """Generate forecast with confidence intervals."""
        pass
    
    def detect_declining_trend(self, forecast: ForecastResult) -> bool:
        """Flag if forecast shows declining trend."""
        pass
```

### 7. Clustering Module (`src/clustering/`)

```python
# src/clustering/segmentation.py
@dataclass
class ClusterProfile:
    cluster_id: int
    label: str  # e.g., 'Stable-HighCoverage'
    districts: List[str]
    characteristics: Dict[str, float]  # Mean metrics for cluster

class DistrictClusterer:
    """Segments districts by behavioral patterns."""
    
    CLUSTER_LABELS = {
        0: 'Stable-HighCoverage',
        1: 'Migratory-HighChurn',
        2: 'ChildGap-HighRisk',
        3: 'LowActivity-Rural',
    }
    
    def __init__(self, n_clusters: int = 4, method: str = 'kmeans'):
        self.n_clusters = n_clusters
        self.method = method
    
    def fit_predict(self, district_features: pd.DataFrame) -> List[ClusterProfile]:
        """Cluster districts and return profiles."""
        pass
```

### 8. Dashboard (`src/dashboard/`)

```python
# src/dashboard/app.py (Streamlit)
class AadhaarSentinelDashboard:
    """Interactive Streamlit dashboard."""
    
    def render_heatmap(self, alri_scores: pd.DataFrame) -> None:
        """Render district heatmap colored by ALRI."""
        pass
    
    def render_district_detail(self, district: str) -> None:
        """Render time-series and metrics for selected district."""
        pass
    
    def render_alerts_panel(self, alerts: List[ALRIResult]) -> None:
        """Render alerts for districts exceeding thresholds."""
        pass
    
    def render_filters(self) -> Dict[str, Any]:
        """Render state/district/time filters."""
        pass
```

### 9. Report Generator (`src/reports/`)

```python
# src/reports/generator.py
class PDFReportGenerator:
    """Generates automated PDF reports."""
    
    def generate_state_report(self, state: str, alri_results: List[ALRIResult]) -> bytes:
        """Generate PDF with top-10 at-risk districts for state."""
        pass
    
    def generate_district_report(self, district: str, alri_result: ALRIResult) -> bytes:
        """Generate detailed PDF for single district."""
        pass
```

### 10. Storage Module (`src/storage/`)

```python
# src/storage/serializer.py
@dataclass
class ALRIRecord:
    district: str
    state: str
    alri_score: float
    sub_scores: Dict[str, float]
    reason_codes: List[str]
    computed_at: str  # ISO format

class ALRISerializer:
    """Serializes and deserializes ALRI records to JSON."""
    
    def serialize(self, record: ALRIRecord) -> str:
        """Convert ALRIRecord to JSON string."""
        pass
    
    def deserialize(self, json_str: str) -> ALRIRecord:
        """Convert JSON string to ALRIRecord."""
        pass

class ALRIStorage:
    """Persists ALRI scores to JSON files."""
    
    def save(self, records: List[ALRIRecord], filepath: str) -> None:
        """Save records to JSON file."""
        pass
    
    def load(self, filepath: str) -> List[ALRIRecord]:
        """Load records from JSON file."""
        pass
    
    def query(self, district: str = None, date_range: tuple = None, 
              min_score: float = None) -> List[ALRIRecord]:
        """Query records by filters."""
        pass
```

## Data Models

### Input Data Schema

```python
# Enrollment CSV Schema
enrollment_schema = {
    'state': str,           # State name (lowercase)
    'district': str,        # District name (lowercase)
    'pincode': int,         # 6-digit PIN code
    'year': int,            # Year (e.g., 2025)
    'month': int,           # Month (1-12)
    'day': int,             # Day (1-31)
    'total_enrollment_age': int  # Total enrollments for the day
}

# Demographic CSV Schema
demographic_schema = {
    'state': str,
    'district': str,
    'pincode': int,
    'year': int,
    'month': int,
    'day': int,
    'total_demographic_age': int  # Total demographic updates for the day
}

# Biometric CSV Schema
biometric_schema = {
    'state': str,
    'district': str,
    'pincode': int,
    'year': int,
    'month': int,
    'day': int,
    'total_biometric_age': int  # Total biometric updates for the day
}
```

### Aggregated Data Model

```python
@dataclass
class DistrictMonthlyAggregate:
    state: str
    district: str
    year: int
    month: int
    total_enrollments: int
    total_demographic_updates: int
    total_biometric_updates: int
    enrollment_rate: float  # Per 1000 baseline
    demographic_rate: float
    biometric_rate: float
    pincode_count: int  # Number of unique pincodes
```

### ALRI Output Model

```python
@dataclass
class ALRIOutput:
    district: str
    state: str
    year: int
    month: int
    alri_score: float  # 0-100
    coverage_risk: float  # 0-1
    instability_risk: float  # 0-1
    biometric_risk: float  # 0-1
    anomaly_factor: float  # 0-1
    reason_codes: List[ReasonCode]
    recommendations: List[Intervention]
    cluster_label: str
    forecast_trend: str
    computed_at: datetime
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the acceptance criteria analysis, the following correctness properties must be validated through property-based testing:

### Property 1: CSV Parsing Correctness

*For any* valid CSV row containing state, district, pincode, year, month, day, and count fields, parsing the row SHALL extract all fields with correct types and values matching the original data.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Aggregation Sum Invariant

*For any* set of daily records for a district-month combination, the monthly aggregate total SHALL equal the sum of all daily values for that district-month.

**Validates: Requirements 1.4**

### Property 3: Sub-score Range Invariant

*For any* computed sub-score (Coverage_Risk, Instability_Risk, Biometric_Risk, Anomaly_Factor), the value SHALL be in the range [0, 1].

**Validates: Requirements 2.3, 3.3, 4.3, 5.3**

### Property 4: ALRI Score Range Invariant

*For any* computed ALRI score, the value SHALL be in the range [0, 100].

**Validates: Requirements 6.2**

### Property 5: ALRI Weighted Sum Correctness

*For any* set of valid sub-scores (C, D, B, A in [0,1]) and weights (w1, w2, w3, w4 summing to 1.0), the ALRI score SHALL equal (w1×C + w2×D + w3×B + w4×A) × 100.

**Validates: Requirements 6.1**

### Property 6: Coverage Risk Monotonicity

*For any* two districts where district A has lower child enrollment proportion than district B, district A SHALL have a higher or equal Coverage_Risk score than district B.

**Validates: Requirements 2.4**

### Property 7: Instability Risk Monotonicity

*For any* two districts where district A has higher demographic update frequency than district B, district A SHALL have a higher or equal Data_Instability_Risk score than district B.

**Validates: Requirements 3.4**

### Property 8: Anomaly Detection Threshold

*For any* time-series residual value exceeding 3 standard deviations from the mean, the Anomaly_Detector SHALL flag it as an anomaly.

**Validates: Requirements 5.2**

### Property 9: STL Decomposition Reconstruction

*For any* time-series input, the sum of trend, seasonal, and residual components from STL decomposition SHALL reconstruct the original series (within floating-point tolerance).

**Validates: Requirements 5.1**

### Property 10: Reason Code Ranking

*For any* ALRI result with multiple contributing factors, the generated Reason_Codes SHALL be ordered by contribution magnitude (highest first).

**Validates: Requirements 7.3**

### Property 11: Reason Code Completeness

*For any* generated Reason_Code, it SHALL include a valid label from the defined set, a severity level (Low/Medium/High/Critical), and a contribution value.

**Validates: Requirements 7.2, 7.4**

### Property 12: Recommendation Mapping

*For any* valid Reason_Code, the Recommendation_Engine SHALL produce between 1 and 3 interventions from the defined intervention set.

**Validates: Requirements 8.1, 8.2**

### Property 13: Recommendation Priority Ordering

*For any* list of recommendations, they SHALL be ordered by priority (low-cost, high-impact first).

**Validates: Requirements 8.4**

### Property 14: Forecast Confidence Interval

*For any* forecast result, the lower_bound values SHALL be less than or equal to forecast_values, and forecast_values SHALL be less than or equal to upper_bound values.

**Validates: Requirements 9.2**

### Property 15: Cluster Assignment Completeness

*For any* set of districts passed to the Clustering_Module, every district SHALL be assigned to exactly one cluster with a valid label.

**Validates: Requirements 10.2, 10.3**

### Property 16: ALRI Record Serialization Round-Trip

*For any* valid ALRIRecord, serializing to JSON then deserializing SHALL produce an equivalent record with identical field values.

**Validates: Requirements 13.4**

### Property 17: Query Filter Correctness

*For any* query with district, date_range, or min_score filters, all returned records SHALL satisfy all specified filter conditions.

**Validates: Requirements 13.3**

## Error Handling

### Data Loading Errors

| Error Condition | Handling Strategy | User Feedback |
|----------------|-------------------|---------------|
| File not found | Raise `FileNotFoundError` with filepath | "Data file not found: {filepath}" |
| Invalid CSV format | Raise `ValueError` with row details | "Invalid CSV format at row {n}: {details}" |
| Missing required columns | Raise `SchemaValidationError` | "Missing required columns: {columns}" |
| Invalid data types | Log warning, attempt coercion | "Type coercion applied to column {col}" |

### Computation Errors

| Error Condition | Handling Strategy | User Feedback |
|----------------|-------------------|---------------|
| Division by zero (rate calc) | Return 0.0, log warning | "Zero baseline for district {d}, rate set to 0" |
| Insufficient data for STL | Skip anomaly detection | "Insufficient data points for anomaly detection" |
| Negative values in counts | Clip to 0, log warning | "Negative count corrected to 0 for {district}" |
| NaN in sub-scores | Impute with median, log | "NaN imputed for {metric} in {district}" |

### Forecasting Errors

| Error Condition | Handling Strategy | User Feedback |
|----------------|-------------------|---------------|
| < 12 months data | Use simple trend extrapolation | "Limited data, using linear extrapolation" |
| Model convergence failure | Fall back to naive forecast | "Forecast model failed, using naive method" |
| Extreme outliers in history | Apply winsorization before fit | "Outliers winsorized for stable forecast" |

### Storage Errors

| Error Condition | Handling Strategy | User Feedback |
|----------------|-------------------|---------------|
| JSON serialization failure | Raise `SerializationError` | "Failed to serialize record: {details}" |
| Disk write failure | Retry 3x, then raise | "Storage write failed after retries" |
| Corrupted JSON on load | Skip record, log error | "Skipped corrupted record at line {n}" |

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property-based tests**: Verify universal properties across randomly generated inputs

### Property-Based Testing Framework

- **Library**: Hypothesis (Python)
- **Minimum iterations**: 100 per property test
- **Tag format**: `# Feature: aadhaar-sentinel, Property {N}: {property_text}`

### Test Organization

```
tests/
├── unit/
│   ├── test_data_loader.py      # CSV parsing edge cases
│   ├── test_aggregator.py       # Aggregation examples
│   ├── test_alri_calculator.py  # Score computation examples
│   ├── test_anomaly_detector.py # Anomaly detection examples
│   ├── test_reason_codes.py     # Reason code generation
│   ├── test_recommendations.py  # Intervention mapping
│   ├── test_forecaster.py       # Forecast generation
│   ├── test_clustering.py       # District segmentation
│   └── test_serializer.py       # JSON serialization
├── property/
│   ├── test_parsing_properties.py    # Property 1
│   ├── test_aggregation_properties.py # Property 2
│   ├── test_scoring_properties.py    # Properties 3-7
│   ├── test_anomaly_properties.py    # Properties 8-9
│   ├── test_explainability_properties.py # Properties 10-13
│   ├── test_forecast_properties.py   # Property 14
│   ├── test_clustering_properties.py # Property 15
│   └── test_storage_properties.py    # Properties 16-17
└── conftest.py                  # Shared fixtures and generators
```

### Custom Generators (Hypothesis)

```python
# conftest.py - Example generators for property tests

from hypothesis import strategies as st

# Generate valid district data
district_data = st.fixed_dictionaries({
    'state': st.sampled_from(['karnataka', 'maharashtra', 'uttar pradesh']),
    'district': st.text(min_size=3, max_size=30, alphabet=st.characters(whitelist_categories=('L',))),
    'pincode': st.integers(min_value=100000, max_value=999999),
    'year': st.integers(min_value=2020, max_value=2026),
    'month': st.integers(min_value=1, max_value=12),
    'day': st.integers(min_value=1, max_value=28),
    'count': st.integers(min_value=0, max_value=10000)
})

# Generate valid sub-scores
sub_scores = st.fixed_dictionaries({
    'coverage': st.floats(min_value=0.0, max_value=1.0),
    'instability': st.floats(min_value=0.0, max_value=1.0),
    'biometric': st.floats(min_value=0.0, max_value=1.0),
    'anomaly': st.floats(min_value=0.0, max_value=1.0)
})

# Generate valid ALRI records for serialization
alri_records = st.fixed_dictionaries({
    'district': st.text(min_size=3, max_size=30),
    'state': st.text(min_size=3, max_size=30),
    'alri_score': st.floats(min_value=0.0, max_value=100.0),
    'sub_scores': sub_scores,
    'reason_codes': st.lists(st.sampled_from([
        'Low_Child_Enrolment', 'High_Address_Churn', 
        'Low_Biometric_Update_5to15', 'Anomalous_Data_Entry'
    ]), min_size=1, max_size=4),
    'computed_at': st.datetimes().map(lambda d: d.isoformat())
})
```

### Test Coverage Requirements

| Component | Unit Test Coverage | Property Test Coverage |
|-----------|-------------------|----------------------|
| ETL Pipeline | Edge cases, error handling | Properties 1, 2 |
| ALRI Calculator | Example computations | Properties 3, 4, 5, 6, 7 |
| Anomaly Detector | Known anomaly patterns | Properties 8, 9 |
| Reason Codes | Label generation | Properties 10, 11 |
| Recommendations | Intervention mapping | Properties 12, 13 |
| Forecasting | Trend detection | Property 14 |
| Clustering | Segment assignment | Property 15 |
| Storage | JSON format | Properties 16, 17 |
