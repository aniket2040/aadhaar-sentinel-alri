# 🛡️ Aadhaar Sentinel

**Aadhaar Lifecycle Risk Index (ALRI) Platform** — A comprehensive decision-support system that transforms Aadhaar enrolment, demographic-update, and biometric-update datasets into district-level early-warning scores with actionable recommendations for UIDAI.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)]()
[![Property-Based Testing](https://img.shields.io/badge/PBT-Hypothesis-purple.svg)](https://hypothesis.readthedocs.io/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [ALRI Scoring Methodology](#alri-scoring-methodology)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Data Requirements](#data-requirements)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Dashboard Guide](#dashboard-guide)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Aadhaar Sentinel is a Python-based analytics platform designed to help UIDAI identify and address gaps in Aadhaar coverage, data quality issues, and compliance risks across India's districts. The system processes three core datasets—enrollment records, demographic updates, and biometric updates—to compute a composite risk score called the **Aadhaar Lifecycle Risk Index (ALRI)**.

### What Problems Does It Solve?

1. **Inclusion Gaps**: Identifies districts with low enrollment coverage, especially for vulnerable populations (children 0-5, 5-17 age bands)
2. **Data Quality Issues**: Detects unusual patterns in demographic and biometric updates that may indicate fraud or data entry errors
3. **Compliance Monitoring**: Tracks biometric update compliance for mandatory 5-year and 15-year updates
4. **Resource Allocation**: Provides actionable recommendations for deploying mobile camps, school drives, and awareness campaigns
5. **Predictive Planning**: Forecasts future enrollment and update volumes for proactive resource planning

### Who Is It For?

- **UIDAI Policy Makers**: Prioritize outreach for underserved populations
- **Operations Managers**: Understand population mobility and demographic churn patterns
- **Compliance Officers**: Monitor biometric update rates and identify non-compliant districts
- **Fraud Analysts**: Detect anomalous patterns in update data
- **Regional Directors**: Generate automated reports for stakeholder communication
- **Field Officers**: Receive actionable recommendations with reason codes

---

## Key Features

### 📊 Data Processing & ETL

- **Multi-format CSV Ingestion**: Load enrollment, demographic, and biometric datasets with automatic schema validation
- **Monthly Aggregation**: Aggregate daily records to district-month level for trend analysis
- **Missing Value Handling**: Configurable imputation strategies with audit logging
- **Baseline Computation**: Calculate district-level baselines for normalization

### 🎯 Risk Scoring Engine

- **Four-Dimensional Risk Model**: Coverage, Instability, Biometric Compliance, and Anomaly factors
- **Configurable Weights**: Adjust component weights based on policy priorities
- **Z-Score Normalization**: Standardized scoring across diverse districts
- **Monotonic Risk Properties**: Higher risk factors always yield higher scores

### 🔍 Anomaly Detection

- **STL Decomposition**: Separate trend, seasonal, and residual components from time-series
- **Statistical Thresholding**: Flag residuals exceeding 3 standard deviations
- **Anomaly Classification**: Categorize anomalies as spikes, drops, or normal variations
- **Configurable Sensitivity**: Adjust detection thresholds for different use cases

### 💡 Explainability Layer

- **Human-Readable Reason Codes**: `Low_Child_Enrolment`, `High_Address_Churn`, `Low_Biometric_Update_5to15`, `Anomalous_Data_Entry`
- **Severity Classification**: Low, Medium, High, Critical severity levels
- **Contribution Ranking**: Reason codes ranked by their contribution to the ALRI score
- **Affected Population Estimates**: Quantify the impact of each risk factor

### 📋 Recommendation Engine

- **Intervention Mapping**: Each reason code maps to 1-3 recommended actions
- **Cost-Impact Prioritization**: Low-cost, high-impact interventions prioritized first
- **Action Types**: Mobile van deployment, school enrollment drives, SMS/IVR campaigns, biometric camps, data quality audits

### 📈 Time-Series Forecasting

- **Prophet Integration**: Industry-standard forecasting with confidence intervals
- **Multi-Metric Forecasts**: Separate predictions for enrollments, demographic updates, biometric updates
- **Trend Detection**: Automatic flagging of declining trends for proactive intervention
- **Configurable Horizon**: 3-6 month forecast windows

### 🗂️ District Clustering

- **Behavioral Segmentation**: Group districts by similar patterns using KMeans
- **Pre-defined Labels**: Stable-HighCoverage, Migratory-HighChurn, ChildGap-HighRisk, LowActivity-Rural
- **Cluster Profiles**: Characteristic metrics for each segment
- **Policy Targeting**: Design targeted interventions for each cluster type

### 🖥️ Interactive Dashboard

- **District Heatmap**: Visual representation of ALRI scores across all districts
- **Time-Series Charts**: Enrollment, demographic, and biometric trends over time
- **Alerts Panel**: Real-time alerts for districts crossing risk thresholds
- **Multi-Level Filtering**: Filter by state, district, time period, and score threshold
- **Sub-Score Breakdown**: Radar charts showing component contributions

### 📄 Automated Reporting

- **PDF Generation**: Professional reports with charts and tables
- **State-Level Reports**: Top-10 at-risk districts per state
- **District-Level Reports**: Detailed metrics, reason codes, and recommendations
- **Trend Visualizations**: Embedded time-series charts in reports

### 💾 Data Persistence

- **JSON Serialization**: Structured data export for interoperability
- **Round-Trip Integrity**: Guaranteed data preservation through serialization cycles
- **Query Interface**: Filter records by district, date range, and score thresholds
- **Audit Trail**: Timestamped records for compliance and reproducibility

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Aadhaar Sentinel Platform                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                       │
│  │ enrollment   │    │ demographic  │    │ biometric    │   DATA SOURCES        │
│  │    .csv      │    │    .csv      │    │    .csv      │                       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                   │                               │
│         └───────────────────┼───────────────────┘                               │
│                             ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         ETL PIPELINE                                      │   │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐              │   │
│  │  │ DataLoader  │  │MonthlyAggregator│  │MissingValueHandler│             │   │
│  │  │ • Validate  │─▶│ • Group by      │─▶│ • Impute missing │             │   │
│  │  │ • Parse     │  │   district/month│  │ • Log affected   │             │   │
│  │  │ • Coerce    │  │ • Sum totals    │  │ • Version data   │             │   │
│  │  └─────────────┘  └─────────────────┘  └──────────────────┘              │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        ALRI CALCULATOR                                    │   │
│  │                                                                           │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────┐ │   │
│  │  │ Coverage Risk  │ │ Instability    │ │ Biometric Risk │ │ Anomaly    │ │   │
│  │  │     (C)        │ │ Risk (D)       │ │     (B)        │ │ Factor (A) │ │   │
│  │  │                │ │                │ │                │ │            │ │   │
│  │  │ • Enrollment   │ │ • Update rates │ │ • Update rates │ │ • STL      │ │   │
│  │  │   rates        │ │ • Volatility   │ │ • Compliance   │ │   decomp   │ │   │
│  │  │ • Child gaps   │ │ • Churn        │ │ • Trends       │ │ • Residual │ │   │
│  │  │                │ │                │ │                │ │   analysis │ │   │
│  │  │  Weight: 0.30  │ │  Weight: 0.30  │ │  Weight: 0.30  │ │Weight: 0.10│ │   │
│  │  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └─────┬──────┘ │   │
│  │          │                  │                  │                │        │   │
│  │          └──────────────────┴──────────────────┴────────────────┘        │   │
│  │                                    │                                      │   │
│  │                                    ▼                                      │   │
│  │              ┌─────────────────────────────────────────┐                  │   │
│  │              │  ALRI = (0.30×C + 0.30×D + 0.30×B +    │                  │   │
│  │              │          0.10×A) × 100                  │                  │   │
│  │              │                                         │                  │   │
│  │              │  Output: Score 0-100 per district       │                  │   │
│  │              └─────────────────────────────────────────┘                  │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                               │
│         ┌───────────────────────┼───────────────────────┐                       │
│         ▼                       ▼                       ▼                       │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐               │
│  │ EXPLAINABILITY │     │  FORECASTING   │     │   CLUSTERING   │               │
│  │                │     │                │     │                │               │
│  │ ReasonCode     │     │ TimeSeriesFor- │     │ DistrictClus-  │               │
│  │ Generator      │     │ ecaster        │     │ terer          │               │
│  │ • Top factors  │     │ • Prophet      │     │ • KMeans       │               │
│  │ • Severity     │     │ • 3-6 months   │     │ • 4 segments   │               │
│  │ • Ranking      │     │ • Confidence   │     │ • Profiles     │               │
│  └───────┬────────┘     └───────┬────────┘     └───────┬────────┘               │
│          │                      │                      │                        │
│          └──────────────────────┼──────────────────────┘                        │
│                                 ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     RECOMMENDATION ENGINE                                 │   │
│  │                                                                           │   │
│  │  Reason Code → Interventions Mapping:                                     │   │
│  │  ┌─────────────────────────┬─────────────────────────────────────────┐   │   │
│  │  │ Low_Child_Enrolment     │ School Drive, Mobile Van, SMS Campaign  │   │   │
│  │  │ High_Address_Churn      │ SMS/IVR Campaign, Update Kiosks         │   │   │
│  │  │ Low_Biometric_Update    │ Free Biometric Camp, School Drive       │   │   │
│  │  │ Anomalous_Data_Entry    │ Data Quality Audit                      │   │   │
│  │  └─────────────────────────┴─────────────────────────────────────────┘   │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                               │
│         ┌───────────────────────┴───────────────────────┐                       │
│         ▼                                               ▼                       │
│  ┌──────────────────────────┐                ┌──────────────────────────┐       │
│  │   STREAMLIT DASHBOARD    │                │   PDF REPORT GENERATOR   │       │
│  │                          │                │                          │       │
│  │  • District heatmap      │                │  • State reports         │       │
│  │  • Time-series charts    │                │  • District reports      │       │
│  │  • Alerts panel          │                │  • Top-10 at-risk        │       │
│  │  • Filters (state/time)  │                │  • Recommendations       │       │
│  │  • Sub-score breakdown   │                │  • Trend charts          │       │
│  └──────────────────────────┘                └──────────────────────────┘       │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                          STORAGE MODULE                                   │   │
│  │  ALRISerializer → JSON files → ALRIStorage → Query Interface             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ALRI Scoring Methodology

### The ALRI Formula

The Aadhaar Lifecycle Risk Index is computed as a weighted sum of four normalized sub-scores:

```
ALRI = (w₁ × Coverage_Risk + w₂ × Instability_Risk + w₃ × Biometric_Risk + w₄ × Anomaly_Factor) × 100
```

**Default Weights:**
| Component | Weight | Rationale |
|-----------|--------|-----------|
| Coverage Risk (C) | 0.30 | Enrollment gaps directly impact inclusion |
| Instability Risk (D) | 0.30 | High churn indicates data quality issues |
| Biometric Risk (B) | 0.30 | Compliance is critical for identity validity |
| Anomaly Factor (A) | 0.10 | Anomalies may indicate fraud or errors |

### Sub-Score Computation

#### 1. Coverage Risk (0-1)

Measures enrollment coverage gaps relative to district baseline:

```python
coverage_risk = z_score_normalize(
    baseline_enrollment - current_enrollment
) / baseline_enrollment

# Clipped to [0, 1] range
# Higher values = lower coverage = higher risk
```

**Key Indicators:**
- Enrollment rates relative to district baseline
- Month-on-month enrollment decline trends
- Child enrollment proportions (0-5, 5-17 age bands)

#### 2. Data Instability Risk (0-1)

Measures demographic update frequency and volatility:

```python
update_rate = demographic_updates / (enrollments / 1000)
volatility = rolling_std(update_volumes, window=3)

instability_risk = normalize(update_rate + volatility)
# Scaled to [0, 1] range
```

**Key Indicators:**
- Demographic update rates per 1000 enrollments
- Rolling volatility (standard deviation) of update volumes
- Address and mobile update frequency

#### 3. Biometric Compliance Risk (0-1)

Measures missed or delayed biometric updates:

```python
expected_updates = estimate_5yr_15yr_cohort_size(district)
actual_updates = biometric_update_count

compliance_gap = (expected_updates - actual_updates) / expected_updates
biometric_risk = normalize(compliance_gap)
# Higher missing updates = higher risk
```

**Key Indicators:**
- Biometric update rates relative to expected volumes
- Declining biometric update trends
- Threshold compliance (e.g., < 80% update rate)

#### 4. Anomaly Factor (0-1)

Detects unusual patterns using STL decomposition:

```python
trend, seasonal, residual = stl_decompose(time_series)
z_scores = (residual - mean(residual)) / std(residual)

anomalies = abs(z_scores) > 3.0  # 3-sigma threshold
anomaly_factor = count(anomalies) / len(time_series)
```

**Key Indicators:**
- Residuals exceeding 3 standard deviations
- Sudden spikes or drops in update volumes
- Seasonal pattern deviations

### Score Interpretation

| ALRI Score | Risk Level | Action Required |
|------------|------------|-----------------|
| 0-25 | 🟢 Low | Routine monitoring |
| 25-50 | 🟡 Medium | Enhanced monitoring, consider interventions |
| 50-75 | 🟠 High | Priority intervention required |
| 75-100 | 🔴 Critical | Immediate action, escalate to leadership |

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- 4GB+ RAM recommended for large datasets
- Modern web browser for dashboard

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/aadhaar-sentinel.git
cd aadhaar-sentinel

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "from src.scoring.alri_calculator import ALRICalculator; print('Installation successful!')"
```

### Dependencies

The `requirements.txt` includes:

```
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Scientific computing
statsmodels>=0.14.0    # STL decomposition
prophet>=1.1.0         # Time-series forecasting
scikit-learn>=1.3.0    # Machine learning (clustering)
streamlit>=1.28.0      # Interactive dashboard
plotly>=5.18.0         # Interactive visualizations
fpdf2>=2.7.0           # PDF report generation
hypothesis>=6.90.0     # Property-based testing
pytest>=7.4.0          # Test framework
```

### Troubleshooting Installation

**Prophet Installation Issues:**
```bash
# If Prophet fails to install, try:
pip install pystan==2.19.1.1
pip install prophet
```

**Windows-Specific:**
```bash
# Install Visual C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

---

## Quick Start Guide

### Step 1: Prepare Your Data

Create three CSV files with the following schemas:

**enrollment.csv:**
```csv
state,district,pincode,year,month,day,total_enrollment_age
karnataka,bengaluru urban,560001,2025,1,15,1250
karnataka,bengaluru urban,560001,2025,1,16,1180
maharashtra,mumbai,400001,2025,1,15,2340
...
```

**demographic.csv:**
```csv
state,district,pincode,year,month,day,total_demographic_age
karnataka,bengaluru urban,560001,2025,1,15,450
karnataka,bengaluru urban,560001,2025,1,16,420
maharashtra,mumbai,400001,2025,1,15,890
...
```

**biometric.csv:**
```csv
state,district,pincode,year,month,day,total_biometric_age
karnataka,bengaluru urban,560001,2025,1,15,320
karnataka,bengaluru urban,560001,2025,1,16,290
maharashtra,mumbai,400001,2025,1,15,560
...
```

### Step 2: Run the Pipeline

```bash
python main.py
```

**Expected Output:**
```
============================================================
Starting Aadhaar Sentinel ALRI Pipeline
============================================================
Loading data files...
Loaded 499991 enrollment records
Loaded 500000 demographic records
Loaded 500000 biometric records
Running ETL pipeline...
Aggregated enrollment: 2910 monthly records
Aggregated demographic: 5365 monthly records
Aggregated biometric: 5174 monthly records
Computing ALRI scores...
Processing 976 districts
Computed ALRI scores for 51 districts
Saving results to data/alri_storage/alri_results.json...
============================================================
Pipeline completed successfully!
Total time: 0:00:15.714424
Districts processed: 51
============================================================
```

### Step 3: Launch the Dashboard

```bash
streamlit run run_dashboard.py
```

Open your browser to `http://localhost:8501`

### Step 4: Explore Results

The dashboard provides:
- **Heatmap Tab**: Visual overview of all district ALRI scores
- **District Detail Tab**: Deep-dive into individual district metrics
- **Alerts Tab**: Districts exceeding risk thresholds

---

## Data Requirements

### Input Data Schemas

#### Enrollment Data

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| state | string | State name (lowercase) | karnataka |
| district | string | District name (lowercase) | bengaluru urban |
| pincode | integer | 6-digit PIN code | 560001 |
| year | integer | Year (YYYY) | 2025 |
| month | integer | Month (1-12) | 6 |
| day | integer | Day (1-31) | 15 |
| total_enrollment_age | integer | Total enrollments for the day | 1250 |

#### Demographic Data

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| state | string | State name (lowercase) | karnataka |
| district | string | District name (lowercase) | bengaluru urban |
| pincode | integer | 6-digit PIN code | 560001 |
| year | integer | Year (YYYY) | 2025 |
| month | integer | Month (1-12) | 6 |
| day | integer | Day (1-31) | 15 |
| total_demographic_age | integer | Total demographic updates | 450 |

#### Biometric Data

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| state | string | State name (lowercase) | karnataka |
| district | string | District name (lowercase) | bengaluru urban |
| pincode | integer | 6-digit PIN code | 560001 |
| year | integer | Year (YYYY) | 2025 |
| month | integer | Month (1-12) | 6 |
| day | integer | Day (1-31) | 15 |
| total_biometric_age | integer | Total biometric updates | 320 |

### Data Quality Requirements

- **Minimum Records**: At least 12 months of data recommended for reliable scoring
- **Completeness**: All required columns must be present
- **Consistency**: State and district names should be consistent across files
- **Date Range**: Data should cover the same time period across all three files

### Sample Data Generation

For testing, you can generate sample data:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(num_records=10000):
    states = ['karnataka', 'maharashtra', 'uttar pradesh']
    districts = {
        'karnataka': ['bengaluru urban', 'mysuru', 'mangaluru'],
        'maharashtra': ['mumbai', 'pune', 'nagpur'],
        'uttar pradesh': ['lucknow', 'kanpur nagar', 'ghaziabad']
    }
    
    records = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_records):
        state = np.random.choice(states)
        district = np.random.choice(districts[state])
        date = base_date + timedelta(days=np.random.randint(0, 365))
        
        records.append({
            'state': state,
            'district': district,
            'pincode': np.random.randint(100000, 999999),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'total_enrollment_age': np.random.randint(100, 2000)
        })
    
    return pd.DataFrame(records)

# Generate and save
df = generate_sample_data()
df.to_csv('enrollment.csv', index=False)
```

---

## Project Structure

```
aadhaar-sentinel/
│
├── 📁 src/                          # Source code
│   ├── 📁 etl/                      # Extract-Transform-Load pipeline
│   │   ├── __init__.py
│   │   ├── data_loader.py           # CSV loading and validation
│   │   ├── aggregator.py            # Monthly aggregation
│   │   └── imputer.py               # Missing value handling
│   │
│   ├── 📁 scoring/                  # ALRI calculation
│   │   ├── __init__.py
│   │   └── alri_calculator.py       # Core scoring engine
│   │
│   ├── 📁 anomaly/                  # Anomaly detection
│   │   ├── __init__.py
│   │   └── detector.py              # STL-based detection
│   │
│   ├── 📁 explainability/           # Reason code generation
│   │   ├── __init__.py
│   │   └── reason_codes.py          # Explainable AI layer
│   │
│   ├── 📁 recommendations/          # Intervention mapping
│   │   ├── __init__.py
│   │   └── engine.py                # Recommendation engine
│   │
│   ├── 📁 forecasting/              # Time-series forecasting
│   │   ├── __init__.py
│   │   └── forecaster.py            # Prophet integration
│   │
│   ├── 📁 clustering/               # District segmentation
│   │   ├── __init__.py
│   │   └── segmentation.py          # KMeans clustering
│   │
│   ├── 📁 dashboard/                # Interactive UI
│   │   ├── __init__.py
│   │   └── app.py                   # Streamlit dashboard
│   │
│   ├── 📁 reports/                  # Report generation
│   │   ├── __init__.py
│   │   └── generator.py             # PDF report generator
│   │
│   ├── 📁 storage/                  # Data persistence
│   │   ├── __init__.py
│   │   └── serializer.py            # JSON serialization
│   │
│   └── __init__.py
│
├── 📁 tests/                        # Test suite
│   ├── 📁 property/                 # Property-based tests
│   │   ├── test_aggregation_properties.py
│   │   ├── test_anomaly_properties.py
│   │   ├── test_clustering_properties.py
│   │   ├── test_explainability_properties.py
│   │   ├── test_forecast_properties.py
│   │   ├── test_parsing_properties.py
│   │   ├── test_scoring_properties.py
│   │   └── test_storage_properties.py
│   │
│   ├── 📁 unit/                     # Unit tests
│   │   └── __init__.py
│   │
│   ├── conftest.py                  # Shared fixtures and generators
│   └── __init__.py
│
├── 📁 data/                         # Data storage
│   └── 📁 alri_storage/             # Computed results
│       └── alri_results.json
│
├── 📁 .kiro/                        # Kiro spec files
│   └── 📁 specs/
│       └── 📁 aadhaar-sentinel/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
│
├── main.py                          # Main orchestration script
├── run_dashboard.py                 # Dashboard launcher
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## Module Documentation

### ETL Module (`src/etl/`)

#### DataLoader

Loads and validates CSV data files with schema enforcement.

```python
from src.etl.data_loader import DataLoader

loader = DataLoader()

# Load enrollment data
enrollment_df = loader.load_enrollment('enrollment.csv')

# Load demographic data
demographic_df = loader.load_demographic('demographic.csv')

# Load biometric data
biometric_df = loader.load_biometric('biometric.csv')
```

**Methods:**
- `load_enrollment(filepath)` → `pd.DataFrame`
- `load_demographic(filepath)` → `pd.DataFrame`
- `load_biometric(filepath)` → `pd.DataFrame`

#### MonthlyAggregator

Aggregates daily records to monthly district-level summaries.

```python
from src.etl.aggregator import MonthlyAggregator

aggregator = MonthlyAggregator()

# Aggregate by district and month
monthly_df = aggregator.aggregate_by_district_month(daily_df, 'total_enrollment_age')

# Compute district baselines
baselines = aggregator.compute_baselines(monthly_df, 'total_enrollment_age')
```

**Methods:**
- `aggregate_by_district_month(df, value_column)` → `pd.DataFrame`
- `compute_baselines(df, value_column)` → `pd.DataFrame`

#### MissingValueHandler

Handles missing values with configurable imputation strategies.

```python
from src.etl.imputer import MissingValueHandler

handler = MissingValueHandler()

# Impute missing values using median strategy
imputed_df, log = handler.impute(df, columns=['total_enrollment_age'], strategy='median')
```

**Methods:**
- `impute(df, columns, strategy='median')` → `Tuple[pd.DataFrame, List[str]]`

### Scoring Module (`src/scoring/`)

#### ALRICalculator

Core scoring engine for computing ALRI scores.

```python
from src.scoring.alri_calculator import ALRICalculator, ALRIWeights

# Initialize with default weights
calculator = ALRICalculator()

# Or with custom weights
custom_weights = ALRIWeights(
    coverage=0.35,
    instability=0.25,
    biometric=0.30,
    anomaly=0.10
)
calculator = ALRICalculator(weights=custom_weights)

# Compute individual sub-scores
coverage_risk = calculator.compute_coverage_risk(district_data, baseline)
instability_risk = calculator.compute_instability_risk(district_data, baseline)
biometric_risk = calculator.compute_biometric_risk(district_data, baseline)

# Compute full ALRI result
result = calculator.compute_alri(
    district='bengaluru urban',
    state='karnataka',
    enrollment_data=enrollment_df,
    demographic_data=demographic_df,
    biometric_data=biometric_df,
    baselines=baselines
)

print(f"ALRI Score: {result.alri_score:.1f}")
print(f"Coverage Risk: {result.coverage_risk:.2f}")
print(f"Instability Risk: {result.instability_risk:.2f}")
print(f"Biometric Risk: {result.biometric_risk:.2f}")
print(f"Anomaly Factor: {result.anomaly_factor:.2f}")
```

**Classes:**
- `ALRIWeights` - Dataclass for configurable weights
- `ALRIResult` - Dataclass containing all score components
- `ALRICalculator` - Main calculator class

### Anomaly Module (`src/anomaly/`)

#### STLAnomalyDetector

Detects anomalies using Seasonal-Trend decomposition.

```python
from src.anomaly.detector import STLAnomalyDetector

detector = STLAnomalyDetector(zscore_threshold=3.0)

# Decompose time series
components = detector.decompose(time_series)
# Returns: {'trend': Series, 'seasonal': Series, 'residual': Series}

# Detect anomalies
anomalies = detector.detect_anomalies(time_series)
# Returns: List[AnomalyResult]

# Compute anomaly factor (0-1)
factor = detector.compute_anomaly_factor(anomalies)
```

**Classes:**
- `AnomalyResult` - Dataclass with anomaly details
- `STLAnomalyDetector` - Main detector class

### Explainability Module (`src/explainability/`)

#### ReasonCodeGenerator

Generates human-readable reason codes from ALRI results.

```python
from src.explainability.reason_codes import ReasonCodeGenerator, Severity

generator = ReasonCodeGenerator()

# Generate reason codes from ALRI result
reason_codes = generator.generate(alri_result)

for code in reason_codes:
    print(f"{code.code}: {code.description}")
    print(f"  Severity: {code.severity.value}")
    print(f"  Contribution: {code.contribution:.2%}")

# Determine severity for a score
severity = generator.determine_severity(0.75)  # Returns Severity.HIGH
```

**Classes:**
- `Severity` - Enum (LOW, MEDIUM, HIGH, CRITICAL)
- `ReasonCode` - Dataclass with code details
- `ReasonCodeGenerator` - Main generator class

### Recommendations Module (`src/recommendations/`)

#### RecommendationEngine

Maps reason codes to actionable interventions.

```python
from src.recommendations.engine import RecommendationEngine

engine = RecommendationEngine()

# Get recommendations for reason codes
interventions = engine.recommend(reason_codes)

for intervention in interventions:
    print(f"{intervention.action}")
    print(f"  Description: {intervention.description}")
    print(f"  Cost: {intervention.estimated_cost}")
    print(f"  Impact: {intervention.estimated_impact} people")
    print(f"  Priority: {intervention.priority}")
```

**Classes:**
- `Intervention` - Dataclass with intervention details
- `RecommendationEngine` - Main engine class

### Forecasting Module (`src/forecasting/`)

#### TimeSeriesForecaster

Forecasts future volumes using Prophet.

```python
from src.forecasting.forecaster import TimeSeriesForecaster

forecaster = TimeSeriesForecaster(model_type='prophet', horizon_months=6)

# Fit model to historical data
forecaster.fit(historical_df)

# Generate forecast
forecast = forecaster.predict()

print(f"Forecast values: {forecast.forecast_values}")
print(f"Lower bound: {forecast.lower_bound}")
print(f"Upper bound: {forecast.upper_bound}")
print(f"Trend: {forecast.trend}")

# Check for declining trend
is_declining = forecaster.detect_declining_trend(forecast)
```

**Classes:**
- `ForecastResult` - Dataclass with forecast details
- `TimeSeriesForecaster` - Main forecaster class

### Clustering Module (`src/clustering/`)

#### DistrictClusterer

Segments districts by behavioral patterns.

```python
from src.clustering.segmentation import DistrictClusterer

clusterer = DistrictClusterer(n_clusters=4, method='kmeans')

# Cluster districts based on features
profiles = clusterer.fit_predict(district_features_df)

for profile in profiles:
    print(f"Cluster {profile.cluster_id}: {profile.label}")
    print(f"  Districts: {len(profile.districts)}")
    print(f"  Characteristics: {profile.characteristics}")
```

**Classes:**
- `ClusterProfile` - Dataclass with cluster details
- `DistrictClusterer` - Main clusterer class

### Storage Module (`src/storage/`)

#### ALRISerializer and ALRIStorage

Persists and queries ALRI records.

```python
from src.storage.serializer import ALRISerializer, ALRIStorage, ALRIRecord

# Create a record
record = ALRIRecord(
    district='bengaluru urban',
    state='karnataka',
    alri_score=45.5,
    sub_scores={'coverage': 0.4, 'instability': 0.5, 'biometric': 0.45, 'anomaly': 0.1},
    reason_codes=['Low_Child_Enrolment', 'High_Address_Churn'],
    computed_at='2025-01-15T10:30:00'
)

# Serialize to JSON
serializer = ALRISerializer()
json_str = serializer.serialize(record)

# Deserialize from JSON
restored_record = serializer.deserialize(json_str)

# Storage operations
storage = ALRIStorage()
storage.save([record], 'data/alri_results.json')

# Load records
records = storage.load('data/alri_results.json')

# Query with filters
high_risk = storage.query(min_score=75.0)
karnataka_records = storage.query(district='bengaluru urban')
recent_records = storage.query(date_range=('2025-01-01', '2025-01-31'))
```

**Classes:**
- `ALRIRecord` - Dataclass for ALRI data
- `ALRISerializer` - JSON serialization
- `ALRIStorage` - File-based storage with query support

---

## Dashboard Guide

### Launching the Dashboard

```bash
streamlit run run_dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Dashboard Tabs

#### 📊 Heatmap Tab

The heatmap provides a visual overview of ALRI scores across all districts:

- **Treemap View**: Districts grouped by state, colored by ALRI score
- **Bar Chart**: Top 20 districts by ALRI score for easy comparison
- **Color Scale**: Green (low risk) → Yellow → Orange → Red → Purple (extreme risk)

**Interactions:**
- Hover over districts to see detailed metrics
- Click on states to drill down to district level
- Use sidebar filters to narrow the view

#### 📋 District Detail Tab

Deep-dive into individual district metrics:

- **Metric Cards**: ALRI score and all four sub-scores
- **Radar Chart**: Visual breakdown of sub-score contributions
- **Time-Series Charts**: Historical trends for enrollments, demographic updates, biometric updates
- **Reason Codes**: List of risk factors for the selected district

**Interactions:**
- Select a district from the dropdown
- View historical trends over time
- Identify which sub-scores are driving the risk

#### 🚨 Alerts Tab

Real-time alerts for districts exceeding risk thresholds:

- **Critical Alerts** (🔴): ALRI ≥ 75
- **High Risk Alerts** (🟠): ALRI ≥ 50
- **Medium Risk Alerts** (🟡): ALRI ≥ 25

**Alert Details:**
- Sub-score breakdown
- Reason codes
- Computation timestamp

**Summary Statistics:**
- Total alerts count
- Breakdown by severity level

### Sidebar Filters

- **State Filter**: Select a specific state or view all
- **District Filter**: Select a specific district (filtered by state)
- **Time Period**: Set start and end dates for analysis
- **Risk Threshold**: Filter districts by minimum ALRI score

---

## API Reference

### Core Classes Summary

| Class | Module | Purpose |
|-------|--------|---------|
| `DataLoader` | `src.etl.data_loader` | Load and validate CSV files |
| `MonthlyAggregator` | `src.etl.aggregator` | Aggregate daily data to monthly |
| `MissingValueHandler` | `src.etl.imputer` | Handle missing values |
| `ALRICalculator` | `src.scoring.alri_calculator` | Compute ALRI scores |
| `ALRIWeights` | `src.scoring.alri_calculator` | Configure scoring weights |
| `ALRIResult` | `src.scoring.alri_calculator` | Store scoring results |
| `STLAnomalyDetector` | `src.anomaly.detector` | Detect time-series anomalies |
| `AnomalyResult` | `src.anomaly.detector` | Store anomaly details |
| `ReasonCodeGenerator` | `src.explainability.reason_codes` | Generate reason codes |
| `ReasonCode` | `src.explainability.reason_codes` | Store reason code details |
| `Severity` | `src.explainability.reason_codes` | Severity level enum |
| `RecommendationEngine` | `src.recommendations.engine` | Map codes to interventions |
| `Intervention` | `src.recommendations.engine` | Store intervention details |
| `TimeSeriesForecaster` | `src.forecasting.forecaster` | Forecast future volumes |
| `ForecastResult` | `src.forecasting.forecaster` | Store forecast results |
| `DistrictClusterer` | `src.clustering.segmentation` | Cluster districts |
| `ClusterProfile` | `src.clustering.segmentation` | Store cluster details |
| `AadhaarSentinelDashboard` | `src.dashboard.app` | Interactive dashboard |
| `PDFReportGenerator` | `src.reports.generator` | Generate PDF reports |
| `ALRISerializer` | `src.storage.serializer` | JSON serialization |
| `ALRIStorage` | `src.storage.serializer` | Persist and query records |
| `ALRIRecord` | `src.storage.serializer` | Store ALRI data |

### Reason Codes Reference

| Code | Description | Typical Interventions |
|------|-------------|----------------------|
| `Low_Child_Enrolment` | Low enrollment rates for children | School drives, mobile vans, SMS campaigns |
| `High_Address_Churn` | High frequency of address updates | SMS/IVR campaigns, update kiosks |
| `Low_Biometric_Update_5to15` | Low biometric update compliance | Free biometric camps, school drives |
| `Anomalous_Data_Entry` | Unusual patterns in data | Data quality audits |

### Intervention Types

| Intervention | Cost | Typical Impact |
|--------------|------|----------------|
| School Enrollment Drive | Low | 500-2000 children |
| Mobile Van Deployment | Medium | 1000-5000 people |
| SMS/IVR Campaign | Low | 10000+ people |
| Additional Update Kiosks | High | 5000-20000 people |
| Free Biometric Camp | Medium | 2000-10000 people |
| Data Quality Audit | Low | N/A (process improvement) |

---

## Testing

### Test Suite Overview

The project uses property-based testing with Hypothesis for comprehensive validation of correctness properties.

**Test Statistics:**
- Total Tests: 63
- Property-Based Tests: 63
- Test Duration: ~150 seconds

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with detailed output
python -m pytest tests/ -v --tb=long

# Run specific test file
python -m pytest tests/property/test_scoring_properties.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run tests matching a pattern
python -m pytest tests/ -k "scoring" -v
```

### Test Categories

#### Parsing Properties (`test_parsing_properties.py`)
- Property 1: CSV parsing preserves all field values and types

#### Aggregation Properties (`test_aggregation_properties.py`)
- Property 2: Monthly aggregates equal sum of daily values

#### Scoring Properties (`test_scoring_properties.py`)
- Property 3: Sub-scores always in [0, 1] range
- Property 4: ALRI scores always in [0, 100] range
- Property 5: Weighted sum formula correctness
- Property 6: Coverage risk monotonicity
- Property 7: Instability risk monotonicity

#### Anomaly Properties (`test_anomaly_properties.py`)
- Property 8: Anomalies flagged when z-score > threshold
- Property 9: STL decomposition reconstructs original series

#### Explainability Properties (`test_explainability_properties.py`)
- Property 10: Reason codes ranked by contribution
- Property 11: Reason codes have valid labels and severity
- Property 12: Each reason code maps to 1-3 interventions
- Property 13: Recommendations ordered by priority

#### Forecast Properties (`test_forecast_properties.py`)
- Property 14: Confidence intervals are valid (lower ≤ forecast ≤ upper)

#### Clustering Properties (`test_clustering_properties.py`)
- Property 15: Every district assigned to exactly one cluster

#### Storage Properties (`test_storage_properties.py`)
- Property 16: Serialization round-trip preserves data
- Property 17: Query filters return correct subsets

### Custom Hypothesis Generators

The `tests/conftest.py` file contains custom generators for property-based testing:

```python
# Example generators
district_data = st.fixed_dictionaries({
    'state': st.sampled_from(['karnataka', 'maharashtra', 'uttar pradesh']),
    'district': st.text(min_size=3, max_size=30),
    'pincode': st.integers(min_value=100000, max_value=999999),
    'year': st.integers(min_value=2020, max_value=2026),
    'month': st.integers(min_value=1, max_value=12),
    'day': st.integers(min_value=1, max_value=28),
    'count': st.integers(min_value=0, max_value=10000)
})

sub_scores = st.fixed_dictionaries({
    'coverage': st.floats(min_value=0.0, max_value=1.0),
    'instability': st.floats(min_value=0.0, max_value=1.0),
    'biometric': st.floats(min_value=0.0, max_value=1.0),
    'anomaly': st.floats(min_value=0.0, max_value=1.0)
})
```

---

## Configuration

### ALRI Weights Configuration

Modify weights in `src/scoring/alri_calculator.py`:

```python
@dataclass
class ALRIWeights:
    coverage: float = 0.30      # Enrollment coverage gaps
    instability: float = 0.30   # Demographic update frequency
    biometric: float = 0.30     # Biometric compliance
    anomaly: float = 0.10       # Anomaly detection
```

**Note:** Weights should sum to 1.0 for proper scaling.

### Alert Thresholds Configuration

Modify thresholds in `src/dashboard/app.py`:

```python
ALERT_THRESHOLDS = {
    'critical': 75.0,   # Red alerts
    'high': 50.0,       # Orange alerts
    'medium': 25.0      # Yellow alerts
}
```

### Anomaly Detection Configuration

Modify sensitivity in `src/anomaly/detector.py`:

```python
class STLAnomalyDetector:
    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold  # Standard deviations
```

### Forecasting Configuration

Modify forecast horizon in `src/forecasting/forecaster.py`:

```python
class TimeSeriesForecaster:
    def __init__(self, model_type: str = 'prophet', horizon_months: int = 6):
        self.model_type = model_type
        self.horizon_months = horizon_months
```

### Clustering Configuration

Modify cluster count in `src/clustering/segmentation.py`:

```python
class DistrictClusterer:
    CLUSTER_LABELS = {
        0: 'Stable-HighCoverage',
        1: 'Migratory-HighChurn',
        2: 'ChildGap-HighRisk',
        3: 'LowActivity-Rural',
    }
    
    def __init__(self, n_clusters: int = 4, method: str = 'kmeans'):
        self.n_clusters = n_clusters
```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'src'"

**Solution:** Run from the project root directory or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Prophet Installation Fails

**Solution:** Install dependencies first:
```bash
pip install pystan==2.19.1.1
pip install prophet
```

#### 3. "Insufficient data for STL decomposition"

**Cause:** Less than 2 complete seasonal cycles (typically 24 months)

**Solution:** Provide more historical data or the system will skip anomaly detection for that district.

#### 4. Dashboard Shows "No data available"

**Cause:** ALRI results not computed or file not found

**Solution:** Run `python main.py` first to compute and save results.

#### 5. Memory Error with Large Datasets

**Solution:** Process data in chunks:
```python
# In main.py, process districts in batches
batch_size = 100
for i in range(0, len(districts), batch_size):
    batch = districts[i:i+batch_size]
    # Process batch
```

#### 6. Slow Dashboard Performance

**Solution:** 
- Reduce the number of districts displayed
- Use date filters to limit data range
- Pre-aggregate data before loading

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check the [Issues](https://github.com/yourusername/aadhaar-sentinel/issues) page
2. Review the design documentation in `.kiro/specs/aadhaar-sentinel/`
3. Open a new issue with:
   - Python version
   - Operating system
   - Error message and stack trace
   - Steps to reproduce

---

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/aadhaar-sentinel.git
cd aadhaar-sentinel

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest-cov black flake8

# Run tests to verify setup
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public methods
- Keep functions focused and under 50 lines

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest tests/ -v`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Adding New Features

1. Update requirements in `.kiro/specs/aadhaar-sentinel/requirements.md`
2. Update design in `.kiro/specs/aadhaar-sentinel/design.md`
3. Add tasks to `.kiro/specs/aadhaar-sentinel/tasks.md`
4. Implement with property-based tests
5. Update this README

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Aadhaar Sentinel Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **UIDAI** for the Aadhaar ecosystem and the mission of universal identity
- **Meta's Prophet** for robust time-series forecasting
- **Hypothesis** for property-based testing framework
- **Streamlit** for rapid dashboard development
- **Plotly** for interactive visualizations
- The open-source community for the excellent libraries used in this project

---

<p align="center">
  Made with ❤️ for Digital India
</p>
