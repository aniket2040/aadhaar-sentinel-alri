# Requirements Document

## Introduction

Aadhaar Sentinel is an Aadhaar Lifecycle Risk Index (ALRI) Platform — a decision-support system that converts enrolment, demographic-update, and biometric-update datasets into district/PIN-level early-warning scores with actionable recommendations for UIDAI. The system detects inclusion gaps, predicts future identity failures (especially for children), and recommends prioritized, low-cost operational actions.

## Glossary

- **ALRI**: Aadhaar Lifecycle Risk Index — a composite score (0–100) indicating risk level per district/PIN
- **Coverage_Risk**: Sub-score measuring enrolment coverage gaps, especially for children (0–5, 5–17 age bands)
- **Data_Instability_Risk**: Sub-score measuring frequency of demographic updates (address, mobile, name/DOB changes)
- **Biometric_Compliance_Risk**: Sub-score measuring missed or delayed biometric updates for 5 and 15-year cohorts
- **Anomaly_Factor**: Sub-score detecting unusual spikes or drops in update patterns
- **Reason_Code**: Explainable tag indicating the primary risk driver (e.g., Low_Child_Enrolment, High_Address_Churn)
- **District**: Administrative unit for aggregation and risk scoring
- **PIN**: Postal Index Number for granular location-based analysis
- **ETL_Pipeline**: Extract-Transform-Load process for data ingestion and aggregation
- **STL_Decomposition**: Seasonal-Trend decomposition using Loess for time-series analysis
- **Intervention**: Recommended operational action (mobile camp, school drive, SMS campaign, etc.)

## Requirements

### Requirement 1: Data Ingestion and ETL Pipeline

**User Story:** As a UIDAI data analyst, I want to ingest and aggregate the three Aadhaar datasets, so that I can compute risk scores at district/PIN level.

#### Acceptance Criteria

1. WHEN enrollment data is loaded, THE ETL_Pipeline SHALL parse state, district, pincode, date fields, and total_enrollment_age counts
2. WHEN demographic data is loaded, THE ETL_Pipeline SHALL parse state, district, pincode, date fields, and total_demographic_age counts
3. WHEN biometric data is loaded, THE ETL_Pipeline SHALL parse state, district, pincode, date fields, and total_biometric_age counts
4. WHEN data is aggregated, THE ETL_Pipeline SHALL group records by district and month to produce monthly aggregates
5. WHEN missing values are encountered, THE ETL_Pipeline SHALL apply documented imputation rules and log affected records
6. THE ETL_Pipeline SHALL produce versioned datasets with timestamps for audit trails

### Requirement 2: Coverage Risk Score Computation

**User Story:** As a UIDAI policy maker, I want to identify districts with low enrolment coverage, so that I can prioritize outreach for underserved populations.

#### Acceptance Criteria

1. WHEN computing Coverage_Risk, THE ALRI_Calculator SHALL calculate enrolment rates relative to district baseline
2. WHEN computing Coverage_Risk, THE ALRI_Calculator SHALL detect month-on-month enrolment decline trends
3. WHEN normalizing Coverage_Risk, THE ALRI_Calculator SHALL apply z-score normalization and clip values to [0,1] range
4. THE ALRI_Calculator SHALL assign higher Coverage_Risk scores to districts with lower child enrolment proportions

### Requirement 3: Data Instability Risk Score Computation

**User Story:** As a UIDAI operations manager, I want to identify districts with high demographic churn, so that I can understand population mobility patterns.

#### Acceptance Criteria

1. WHEN computing Data_Instability_Risk, THE ALRI_Calculator SHALL calculate demographic update rates per 1000 enrollments
2. WHEN computing Data_Instability_Risk, THE ALRI_Calculator SHALL measure rolling volatility (standard deviation) of update volumes
3. WHEN normalizing Data_Instability_Risk, THE ALRI_Calculator SHALL scale values to [0,1] range
4. THE ALRI_Calculator SHALL assign higher Data_Instability_Risk scores to districts with frequent address and mobile updates

### Requirement 4: Biometric Compliance Risk Score Computation

**User Story:** As a UIDAI compliance officer, I want to identify districts with low biometric update rates, so that I can ensure identity records remain current.

#### Acceptance Criteria

1. WHEN computing Biometric_Compliance_Risk, THE ALRI_Calculator SHALL calculate biometric update rates relative to expected volumes
2. WHEN computing Biometric_Compliance_Risk, THE ALRI_Calculator SHALL identify districts with declining biometric update trends
3. WHEN normalizing Biometric_Compliance_Risk, THE ALRI_Calculator SHALL scale values to [0,1] range with higher missing updates yielding higher risk
4. THE ALRI_Calculator SHALL flag districts where biometric updates fall below threshold percentages

### Requirement 5: Anomaly Detection

**User Story:** As a UIDAI fraud analyst, I want to detect unusual patterns in update data, so that I can investigate potential data quality issues or fraud.

#### Acceptance Criteria

1. WHEN analyzing time-series data, THE Anomaly_Detector SHALL apply STL_Decomposition to separate trend, seasonal, and residual components
2. WHEN detecting anomalies, THE Anomaly_Detector SHALL flag residuals exceeding 3 standard deviations from mean
3. WHEN an anomaly is detected, THE Anomaly_Detector SHALL generate an Anomaly_Factor score between 0 (normal) and 1 (extreme)
4. THE Anomaly_Detector SHALL identify sudden spikes or drops in update volumes as potential anomalies

### Requirement 6: ALRI Score Aggregation

**User Story:** As a UIDAI decision maker, I want a single composite risk score per district, so that I can quickly prioritize interventions.

#### Acceptance Criteria

1. WHEN computing ALRI, THE ALRI_Calculator SHALL combine sub-scores using weighted formula: ALRI = 0.30×Coverage_Risk + 0.30×Data_Instability_Risk + 0.30×Biometric_Compliance_Risk + 0.10×Anomaly_Factor
2. WHEN computing ALRI, THE ALRI_Calculator SHALL produce scores in the range 0–100
3. WHEN weights are modified, THE ALRI_Calculator SHALL allow configurable weight parameters
4. THE ALRI_Calculator SHALL log all computation steps for audit and reproducibility

### Requirement 7: Reason Code Generation

**User Story:** As a UIDAI field officer, I want to understand why a district has high risk, so that I can take appropriate action.

#### Acceptance Criteria

1. WHEN ALRI exceeds threshold, THE Reason_Code_Generator SHALL identify the top contributing sub-scores
2. WHEN generating Reason_Codes, THE Reason_Code_Generator SHALL produce human-readable labels (e.g., Low_Child_Enrolment, High_Address_Churn, Low_Biometric_Update_5to15)
3. WHEN multiple risk factors exist, THE Reason_Code_Generator SHALL rank them by contribution magnitude
4. THE Reason_Code_Generator SHALL include severity level (Low, Medium, High, Critical) for each Reason_Code

### Requirement 8: Intervention Recommendation Engine

**User Story:** As a UIDAI program manager, I want actionable recommendations for each at-risk district, so that I can deploy resources effectively.

#### Acceptance Criteria

1. WHEN Reason_Codes are generated, THE Recommendation_Engine SHALL map each code to 1–3 recommended interventions
2. WHEN recommending interventions, THE Recommendation_Engine SHALL suggest actions such as: mobile van deployment, school enrollment drives, SMS/IVR campaigns, additional update kiosks, free biometric camps
3. WHEN presenting recommendations, THE Recommendation_Engine SHALL include estimated impact (people affected) where calculable
4. THE Recommendation_Engine SHALL prioritize low-cost, high-impact interventions

### Requirement 9: Time-Series Forecasting

**User Story:** As a UIDAI planner, I want to predict future enrolment and update volumes, so that I can plan resource allocation proactively.

#### Acceptance Criteria

1. WHEN forecasting, THE Forecasting_Module SHALL use interpretable models (Prophet or SARIMA) for 3–6 month predictions
2. WHEN generating forecasts, THE Forecasting_Module SHALL produce expected volumes with confidence intervals
3. WHEN forecast indicates declining trend, THE Forecasting_Module SHALL flag the district for proactive intervention
4. THE Forecasting_Module SHALL generate forecasts for enrolments, demographic updates, and biometric updates separately

### Requirement 10: District Clustering and Segmentation

**User Story:** As a UIDAI strategist, I want to segment districts by behavior patterns, so that I can design targeted policies for each segment.

#### Acceptance Criteria

1. WHEN clustering districts, THE Clustering_Module SHALL use KMeans or hierarchical clustering algorithms
2. WHEN segmenting districts, THE Clustering_Module SHALL identify patterns such as: Stable-HighCoverage, Migratory-HighChurn, ChildGap-HighRisk
3. WHEN presenting clusters, THE Clustering_Module SHALL provide descriptive labels and characteristic metrics for each segment
4. THE Clustering_Module SHALL allow configurable number of clusters

### Requirement 11: Interactive Dashboard

**User Story:** As a UIDAI control room operator, I want an interactive dashboard, so that I can monitor district risk levels in real-time.

#### Acceptance Criteria

1. WHEN displaying the dashboard, THE Dashboard SHALL show a district heatmap colored by ALRI score
2. WHEN a district is selected, THE Dashboard SHALL display time-series charts for enrolment, demographic updates, biometric updates, and ALRI trend
3. WHEN alerts exist, THE Dashboard SHALL show an alerts panel with districts that crossed risk thresholds
4. WHEN viewing alerts, THE Dashboard SHALL display Reason_Codes and recommended actions for each alert
5. THE Dashboard SHALL support filtering by state, district, and time period

### Requirement 12: Automated Report Generation

**User Story:** As a UIDAI regional director, I want automated PDF reports, so that I can share risk assessments with stakeholders.

#### Acceptance Criteria

1. WHEN generating reports, THE Report_Generator SHALL produce PDF documents with top-10 at-risk districts per state
2. WHEN generating reports, THE Report_Generator SHALL include ALRI scores, Reason_Codes, and recommended interventions
3. WHEN generating reports, THE Report_Generator SHALL include time-series visualizations and trend analysis
4. THE Report_Generator SHALL support export at state and district granularity

### Requirement 13: Data Serialization and Storage

**User Story:** As a UIDAI system administrator, I want computed scores and results persisted, so that I can track historical trends and audit computations.

#### Acceptance Criteria

1. WHEN scores are computed, THE Storage_Module SHALL serialize ALRI scores and sub-scores to persistent storage
2. WHEN storing data, THE Storage_Module SHALL use JSON format for structured data export
3. WHEN retrieving historical data, THE Storage_Module SHALL support queries by district, date range, and score thresholds
4. FOR ALL valid ALRI records, serializing then deserializing SHALL produce an equivalent record (round-trip property)
