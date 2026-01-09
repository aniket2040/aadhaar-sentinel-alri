# Implementation Plan: Aadhaar Sentinel ALRI Platform

## Overview

This implementation plan builds the Aadhaar Sentinel platform incrementally, starting with core data processing, then scoring logic, explainability, forecasting, clustering, and finally the presentation layer (dashboard and reports). Each task builds on previous work, ensuring no orphaned code.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `src/` directory structure with subfolders: `etl/`, `scoring/`, `anomaly/`, `explainability/`, `recommendations/`, `forecasting/`, `clustering/`, `dashboard/`, `reports/`, `storage/`
  - Create `requirements.txt` with: pandas, numpy, scipy, statsmodels, prophet, scikit-learn, streamlit, plotly, fpdf2, hypothesis
  - Create `__init__.py` files for all packages
  - Create `conftest.py` with shared test fixtures and Hypothesis generators
  - _Requirements: 1.1-1.6, 13.1-13.4_

- [x] 2. Implement ETL Pipeline
  - [x] 2.1 Implement DataLoader class
    - Create `src/etl/data_loader.py` with `load_enrollment()`, `load_demographic()`, `load_biometric()` methods
    - Implement schema validation for each CSV type
    - Handle type coercion for state, district (str), pincode (int), date fields (int), counts (int)
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Write property test for CSV parsing
    - **Property 1: CSV Parsing Correctness**
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 2.3 Implement MonthlyAggregator class
    - Create `src/etl/aggregator.py` with `aggregate_by_district_month()` method
    - Group by state, district, year, month and sum totals
    - Implement `compute_baselines()` for district-level baseline metrics
    - _Requirements: 1.4_

  - [x] 2.4 Write property test for aggregation
    - **Property 2: Aggregation Sum Invariant**
    - **Validates: Requirements 1.4**

  - [x] 2.5 Implement MissingValueHandler class
    - Create `src/etl/imputer.py` with `impute()` method
    - Support median imputation strategy
    - Log affected records with timestamps
    - _Requirements: 1.5_

- [x] 3. Checkpoint - Verify ETL pipeline
  - Ensure all ETL tests pass, ask the user if questions arise.

- [x] 4. Implement ALRI Calculator Core
  - [x] 4.1 Implement ALRIWeights and ALRIResult dataclasses
    - Create `src/scoring/alri_calculator.py` with dataclass definitions
    - Default weights: coverage=0.30, instability=0.30, biometric=0.30, anomaly=0.10
    - _Requirements: 6.1, 6.3_

  - [x] 4.2 Implement Coverage Risk computation
    - Add `compute_coverage_risk()` method to ALRICalculator
    - Calculate enrollment rates relative to district baseline
    - Apply z-score normalization and clip to [0,1]
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 4.3 Implement Data Instability Risk computation
    - Add `compute_instability_risk()` method to ALRICalculator
    - Calculate demographic update rates per 1000 enrollments
    - Measure rolling volatility (std) of update volumes
    - Scale to [0,1] range
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.4 Implement Biometric Compliance Risk computation
    - Add `compute_biometric_risk()` method to ALRICalculator
    - Calculate biometric update rates relative to expected volumes
    - Scale to [0,1] with higher missing updates yielding higher risk
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 4.5 Implement ALRI score aggregation
    - Add `compute_alri()` method combining all sub-scores
    - Apply weighted formula: ALRI = (w1×C + w2×D + w3×B + w4×A) × 100
    - Ensure output in [0,100] range
    - _Requirements: 6.1, 6.2_

  - [x] 4.6 Write property tests for scoring
    - **Property 3: Sub-score Range Invariant**
    - **Property 4: ALRI Score Range Invariant**
    - **Property 5: ALRI Weighted Sum Correctness**
    - **Property 6: Coverage Risk Monotonicity**
    - **Property 7: Instability Risk Monotonicity**
    - **Validates: Requirements 2.3, 2.4, 3.3, 3.4, 4.3, 5.3, 6.1, 6.2**

- [x] 5. Checkpoint - Verify ALRI Calculator
  - Ensure all scoring tests pass, ask the user if questions arise.

- [x] 6. Implement Anomaly Detection
  - [x] 6.1 Implement STLAnomalyDetector class
    - Create `src/anomaly/detector.py` with `decompose()` method using statsmodels STL
    - Implement `detect_anomalies()` flagging residuals > 3σ
    - Implement `compute_anomaly_factor()` returning [0,1] score
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 Write property tests for anomaly detection
    - **Property 8: Anomaly Detection Threshold**
    - **Property 9: STL Decomposition Reconstruction**
    - **Validates: Requirements 5.1, 5.2**

- [x] 7. Implement Explainability Layer
  - [x] 7.1 Implement ReasonCodeGenerator class
    - Create `src/explainability/reason_codes.py` with Severity enum and ReasonCode dataclass
    - Implement `generate()` method identifying top contributing sub-scores
    - Implement `determine_severity()` mapping scores to Low/Medium/High/Critical
    - Rank reason codes by contribution magnitude
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 7.2 Write property tests for reason codes
    - **Property 10: Reason Code Ranking**
    - **Property 11: Reason Code Completeness**
    - **Validates: Requirements 7.2, 7.3, 7.4**

  - [x] 7.3 Implement RecommendationEngine class
    - Create `src/recommendations/engine.py` with Intervention dataclass
    - Define INTERVENTION_MAP mapping reason codes to 1-3 interventions
    - Implement `recommend()` method with priority ordering
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 7.4 Write property tests for recommendations
    - **Property 12: Recommendation Mapping**
    - **Property 13: Recommendation Priority Ordering**
    - **Validates: Requirements 8.1, 8.4**

- [x] 8. Checkpoint - Verify Explainability
  - Ensure all explainability tests pass, ask the user if questions arise.

- [x] 9. Implement Forecasting Module
  - [x] 9.1 Implement TimeSeriesForecaster class
    - Create `src/forecasting/forecaster.py` with ForecastResult dataclass
    - Implement `fit()` and `predict()` methods using Prophet
    - Generate forecasts with confidence intervals for 3-6 months
    - Implement `detect_declining_trend()` for proactive flagging
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 9.2 Write property test for forecasting
    - **Property 14: Forecast Confidence Interval**
    - **Validates: Requirements 9.2**

- [x] 10. Implement Clustering Module
  - [x] 10.1 Implement DistrictClusterer class
    - Create `src/clustering/segmentation.py` with ClusterProfile dataclass
    - Implement `fit_predict()` using KMeans with configurable n_clusters
    - Define cluster labels: Stable-HighCoverage, Migratory-HighChurn, ChildGap-HighRisk, LowActivity-Rural
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 10.2 Write property test for clustering
    - **Property 15: Cluster Assignment Completeness**
    - **Validates: Requirements 10.2, 10.3**

- [x] 11. Implement Storage Module
  - [x] 11.1 Implement ALRISerializer and ALRIStorage classes
    - Create `src/storage/serializer.py` with ALRIRecord dataclass
    - Implement `serialize()` and `deserialize()` for JSON conversion
    - Implement `save()`, `load()`, and `query()` methods
    - Support queries by district, date_range, min_score
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x] 11.2 Write property tests for storage
    - **Property 16: ALRI Record Serialization Round-Trip**
    - **Property 17: Query Filter Correctness**
    - **Validates: Requirements 13.3, 13.4**

- [x] 12. Checkpoint - Verify Core Modules
  - Ensure all core module tests pass, ask the user if questions arise.

- [x] 13. Implement Dashboard
  - [x] 13.1 Create Streamlit dashboard app
    - Create `src/dashboard/app.py` with AadhaarSentinelDashboard class
    - Implement `render_heatmap()` using Plotly choropleth for district ALRI scores
    - Implement `render_district_detail()` with time-series charts
    - Implement `render_alerts_panel()` showing threshold-crossing districts
    - Implement `render_filters()` for state/district/time selection
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 14. Implement Report Generator
  - [x] 14.1 Create PDF report generator
    - Create `src/reports/generator.py` with PDFReportGenerator class
    - Implement `generate_state_report()` with top-10 at-risk districts
    - Implement `generate_district_report()` with detailed metrics
    - Include ALRI scores, reason codes, recommendations, and trend visualizations
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 15. Create main entry point and integration
  - [x] 15.1 Create main.py orchestration script
    - Create `main.py` that loads data, runs ETL, computes ALRI for all districts
    - Generate reason codes and recommendations
    - Save results to JSON storage
    - _Requirements: All_

  - [x] 15.2 Create run_dashboard.py launcher
    - Create `run_dashboard.py` to launch Streamlit app
    - Load precomputed ALRI results from storage
    - _Requirements: 11.1-11.5_

- [x] 16. Final Checkpoint - Full Integration
  - Ensure all tests pass and dashboard runs correctly, ask the user if questions arise.

## Notes

- All tasks including property tests are required for comprehensive coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
