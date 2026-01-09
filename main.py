#!/usr/bin/env python
"""
Aadhaar Sentinel Main Orchestration Script

This script orchestrates the complete ALRI computation pipeline:
1. Loads enrollment, demographic, and biometric data from CSV files
2. Runs ETL pipeline (aggregation, imputation)
3. Computes ALRI scores for all districts
4. Generates reason codes and recommendations
5. Saves results to JSON storage

Usage:
    python main.py [--enrollment PATH] [--demographic PATH] [--biometric PATH] [--output PATH]

Requirements: All
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.etl.data_loader import DataLoader, SchemaValidationError
from src.etl.aggregator import MonthlyAggregator
from src.etl.imputer import MissingValueHandler
from src.scoring.alri_calculator import ALRICalculator, ALRIResult, ALRIWeights
from src.anomaly.detector import STLAnomalyDetector
from src.explainability.reason_codes import ReasonCodeGenerator
from src.recommendations.engine import RecommendationEngine
from src.storage.serializer import ALRIRecord, ALRIStorage, StorageError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AadhaarSentinelPipeline:
    """
    Main orchestration class for the Aadhaar Sentinel ALRI pipeline.
    
    Coordinates data loading, ETL processing, ALRI computation,
    reason code generation, and result storage.
    """
    
    DEFAULT_ENROLLMENT_PATH = "enrollment.csv"
    DEFAULT_DEMOGRAPHIC_PATH = "demographic.csv"
    DEFAULT_BIOMETRIC_PATH = "biometric.csv"
    DEFAULT_OUTPUT_PATH = "data/alri_storage/alri_results.json"
    
    def __init__(
        self,
        enrollment_path: str = None,
        demographic_path: str = None,
        biometric_path: str = None,
        output_path: str = None,
        weights: ALRIWeights = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            enrollment_path: Path to enrollment CSV file
            demographic_path: Path to demographic CSV file
            biometric_path: Path to biometric CSV file
            output_path: Path for output JSON file
            weights: Custom ALRI weights (optional)
        """
        self.enrollment_path = enrollment_path or self.DEFAULT_ENROLLMENT_PATH
        self.demographic_path = demographic_path or self.DEFAULT_DEMOGRAPHIC_PATH
        self.biometric_path = biometric_path or self.DEFAULT_BIOMETRIC_PATH
        self.output_path = output_path or self.DEFAULT_OUTPUT_PATH
        
        # Initialize components
        self.data_loader = DataLoader()
        self.aggregator = MonthlyAggregator()
        self.imputer = MissingValueHandler()
        self.alri_calculator = ALRICalculator(weights=weights)
        self.anomaly_detector = STLAnomalyDetector()
        self.reason_code_generator = ReasonCodeGenerator()
        self.recommendation_engine = RecommendationEngine()
        self.storage = ALRIStorage()
        
        # Data containers
        self.enrollment_df: Optional[pd.DataFrame] = None
        self.demographic_df: Optional[pd.DataFrame] = None
        self.biometric_df: Optional[pd.DataFrame] = None
        self.aggregated_data: Dict[str, pd.DataFrame] = {}
        self.baselines: Dict[str, pd.DataFrame] = {}
        self.alri_results: List[ALRIResult] = []
        self.alri_records: List[ALRIRecord] = []
    
    def load_data(self) -> bool:
        """
        Load all CSV data files.
        
        Returns:
            True if all files loaded successfully, False otherwise
        """
        logger.info("Loading data files...")
        
        try:
            # Load enrollment data
            if os.path.exists(self.enrollment_path):
                self.enrollment_df = self.data_loader.load_enrollment(self.enrollment_path)
                logger.info(f"Loaded {len(self.enrollment_df)} enrollment records")
            else:
                logger.warning(f"Enrollment file not found: {self.enrollment_path}")
                self.enrollment_df = pd.DataFrame()
            
            # Load demographic data
            if os.path.exists(self.demographic_path):
                self.demographic_df = self.data_loader.load_demographic(self.demographic_path)
                logger.info(f"Loaded {len(self.demographic_df)} demographic records")
            else:
                logger.warning(f"Demographic file not found: {self.demographic_path}")
                self.demographic_df = pd.DataFrame()
            
            # Load biometric data
            if os.path.exists(self.biometric_path):
                self.biometric_df = self.data_loader.load_biometric(self.biometric_path)
                logger.info(f"Loaded {len(self.biometric_df)} biometric records")
            else:
                logger.warning(f"Biometric file not found: {self.biometric_path}")
                self.biometric_df = pd.DataFrame()
            
            return True
            
        except (FileNotFoundError, SchemaValidationError, ValueError) as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def run_etl(self) -> bool:
        """
        Run ETL pipeline: aggregation and imputation.
        
        Returns:
            True if ETL completed successfully, False otherwise
        """
        logger.info("Running ETL pipeline...")
        
        try:
            # Aggregate enrollment data
            if self.enrollment_df is not None and not self.enrollment_df.empty:
                self.aggregated_data['enrollment'] = self.aggregator.aggregate_by_district_month(
                    self.enrollment_df, 'total_enrollment_age'
                )
                self.baselines['enrollment'] = self.aggregator.compute_baselines(
                    self.aggregated_data['enrollment']
                )
                logger.info(f"Aggregated enrollment: {len(self.aggregated_data['enrollment'])} monthly records")
            
            # Aggregate demographic data
            if self.demographic_df is not None and not self.demographic_df.empty:
                self.aggregated_data['demographic'] = self.aggregator.aggregate_by_district_month(
                    self.demographic_df, 'total_demographic_age'
                )
                self.baselines['demographic'] = self.aggregator.compute_baselines(
                    self.aggregated_data['demographic']
                )
                logger.info(f"Aggregated demographic: {len(self.aggregated_data['demographic'])} monthly records")
            
            # Aggregate biometric data
            if self.biometric_df is not None and not self.biometric_df.empty:
                self.aggregated_data['biometric'] = self.aggregator.aggregate_by_district_month(
                    self.biometric_df, 'total_biometric_age'
                )
                self.baselines['biometric'] = self.aggregator.compute_baselines(
                    self.aggregated_data['biometric']
                )
                logger.info(f"Aggregated biometric: {len(self.aggregated_data['biometric'])} monthly records")
            
            # Apply imputation to aggregated data
            for data_type, df in self.aggregated_data.items():
                if not df.empty:
                    self.aggregated_data[data_type] = self.imputer.impute(df, strategy='median')
            
            imputed_count = self.imputer.get_affected_count()
            if imputed_count > 0:
                logger.info(f"Imputed {imputed_count} missing values")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in ETL pipeline: {e}")
            return False
    
    def _get_unique_districts(self) -> List[Tuple[str, str]]:
        """
        Get list of unique (state, district) pairs from all data sources.
        
        Returns:
            List of (state, district) tuples
        """
        districts = set()
        
        for df in self.aggregated_data.values():
            if not df.empty and 'state' in df.columns and 'district' in df.columns:
                for _, row in df[['state', 'district']].drop_duplicates().iterrows():
                    districts.add((row['state'], row['district']))
        
        return sorted(list(districts))
    
    def _get_district_data(self, state: str, district: str) -> pd.DataFrame:
        """
        Combine all data sources for a specific district.
        
        Args:
            state: State name
            district: District name
            
        Returns:
            Combined DataFrame with all metrics for the district
        """
        combined_data = []
        
        for data_type, df in self.aggregated_data.items():
            if df.empty:
                continue
            
            district_df = df[
                (df['state'] == state) & (df['district'] == district)
            ].copy()
            
            if not district_df.empty:
                # Rename 'total' column to be specific to data type
                if 'total' in district_df.columns:
                    col_name = f'total_{data_type}_age'
                    district_df = district_df.rename(columns={'total': col_name})
                combined_data.append(district_df)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Merge all data on common columns
        result = combined_data[0]
        for df in combined_data[1:]:
            merge_cols = ['state', 'district', 'year', 'month']
            available_cols = [c for c in merge_cols if c in result.columns and c in df.columns]
            if available_cols:
                result = result.merge(df, on=available_cols, how='outer', suffixes=('', '_dup'))
                # Remove duplicate columns
                result = result[[c for c in result.columns if not c.endswith('_dup')]]
        
        return result
    
    def _compute_anomaly_score(self, district_data: pd.DataFrame) -> float:
        """
        Compute anomaly score for a district using STL decomposition.
        
        Args:
            district_data: DataFrame with district time-series data
            
        Returns:
            Anomaly factor score (0-1)
        """
        # Try to compute anomaly from enrollment data
        for col in ['total_enrollment_age', 'total_demographic_age', 'total_biometric_age']:
            if col in district_data.columns:
                series = district_data[col].dropna()
                if len(series) >= 24:  # Need at least 2 years for STL
                    try:
                        return self.anomaly_detector.compute_anomaly_factor(time_series=series)
                    except ValueError:
                        continue
        
        # Fall back to simple anomaly detection
        return self.alri_calculator.compute_anomaly_factor(district_data)
    
    def compute_alri_scores(self) -> bool:
        """
        Compute ALRI scores for all districts.
        
        Returns:
            True if computation completed successfully, False otherwise
        """
        logger.info("Computing ALRI scores...")
        
        try:
            districts = self._get_unique_districts()
            logger.info(f"Processing {len(districts)} districts")
            
            self.alri_results = []
            skipped_count = 0
            
            for state, district in districts:
                # Get combined district data
                district_data = self._get_district_data(state, district)
                
                if district_data.empty:
                    logger.debug(f"No data for district: {district}, {state}")
                    skipped_count += 1
                    continue
                
                # Get baseline enrollment for normalization
                baseline_enrollment = None
                if 'enrollment' in self.baselines and not self.baselines['enrollment'].empty:
                    baseline_df = self.baselines['enrollment']
                    district_baseline = baseline_df[
                        (baseline_df['state'] == state) & (baseline_df['district'] == district)
                    ]
                    if not district_baseline.empty:
                        baseline_enrollment = district_baseline['baseline_mean'].iloc[0]
                        # Handle NaN baseline
                        if pd.isna(baseline_enrollment):
                            baseline_enrollment = None
                
                # Compute anomaly score
                anomaly_score = self._compute_anomaly_score(district_data)
                # Handle NaN anomaly score
                if pd.isna(anomaly_score):
                    anomaly_score = 0.0
                
                try:
                    # Compute ALRI
                    alri_result = self.alri_calculator.compute_alri(
                        district_data=district_data,
                        district=district,
                        state=state,
                        baseline_enrollment=baseline_enrollment,
                        anomaly_score=anomaly_score
                    )
                    
                    # Validate result - skip if any score is NaN
                    if (pd.isna(alri_result.alri_score) or 
                        pd.isna(alri_result.coverage_risk) or
                        pd.isna(alri_result.instability_risk) or
                        pd.isna(alri_result.biometric_risk) or
                        pd.isna(alri_result.anomaly_factor)):
                        logger.debug(f"Skipping district with NaN scores: {district}, {state}")
                        skipped_count += 1
                        continue
                    
                    # Generate reason codes
                    reason_codes = self.reason_code_generator.generate(alri_result)
                    alri_result.reason_codes = reason_codes
                    
                    # Generate recommendations
                    recommendations = self.recommendation_engine.recommend(reason_codes)
                    alri_result.recommendations = recommendations
                    
                    self.alri_results.append(alri_result)
                    
                except (ValueError, Exception) as e:
                    logger.debug(f"Error computing ALRI for {district}, {state}: {e}")
                    skipped_count += 1
                    continue
            
            logger.info(f"Computed ALRI scores for {len(self.alri_results)} districts")
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} districts due to insufficient data")
            return True
            
        except Exception as e:
            logger.error(f"Error computing ALRI scores: {e}")
            return False
    
    def _convert_to_records(self) -> List[ALRIRecord]:
        """
        Convert ALRIResult objects to ALRIRecord for storage.
        
        Returns:
            List of ALRIRecord objects
        """
        records = []
        
        for result in self.alri_results:
            # Extract reason code strings
            reason_code_strings = [rc.code for rc in result.reason_codes] if result.reason_codes else []
            
            record = ALRIRecord(
                district=result.district,
                state=result.state,
                alri_score=result.alri_score,
                sub_scores={
                    'coverage': result.coverage_risk,
                    'instability': result.instability_risk,
                    'biometric': result.biometric_risk,
                    'anomaly': result.anomaly_factor
                },
                reason_codes=reason_code_strings,
                computed_at=result.computed_at.isoformat() if hasattr(result.computed_at, 'isoformat') else str(result.computed_at)
            )
            records.append(record)
        
        return records
    
    def save_results(self) -> bool:
        """
        Save ALRI results to JSON storage.
        
        Returns:
            True if save completed successfully, False otherwise
        """
        logger.info(f"Saving results to {self.output_path}...")
        
        try:
            # Convert results to records
            self.alri_records = self._convert_to_records()
            
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save to storage
            self.storage.save(self.alri_records, self.output_path)
            
            logger.info(f"Saved {len(self.alri_records)} ALRI records")
            return True
            
        except (StorageError, OSError) as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run(self) -> bool:
        """
        Run the complete ALRI pipeline.
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting Aadhaar Sentinel ALRI Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        if not self.load_data():
            logger.error("Pipeline failed at data loading stage")
            return False
        
        # Step 2: Run ETL
        if not self.run_etl():
            logger.error("Pipeline failed at ETL stage")
            return False
        
        # Step 3: Compute ALRI scores
        if not self.compute_alri_scores():
            logger.error("Pipeline failed at ALRI computation stage")
            return False
        
        # Step 4: Save results
        if not self.save_results():
            logger.error("Pipeline failed at save stage")
            return False
        
        elapsed_time = datetime.now() - start_time
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total time: {elapsed_time}")
        logger.info(f"Districts processed: {len(self.alri_results)}")
        logger.info(f"Results saved to: {self.output_path}")
        logger.info("=" * 60)
        
        return True
    
    def print_summary(self) -> None:
        """Print a summary of the ALRI results."""
        if not self.alri_results:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 60)
        print("ALRI RESULTS SUMMARY")
        print("=" * 60)
        
        # Sort by ALRI score descending
        sorted_results = sorted(self.alri_results, key=lambda x: x.alri_score, reverse=True)
        
        # Top 10 at-risk districts
        print("\nTop 10 At-Risk Districts:")
        print("-" * 60)
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i:2}. {result.district.title()}, {result.state.title()}")
            print(f"    ALRI Score: {result.alri_score:.1f}")
            print(f"    Coverage: {result.coverage_risk:.2f} | Instability: {result.instability_risk:.2f}")
            print(f"    Biometric: {result.biometric_risk:.2f} | Anomaly: {result.anomaly_factor:.2f}")
            if result.reason_codes:
                codes = [rc.code for rc in result.reason_codes[:2]]
                print(f"    Top Reasons: {', '.join(codes)}")
            print()
        
        # Statistics
        scores = [r.alri_score for r in self.alri_results]
        print("-" * 60)
        print(f"Total Districts: {len(self.alri_results)}")
        print(f"Average ALRI Score: {sum(scores) / len(scores):.1f}")
        print(f"Max ALRI Score: {max(scores):.1f}")
        print(f"Min ALRI Score: {min(scores):.1f}")
        
        # Risk distribution
        critical = sum(1 for s in scores if s >= 75)
        high = sum(1 for s in scores if 50 <= s < 75)
        medium = sum(1 for s in scores if 25 <= s < 50)
        low = sum(1 for s in scores if s < 25)
        
        print(f"\nRisk Distribution:")
        print(f"  Critical (>=75): {critical}")
        print(f"  High (50-74): {high}")
        print(f"  Medium (25-49): {medium}")
        print(f"  Low (<25): {low}")
        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aadhaar Sentinel ALRI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py
    python main.py --enrollment data/enrollment.csv --output results/alri.json
    python main.py --summary
        """
    )
    
    parser.add_argument(
        '--enrollment', '-e',
        type=str,
        default=None,
        help='Path to enrollment CSV file (default: enrollment.csv)'
    )
    
    parser.add_argument(
        '--demographic', '-d',
        type=str,
        default=None,
        help='Path to demographic CSV file (default: demographic.csv)'
    )
    
    parser.add_argument(
        '--biometric', '-b',
        type=str,
        default=None,
        help='Path to biometric CSV file (default: biometric.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path for output JSON file (default: data/alri_storage/alri_results.json)'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print summary of results after computation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline
    pipeline = AadhaarSentinelPipeline(
        enrollment_path=args.enrollment,
        demographic_path=args.demographic,
        biometric_path=args.biometric,
        output_path=args.output
    )
    
    success = pipeline.run()
    
    if success and args.summary:
        pipeline.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
