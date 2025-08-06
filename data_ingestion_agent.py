import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# Import our schema to use as the validation reference.
from vehicle_telemetry_schema import get_validation_dtypes, get_base_telemetry_columns

@dataclass
class DataQualityReport:
    """A structured report detailing the quality of data for a single vehicle."""
    is_suitable: bool
    quality_score: float  # Overall score from 0-100.
    total_samples: int
    issues_found: List[str]
    missing_data_summary: Dict[str, float]

class DataIngestionAgent:
    """Agent responsible for data loading, validation, and quality assessment."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initializes the DataIngestionAgent."""
        self.logger = logger or logging.getLogger(__name__)
        self.expected_dtypes = get_validation_dtypes()

    def run(self, state: Dict) -> Dict:
        """
        Main entry point for the agent's execution within the LangGraph.
        This function orchestrates the loading and validation process.
        """
        self.logger.info("DataIngestionAgent: Starting data ingestion and validation...")
        
        try:
            data_source = state.get("data_source")
            if not data_source:
                raise ValueError("Data source path not provided in state.")

            # 1. Load the raw data from the source file.
            raw_data = self._load_data(data_source)

            # 2. Process data for each vehicle requested in the state.
            all_reports = {}
            clean_vehicle_data = {}
            all_issues = []
            
            # Use all vehicles from the file if none are specified in the query.
            vehicle_ids_to_process = state.get("vehicle_ids") or list(raw_data['vehicle_id'].unique())
            
            for vehicle_id in vehicle_ids_to_process:
                self.logger.info(f"Processing data for vehicle: {vehicle_id}")
                vehicle_df = raw_data[raw_data['vehicle_id'] == vehicle_id].copy()

                if vehicle_df.empty:
                    self.logger.warning(f"No data found for vehicle {vehicle_id} in the specified source.")
                    continue

                # 3. Perform validation and generate a quality report.
                quality_report = self._validate_and_assess(vehicle_df)
                
                # 4. Store the results for this vehicle.
                all_reports[vehicle_id] = asdict(quality_report)
                if quality_report.is_suitable:
                    clean_vehicle_data[vehicle_id] = vehicle_df
                
                all_issues.extend(f"{vehicle_id}: {issue}" for issue in quality_report.issues_found)

            # 5. Update the shared state object with the results.
            state['telemetry_data'] = clean_vehicle_data
            state['data_quality_reports'] = all_reports
            state['missing_data_flags'] = all_issues
            # The overall dataset is suitable if at least one vehicle's data is suitable.
            state['is_data_suitable'] = any(r['is_suitable'] for r in all_reports.values())

            self.logger.info(f"Data ingestion complete. Processed {len(all_reports)} vehicles.")

        except Exception as e:
            self.logger.error(f"Critical error during data ingestion: {e}", exc_info=True)
            state['missing_data_flags'] = [f"Critical Ingestion Error: {e}"]
            state['is_data_suitable'] = False
            state['telemetry_data'] = {}
            state['data_quality_reports'] = {}

        return state

    def _load_data(self, data_source: str) -> pd.DataFrame:
        """Loads data from a parquet file and performs initial type casting."""
        self.logger.info(f"Loading data from: {data_source}")
        try:
            df = pd.read_parquet(data_source)
            # Ensure data types match the schema as best as possible on load.
            for col, dtype in self.expected_dtypes.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype, errors='ignore')
            return df
        except FileNotFoundError:
            self.logger.error(f"Data file not found at path: {data_source}")
            raise

    def _validate_and_assess(self, df: pd.DataFrame) -> DataQualityReport:
        """Performs a series of validation checks and returns a quality report."""
        issues = []
        score = 100.0

        # Check 1: Data Completeness (missing values)
        missing_summary = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        total_missing_pct = sum(missing_summary.values())
        if total_missing_pct > 1.0: # Allow for very minor missing data
            issues.append(f"Data has {total_missing_pct:.2f}% total missing values.")
            score -= total_missing_pct * 0.5 # Penalize based on severity

        # Check 2: Timestamp Integrity
        if 'timestamp' in df.columns:
            if df['timestamp'].isnull().any():
                issues.append("Timestamps contain null values.")
                score -= 10
            elif not df['timestamp'].is_monotonic_increasing:
                issues.append("Timestamps are not in chronological order.")
                score -= 15
        else:
            issues.append("Critical 'timestamp' column is missing.")
            score = 0 # Cannot proceed without timestamps

        # Check 3: Data Volume
        total_samples = len(df)
        if total_samples < 100:
            issues.append(f"Insufficient data volume ({total_samples} records).")
            score -= 20

        # Check 4: Physical Plausibility
        if 'gps_speed' in df.columns and df['gps_speed'].max() > 200:
            issues.append("Detected implausible speed values (>200 km/h).")
            score -= 10
        if 'engine_temp' in df.columns and (df['engine_temp'].max() > 150 or df['engine_temp'].min() < -20):
            issues.append("Detected implausible engine temperature values.")
            score -= 10

        final_score = max(0, score)
        is_suitable = final_score > 60

        return DataQualityReport(
            is_suitable=is_suitable,
            quality_score=final_score,
            total_samples=total_samples,
            issues_found=issues,
            missing_data_summary=missing_summary
        )

# This is the function that will be registered as a node in our LangGraph graph.
# It acts as a clean entry point to the agent's functionality.
def data_ingestion_node(state: Dict) -> Dict:
    """LangGraph node function for the Data Ingestion Agent."""
    agent = DataIngestionAgent()
    return agent.run(state)

# Main execution block for standalone testing of this agent.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("Sentinel Fleet: Standalone Test for DataIngestionAgent")
    print("=" * 60)

    # Pre-requisite: Ensure data exists.
    data_file = 'fleet_telemetry_data.parquet'
    if not pd.io.common.file_exists(data_file):
        print(f"Data file '{data_file}' not found. Please run synthetic_data_generator.py first.")
    else:
        # Create a mock state object, simulating what LangGraph would provide.
        mock_state = {
            "user_query": "Check TRUCK_001 for overheating issues.",
            "vehicle_ids": ["TRUCK_001", "VAN_003"], # Request data for two specific vehicles
            "data_source": data_file
        }

        print(f"Simulating agent run for vehicles: {mock_state['vehicle_ids']}...")
        
        # Execute the agent's main function
        final_state = data_ingestion_node(mock_state)

        # Print a summary of the results
        print("\n--- Agent Execution Summary ---")
        if final_state.get('is_data_suitable'):
            print("✅ Data deemed SUITABLE for analysis.")
        else:
            print("❌ Data deemed NOT SUITABLE for analysis.")
        
        print(f"\nIssues found: {len(final_state.get('missing_data_flags', []))}")
        for issue in final_state.get('missing_data_flags', []):
            print(f"  - {issue}")
            
        print("\n--- Quality Reports ---")
        for vehicle_id, report_dict in final_state.get('data_quality_reports', {}).items():
            print(f"  Vehicle: {vehicle_id}")
            print(f"    - Quality Score: {report_dict['quality_score']:.1f}/100")
            print(f"    - Total Samples: {report_dict['total_samples']}")
            print(f"    - Issues: {len(report_dict['issues_found'])}")
            
        print("\n" + "=" * 60)