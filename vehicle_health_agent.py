import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import the schema to know which columns are relevant.
from vehicle_telemetry_schema import get_agent_relevant_columns

@dataclass
class HealthFinding:
    """Represents a single finding (positive or negative) about vehicle health."""
    check_name: str
    status: str  # e.g., 'OK', 'Warning', 'Critical'
    details: str
    severity: int # 0 for OK, 1 for Warning, 2 for Critical

@dataclass
class VehicleHealthReport:
    """A structured report summarizing the health of a single vehicle."""
    vehicle_id: str
    overall_status: str
    overall_score: float # 0-100 score
    findings: List[HealthFinding]
    recommendations: List[str]

class VehicleHealthAgent:
    """
    Specialist agent for analyzing vehicle health from telemetry data.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initializes the VehicleHealthAgent."""
        self.logger = logger or logging.getLogger(__name__)
        # Define the thresholds for our health checks.
        self.thresholds = {
            'max_engine_temp_celsius': 105.0,
            'min_battery_voltage_running': 12.8,
        }

    def run(self, state: Dict) -> Dict:
        """Main entry point for the agent's execution within the LangGraph."""
        self.logger.info("VehicleHealthAgent: Starting vehicle health analysis...")
        
        telemetry_data = state.get("telemetry_data", {})
        if not telemetry_data:
            self.logger.warning("No telemetry data available to analyze.")
            return state

        all_reports = []
        for vehicle_id, df in telemetry_data.items():
            self.logger.info(f"Analyzing health for vehicle: {vehicle_id}")
            report = self._analyze_single_vehicle(vehicle_id, df)
            all_reports.append(asdict(report))

        # Update the state with the analysis results.
        if 'analysis_results' not in state or state['analysis_results'] is None:
            state['analysis_results'] = []
        state['analysis_results'].extend(all_reports)
        
        # For simplicity, the final report will be the report of the first vehicle analyzed.
        # The FleetAnalystAgent will later be responsible for synthesizing multiple reports.
        state['final_report'] = self._format_report_for_user(all_reports)

        self.logger.info(f"Vehicle health analysis complete for {len(all_reports)} vehicle(s).")
        return state

    def _analyze_single_vehicle(self, vehicle_id: str, df: pd.DataFrame) -> VehicleHealthReport:
        """Performs all health checks on a single vehicle's data."""
        findings = []
        recommendations = []

        # Run each check
        findings.append(self._check_diagnostic_codes(df))
        findings.append(self._check_engine_temperature(df))
        findings.append(self._check_battery_voltage(df))

        # Calculate overall status and score
        total_severity = sum(f.severity for f in findings)
        overall_score = max(0, 100 - (total_severity * 33.3))

        if total_severity >= 2:
            overall_status = "Critical Issues Detected"
            recommendations.append("Immediate service is recommended.")
        elif total_severity == 1:
            overall_status = "Warning Issued"
            recommendations.append("Service at the next opportunity is recommended.")
        else:
            overall_status = "All Systems Normal"
            recommendations.append("No immediate action required. Follow standard maintenance schedule.")

        return VehicleHealthReport(
            vehicle_id=vehicle_id,
            overall_status=overall_status,
            overall_score=overall_score,
            findings=findings,
            recommendations=list(set(recommendations)) # Remove duplicates
        )

    def _check_diagnostic_codes(self, df: pd.DataFrame) -> HealthFinding:
        """Checks for active Diagnostic Trouble Codes (DTCs)."""
        if 'dtc_codes' not in df.columns:
            return HealthFinding('DTC Check', 'Warning', 'DTC data not available.', 1)

        active_codes = df[df['dtc_codes'] != '']['dtc_codes'].unique()
        
        if len(active_codes) > 0:
            codes_str = ", ".join(active_codes)
            return HealthFinding('DTC Check', 'Critical', f"Active fault codes detected: {codes_str}", 2)
        else:
            return HealthFinding('DTC Check', 'OK', 'No active fault codes found.', 0)

    def _check_engine_temperature(self, df: pd.DataFrame) -> HealthFinding:
        """Analyzes engine temperature for overheating."""
        if 'engine_temp' not in df.columns:
            return HealthFinding('Engine Temp', 'Warning', 'Engine temperature data not available.', 1)

        max_temp = df['engine_temp'].max()
        
        if max_temp > self.thresholds['max_engine_temp_celsius']:
            details = f"Engine temperature reached a critical level of {max_temp:.1f}°C (Threshold: {self.thresholds['max_engine_temp_celsius']:.1f}°C)."
            return HealthFinding('Engine Temp', 'Critical', details, 2)
        else:
            details = f"Maximum engine temperature was {max_temp:.1f}°C, which is within the normal range."
            return HealthFinding('Engine Temp', 'OK', details, 0)

    def _check_battery_voltage(self, df: pd.DataFrame) -> HealthFinding:
        """Analyzes battery voltage for signs of electrical issues."""
        if 'battery_voltage' not in df.columns:
            return HealthFinding('Battery Voltage', 'Warning', 'Battery voltage data not available.', 1)

        # We only care about voltage when the engine is running (higher RPM)
        running_df = df[df['engine_rpm'] > 500]
        if running_df.empty:
            return HealthFinding('Battery Voltage', 'OK', 'Engine was not running; no charging voltage to check.', 0)

        min_voltage = running_df['battery_voltage'].min()
        
        if min_voltage < self.thresholds['min_battery_voltage_running']:
            details = f"Charging system voltage dropped to a warning level of {min_voltage:.2f}V (Threshold: {self.thresholds['min_battery_voltage_running']:.2f}V)."
            return HealthFinding('Battery Voltage', 'Warning', details, 1)
        else:
            details = f"Minimum charging voltage was {min_voltage:.2f}V, which is healthy."
            return HealthFinding('Battery Voltage', 'OK', details, 0)
            
    def _format_report_for_user(self, reports: List[Dict]) -> str:
        """Creates a simple, human-readable summary of the analysis."""
        if not reports:
            return "No vehicle health analysis was performed."

        # For now, just format the first report.
        report = reports[0]
        lines = []
        lines.append(f"Vehicle Health Report for: {report['vehicle_id']}")
        lines.append(f"Overall Status: {report['overall_status']} (Score: {report['overall_score']:.0f}/100)")
        lines.append("\n--- Findings ---")
        for finding_dict in report['findings']:
            finding = HealthFinding(**finding_dict) # Recreate dataclass for easy access
            lines.append(f"- {finding.check_name}: {finding.status}")
            lines.append(f"  - Details: {finding.details}")
        
        lines.append("\n--- Recommendations ---")
        for rec in report['recommendations']:
            lines.append(f"- {rec}")
            
        return "\n".join(lines)


def vehicle_health_node(state: Dict) -> Dict:
    """LangGraph node function for the Vehicle Health Agent."""
    agent = VehicleHealthAgent()
    return agent.run(state)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("Sentinel Fleet: Standalone Test for VehicleHealthAgent")
    print("=" * 60)

    # We need data to test. Let's load the file we generated.
    data_file = 'fleet_telemetry_data.parquet'
    if not os.path.exists(data_file):
        print(f"FATAL ERROR: Data file '{data_file}' not found. Please run synthetic_data_generator.py first.")
    else:
        # We also need the ground truth to verify our agent's findings.
        ground_truth_file = 'fleet_ground_truth.json'
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
            
        # Find a trip that has a known health problem to test against.
        health_scenario_trip_id = None
        for scenario_log in ground_truth:
            if scenario_log['scenario']['scenario_type'] == 'vehicle_health':
                health_scenario_trip_id = scenario_log['trip_id']
                print(f"Found a test trip with a known health problem: {health_scenario_trip_id}")
                break
        
        if not health_scenario_trip_id:
            print("Could not find a trip with a health scenario in the ground truth file.")
        else:
            # Load the full dataset and isolate just the data for our test trip.
            full_df = pd.read_parquet(data_file)
            test_df = full_df[full_df['trip_id'] == health_scenario_trip_id]
            vehicle_id = test_df['vehicle_id'].iloc[0]

            # Create the mock state that this agent will receive from the orchestrator.
            mock_state = {
                "user_query": "Test query for vehicle health",
                "telemetry_data": {
                    vehicle_id: test_df
                }
            }
            
            print(f"\n--- Running agent on trip with known health issue ({vehicle_id}) ---")
            final_state = vehicle_health_node(mock_state)
            
            print("\n--- Agent's Final Report ---")
            print(final_state.get('final_report'))
            print("\n" + "=" * 60)