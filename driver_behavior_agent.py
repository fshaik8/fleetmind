import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class BehaviorFinding:
    """Represents a single finding about driver behavior."""
    event_type: str      # e.g., 'Hard Acceleration', 'Hard Braking'
    event_count: int
    severity: str        # e.g., 'Normal', 'Elevated', 'High'

@dataclass
class DriverBehaviorReport:
    """A structured report summarizing the behavior of a single driver."""
    vehicle_id: str
    driver_id: str
    overall_safety_score: float # 0-100 score
    overall_assessment: str     # e.g., 'Safe', 'Aggressive', 'Needs Improvement'
    findings: List[BehaviorFinding]
    recommendations: List[str]

class DriverBehaviorAgent:
    """
    Specialist agent for analyzing driver behavior from telemetry data.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initializes the DriverBehaviorAgent."""
        self.logger = logger or logging.getLogger(__name__)
        # Define thresholds for what constitutes "high" event frequency.
        # This is events per hour of driving.
        self.events_per_hour_thresholds = {
            'hard_accel_event': {'warning': 5, 'high': 10},
            'hard_brake_event': {'warning': 5, 'high': 10},
        }

    def run(self, state: Dict) -> Dict:
        """Main entry point for the agent's execution within LangGraph."""
        self.logger.info("DriverBehaviorAgent: Starting driver behavior analysis...")
        
        telemetry_data = state.get("telemetry_data", {})
        if not telemetry_data:
            self.logger.warning("No telemetry data available to analyze.")
            return state

        all_reports = []
        for vehicle_id, df in telemetry_data.items():
            if df.empty or 'driver_id' not in df.columns:
                continue
            
            driver_id = df['driver_id'].iloc[0]
            self.logger.info(f"Analyzing behavior for driver: {driver_id} in vehicle: {vehicle_id}")
            report = self._analyze_single_driver(vehicle_id, driver_id, df)
            all_reports.append(asdict(report))

        # Update the state with the analysis results.
        if 'analysis_results' not in state or state['analysis_results'] is None:
            state['analysis_results'] = []
        state['analysis_results'].extend(all_reports)
        
        # Format a user-friendly report.
        state['final_report'] = self._format_report_for_user(all_reports)

        self.logger.info(f"Driver behavior analysis complete for {len(all_reports)} driver(s).")
        return state

    def _analyze_single_driver(self, vehicle_id: str, driver_id: str, df: pd.DataFrame) -> DriverBehaviorReport:
        """Performs all behavior checks on a single driver's data."""
        findings = []
        recommendations = []

        # Calculate total driving time in hours for normalization.
        duration_seconds = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        duration_hours = duration_seconds / 3600.0
        # Avoid division by zero for very short trips.
        if duration_hours < 0.01:
            duration_hours = 0.01

        # --- Analyze Specific Events ---
        event_types = ['hard_accel_event', 'hard_brake_event']
        for event in event_types:
            if event in df.columns:
                count = df[event].sum()
                events_per_hour = count / duration_hours
                
                thresholds = self.events_per_hour_thresholds.get(event, {})
                severity = 'Normal'
                if events_per_hour > thresholds.get('high', 10):
                    severity = 'High'
                elif events_per_hour > thresholds.get('warning', 5):
                    severity = 'Elevated'
                
                findings.append(BehaviorFinding(event, int(count), severity))

        # --- Calculate Overall Score and Assessment ---
        safety_score = 100.0
        high_severity_events = 0
        
        for finding in findings:
            if finding.severity == 'Elevated':
                safety_score -= 15 # Penalize moderately
            elif finding.severity == 'High':
                safety_score -= 30 # Penalize heavily
                high_severity_events += 1
        
        safety_score = max(0, safety_score)

        if safety_score < 50 or high_severity_events > 1:
            assessment = "Aggressive Driving Pattern Detected"
            recommendations.append("Driver coaching is strongly recommended to improve safety and efficiency.")
        elif safety_score < 80:
            assessment = "Needs Improvement"
            recommendations.append("Minor improvements in driving style could enhance safety.")
        else:
            assessment = "Safe Driving Pattern"
            recommendations.append("Driver exhibits safe and consistent behavior.")

        return DriverBehaviorReport(
            vehicle_id=vehicle_id,
            driver_id=driver_id,
            overall_safety_score=safety_score,
            overall_assessment=assessment,
            findings=findings,
            recommendations=recommendations
        )
            
    def _format_report_for_user(self, reports: List[Dict]) -> str:
        """Creates a simple, human-readable summary of the analysis."""
        if not reports:
            return "No driver behavior analysis was performed."

        lines = []
        for report in reports: # Format a report for each driver analyzed
            lines.append(f"Driver Behavior Report for: {report['driver_id']} (Vehicle: {report['vehicle_id']})")
            lines.append(f"Overall Assessment: {report['overall_assessment']} (Safety Score: {report['overall_safety_score']:.0f}/100)")
            lines.append("\n--- Key Events ---")
            for finding_dict in report['findings']:
                finding = BehaviorFinding(**finding_dict)
                event_name = finding.event_type.replace('_event', '').replace('_', ' ').title()
                lines.append(f"- {event_name}s: {finding.event_count} instances (Severity: {finding.severity})")
            
            lines.append("\n--- Recommendations ---")
            for rec in report['recommendations']:
                lines.append(f"- {rec}")
            lines.append("\n" + "="*30 + "\n") # Separator for multiple reports
            
        return "\n".join(lines)


def driver_behavior_node(state: Dict) -> Dict:
    """LangGraph node function for the Driver Behavior Agent."""
    agent = DriverBehaviorAgent()
    return agent.run(state)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("Sentinel Fleet: Standalone Test for DriverBehaviorAgent")
    print("=" * 60)

    data_file = 'fleet_telemetry_data.parquet'
    if not os.path.exists(data_file):
        print(f"FATAL ERROR: Data file '{data_file}' not found. Please run synthetic_data_generator.py first.")
    else:
        ground_truth_file = 'fleet_ground_truth.json'
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
            
        # Find a trip that has a known behavior problem to test against.
        behavior_scenario_trip_id = None
        for scenario_log in ground_truth:
            if scenario_log['scenario']['scenario_type'] == 'driver_behavior':
                behavior_scenario_trip_id = scenario_log['trip_id']
                print(f"Found a test trip with a known behavior problem: {behavior_scenario_trip_id}")
                break
        
        if not behavior_scenario_trip_id:
            print("Could not find a trip with a behavior scenario in the ground truth file.")
        else:
            full_df = pd.read_parquet(data_file)
            test_df = full_df[full_df['trip_id'] == behavior_scenario_trip_id]
            vehicle_id = test_df['vehicle_id'].iloc[0]

            mock_state = {
                "user_query": "Test query for driver behavior",
                "telemetry_data": {
                    vehicle_id: test_df
                }
            }
            
            print(f"\n--- Running agent on trip with known behavior issue ({vehicle_id}) ---")
            final_state = driver_behavior_node(mock_state)
            
            print("\n--- Agent's Final Report ---")
            print(final_state.get('final_report'))
            print("\n" + "=" * 60)