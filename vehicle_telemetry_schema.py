from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class GroundTruthScenario:
    """
    Defines a known, programmable scenario to be injected into the synthetic data.
    This is the core of our agent validation strategy, allowing us to quantitatively
    measure if our agents correctly detect pre-defined problems.
    """
    scenario_id: str
    scenario_type: str  # e.g., 'vehicle_health', 'driver_behavior'
    description: str
    # What the agents are expected to find. This is our "answer key".
    expected_detection: Dict
    # How to modify the base data to create this specific scenario.
    data_modifications: Dict

# --- Core Data Schema Definition ---

# This dictionary is the master blueprint. It defines every column in our telemetry data,
# its data type, and its purpose. All other parts of the system will refer to this.
TELEMETRY_SCHEMA = {
    # === Identity & Temporal Fields ===
    'timestamp': 'datetime64[ns, UTC]',  # UTC timestamp for global consistency
    'vehicle_id': 'string',              # Unique vehicle identifier (e.g., 'TRUCK_001')
    'trip_id': 'string',                 # Unique identifier for each trip
    'driver_id': 'string',               # Driver identifier for behavior analysis

    # === GPS & Location Data (Simulated to mimic real-world patterns) ===
    'gps_latitude': 'float64',           # Decimal degrees
    'gps_longitude': 'float64',          # Decimal degrees
    'gps_altitude': 'float64',           # Meters above sea level
    'gps_speed': 'float64',              # Speed in km/h, derived from GPS
    'gps_heading': 'float64',            # Vehicle heading in degrees (0-360)

    # === Engine & Powertrain Data (Synthetic with controllable faults) ===
    'engine_rpm': 'float64',             # Revolutions per minute
    'engine_temp': 'float64',            # Engine coolant temperature in Celsius
    'engine_load': 'float64',            # Calculated engine load percentage (0-100)
    'throttle_position': 'float64',      # Throttle position percentage (0-100)

    # === Vehicle Dynamics (Simulated based on driver behavior profiles) ===
    'acceleration_x': 'float64',         # Forward/backward acceleration in m/s²
    'steering_angle': 'float64',         # Steering wheel angle in degrees

    # === Electrical System ===
    'battery_voltage': 'float64',        # Battery voltage, key for electrical health

    # === Fuel System ===
    'fuel_level': 'float64',             # Fuel level percentage (0-100)
    'fuel_flow_rate': 'float64',         # Fuel consumption rate in L/h

    # === Diagnostic Data (Crucial for the VehicleHealthAgent) ===
    'dtc_codes': 'string',               # Comma-separated Diagnostic Trouble Codes (e.g., 'P0217')
    'mil_status': 'bool',                # Malfunction Indicator Lamp status (True if 'Check Engine' is on)

    # === Ground Truth Fields (For validation, not for agent analysis) ===
    # These fields are added by the generator to make testing easier.
    'ground_truth_scenario_id': 'string',# ID of the scenario injected into this data row
    'is_idling': 'bool',                 # Derived field: True if engine running but speed is near zero
    'hard_brake_event': 'bool',          # Derived field: True if deceleration exceeds threshold
    'hard_accel_event': 'bool',          # Derived field: True if acceleration exceeds threshold
}


# --- Predefined Ground Truth Scenarios for Validation ---

# This list contains the specific, testable problems we will inject into our data.
# Each scenario is a mini-test case for our specialist agents.
VALIDATION_SCENARIOS = [
    # A scenario to test the VehicleHealthAgent's ability to detect overheating.
    GroundTruthScenario(
        scenario_id="vh_overheat_01",
        scenario_type="vehicle_health",
        description="Gradual engine temperature increase to dangerous levels.",
        expected_detection={
            "agent": "VehicleHealthAgent",
            "alert_type": "engine_overheat",
            "severity": "critical",
            "recommendation": "Advise driver to stop immediately and service the vehicle."
        },
        data_modifications={
            "engine_temp": "increase_to:115",  # Normal is ~90°C
            "dtc_codes": "set:P0217",          # Official code for 'Engine Overtemperature'
            "mil_status": "set:True"
        }
    ),
    # A scenario to test the VehicleHealthAgent's ability to detect battery issues.
    GroundTruthScenario(
        scenario_id="vh_low_battery_02",
        scenario_type="vehicle_health",
        description="Battery voltage drops below critical threshold while engine is running.",
        expected_detection={
            "agent": "VehicleHealthAgent",
            "alert_type": "low_battery_voltage",
            "severity": "warning",
            "recommendation": "Check alternator and battery health at next service."
        },
        data_modifications={
            "battery_voltage": "decrease_to:11.5", # Normal is >12.4V when running
            "dtc_codes": "set:P0562"               # Official code for 'System Voltage Low'
        }
    ),
    # A scenario to test the DriverBehaviorAgent's ability to detect aggressive driving.
    GroundTruthScenario(
        scenario_id="db_aggressive_driver_01",
        scenario_type="driver_behavior",
        description="Driver with frequent hard braking and rapid acceleration events.",
        expected_detection={
            "agent": "DriverBehaviorAgent",
            "safety_score": "<50",
            "primary_issue": "aggressive_driving",
            "recommendation": "Recommend driver coaching on smooth acceleration and braking."
        },
        data_modifications={
            "hard_accel_event": "inject_events:15", # Inject 15 hard acceleration events
            "hard_brake_event": "inject_events:12", # Inject 12 hard braking events
            "fuel_flow_rate": "increase_by_percent:20" # Aggressive driving uses more fuel
        }
    )
]


# --- Helper Functions for Schema Management ---

def get_base_telemetry_columns() -> List[str]:
    """Returns the complete list of all defined telemetry columns."""
    return list(TELEMETRY_SCHEMA.keys())

def get_validation_dtypes() -> Dict[str, str]:
    """Returns a dictionary of column names to their pandas data types for validation."""
    return TELEMETRY_SCHEMA

def get_agent_relevant_columns() -> Dict[str, List[str]]:
    """
    Maps each specialist agent to the data columns they need to perform their job.
    This allows the orchestrator to provide only the necessary data to each agent,
    which is an efficient and secure practice.
    """
    return {
        "DataIngestionAgent": [
            'timestamp', 'vehicle_id', 'trip_id'
        ],
        "VehicleHealthAgent": [
            'timestamp', 'engine_rpm', 'engine_temp', 'engine_load',
            'battery_voltage', 'dtc_codes', 'mil_status'
        ],
        "DriverBehaviorAgent": [
            'timestamp', 'gps_speed', 'acceleration_x',
            'hard_brake_event', 'hard_accel_event'
        ],
        "FleetAnalystAgent": [
             # This agent typically needs summary data from other agents,
             # but might access high-level data like these columns.
            'timestamp', 'vehicle_id', 'driver_id', 'fuel_level'
        ]
    }

# This block allows the file to be run directly to print out schema info for verification.
# It's a good practice for ensuring the configuration is readable and valid on its own.
if __name__ == "__main__":
    print("=" * 60)
    print("Sentinel Fleet: Vehicle Telemetry Schema Verification")
    print("=" * 60)
    print(f"Total Columns Defined: {len(get_base_telemetry_columns())}")
    print(f"Total Validation Scenarios: {len(VALIDATION_SCENARIOS)}")
    print("\n--- Agent Data Requirements ---")
    for agent, cols in get_agent_relevant_columns().items():
        print(f"  - {agent}: {len(cols)} columns assigned")

    print("\n--- Sample Scenario ---")
    sample_scenario = VALIDATION_SCENARIOS[0]
    print(f"ID: {sample_scenario.scenario_id}")
    print(f"Type: {sample_scenario.scenario_type}")
    print(f"Description: {sample_scenario.description}")
    print(f"Expected Detection by AI: {sample_scenario.expected_detection}")
    print("\nSchema is valid and ready for use.")
    print("=" * 60)