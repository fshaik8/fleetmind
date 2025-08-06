import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import uuid
import random
import json
import logging
from dataclasses import asdict
from typing import Dict, List, Tuple

# Import our schema definitions, which serve as the blueprint for this factory.
from vehicle_telemetry_schema import (
    TELEMETRY_SCHEMA,
    VALIDATION_SCENARIOS,
    GroundTruthScenario,
    get_base_telemetry_columns,
    get_validation_dtypes
)

class VehicleTelemetryGenerator:
    """
    Generates a complete, realistic, synthetic vehicle telemetry dataset
    with embedded ground truth scenarios for agent validation.
    """

    def __init__(self, seed: int = 42):
        """Initializes the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        self.logger = logging.getLogger(__name__)

        # Configuration for our simulated fleet
        self.vehicles = {
            'TRUCK_001': {'type': 'heavy_truck', 'driver_id': 'driver_01'},
            'TRUCK_002': {'type': 'heavy_truck', 'driver_id': 'driver_02'},
            'VAN_003': {'type': 'delivery_van', 'driver_id': 'driver_03'},
            'VAN_004': {'type': 'delivery_van', 'driver_id': 'driver_04'},
            'CAR_005': {'type': 'sedan', 'driver_id': 'driver_05'}
        }

        # Base parameters for different vehicle types to ensure realism
        self.vehicle_params = {
            'heavy_truck': {'max_speed': 95, 'base_rpm': 800, 'fuel_capacity': 200, 'base_fuel_rate': 20.0},
            'delivery_van': {'max_speed': 110, 'base_rpm': 700, 'fuel_capacity': 80, 'base_fuel_rate': 10.0},
            'sedan': {'max_speed': 130, 'base_rpm': 600, 'fuel_capacity': 60, 'base_fuel_rate': 7.0}
        }

    def generate_trip(self, vehicle_id: str, duration_minutes: int, sample_rate_hz: int = 1) -> pd.DataFrame:
        """Generates telemetry data for a single, continuous trip."""
        num_samples = duration_minutes * 60 * sample_rate_hz
        if num_samples == 0:
            return pd.DataFrame(columns=get_base_telemetry_columns())

        vehicle_info = self.vehicles[vehicle_id]
        params = self.vehicle_params[vehicle_info['type']]

        start_time = datetime.now(timezone.utc) - timedelta(days=random.randint(1, 10))
        timestamps = [start_time + timedelta(seconds=i / sample_rate_hz) for i in range(num_samples)]
        
        # Start with an empty DataFrame with the correct index
        df = pd.DataFrame(index=range(num_samples))
        
        # Fill in static data
        df['timestamp'] = timestamps
        df['vehicle_id'] = vehicle_id
        df['trip_id'] = f"trip_{uuid.uuid4().hex[:8]}"
        df['driver_id'] = vehicle_info['driver_id']
        df['ground_truth_scenario_id'] = 'normal_operation'
        
        # Simulate the core dynamic data
        self._simulate_driving_dynamics(df, params)

        # --- FIX: Ensure all columns from the schema are present ---
        # This loop guarantees our output matches the blueprint exactly.
        for col in get_base_telemetry_columns():
            if col not in df.columns:
                # If a column was not simulated, fill it with a sensible default.
                if TELEMETRY_SCHEMA[col] == 'string':
                    df[col] = ''
                elif TELEMETRY_SCHEMA[col] == 'bool':
                    df[col] = False
                else: # float64, etc.
                    df[col] = 0.0
        
        # Return the DataFrame with columns in the official schema order.
        return df[get_base_telemetry_columns()]


    def _simulate_driving_dynamics(self, df: pd.DataFrame, params: Dict):
        """Fills the DataFrame with realistic, correlated sensor data."""
        num_samples = len(df)
        if num_samples == 0: return

        speed_profile = np.zeros(num_samples)
        cruise_speed = random.uniform(params['max_speed'] * 0.5, params['max_speed'] * 0.9)
        t_accel = int(num_samples * 0.1)
        t_cruise = int(num_samples * 0.8)
        
        if t_accel > 0:
            speed_profile[:t_accel] = np.linspace(0, cruise_speed, t_accel)
        if t_cruise > t_accel:
            speed_profile[t_accel:t_cruise] = cruise_speed + np.random.randn(t_cruise - t_accel) * 5
        if num_samples > t_cruise:
            speed_profile[t_cruise:] = np.linspace(cruise_speed, 0, num_samples - t_cruise)
        
        df['gps_speed'] = np.clip(speed_profile, 0, params['max_speed'])

        df['engine_rpm'] = params['base_rpm'] + (df['gps_speed'] / params['max_speed']) * 1500 + np.random.randn(num_samples) * 50
        df['engine_rpm'] = df['engine_rpm'].clip(lower=params['base_rpm'] - 100)
        df.loc[df['gps_speed'] < 1, 'engine_rpm'] = params['base_rpm'] + np.random.randn(len(df[df['gps_speed'] < 1])) * 20
        
        df['engine_temp'] = 90.0 + np.random.randn(num_samples) * 2.5
        df['engine_load'] = (df['engine_rpm'] / 3000) * 80 + np.random.randn(num_samples) * 5
        df['throttle_position'] = (df['gps_speed'] / params['max_speed']) * 70 + np.random.randn(num_samples) * 5
        
        df['acceleration_x'] = df['gps_speed'].diff().fillna(0) / 3.6
        
        df['battery_voltage'] = 13.8 + np.random.randn(num_samples) * 0.2
        df.loc[df['gps_speed'] < 1, 'battery_voltage'] = 12.6 + np.random.randn(len(df[df['gps_speed'] < 1])) * 0.1
        
        df['fuel_level'] = np.linspace(95, 95 - (len(df) / 3600) * params['base_fuel_rate'], num_samples)
        df['fuel_flow_rate'] = (df['engine_load'] / 100) * params['base_fuel_rate'] + np.random.randn(num_samples) * 0.5
        
        df['dtc_codes'] = ''
        df['mil_status'] = False
        df['is_idling'] = (df['gps_speed'] < 2) & (df['engine_rpm'] > 500)
        df['hard_brake_event'] = df['acceleration_x'] < -3.5
        df['hard_accel_event'] = df['acceleration_x'] > 3.5
        
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    def apply_scenario(self, df: pd.DataFrame, scenario: GroundTruthScenario) -> pd.DataFrame:
        """Modifies a trip DataFrame to inject a ground truth scenario."""
        self.logger.info(f"Applying scenario: {scenario.scenario_id} - {scenario.description}")
        df['ground_truth_scenario_id'] = scenario.scenario_id
        
        for col, mod in scenario.data_modifications.items():
            mod_type, mod_value = mod.split(':', 1)
            
            if mod_type == 'increase_to':
                df[col] = np.linspace(df[col].iloc[0], float(mod_value), len(df))
            elif mod_type == 'decrease_to':
                df[col] = np.linspace(df[col].iloc[0], float(mod_value), len(df))
            elif mod_type == 'set':
                try:
                    final_value = json.loads(mod_value.lower())
                except json.JSONDecodeError:
                    final_value = mod_value
                df[col] = final_value
            elif mod_type == 'inject_events':
                num_events = int(mod_value)
                if num_events > len(df):
                    num_events = len(df)
                indices = np.random.choice(df.index, num_events, replace=False)
                df.loc[indices, col] = True
            elif mod_type == 'increase_by_percent':
                df[col] *= (1 + float(mod_value) / 100)
        
        return df

    def generate_fleet_dataset(self, trips_per_vehicle: int, scenarios_to_inject: List[str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """Generates a full dataset for the entire fleet, including specified scenarios."""
        self.logger.info(f"Starting fleet data generation for {len(self.vehicles)} vehicles...")
        
        all_trips = []
        applied_scenarios_log = []
        
        scenarios = {s.scenario_id: s for s in VALIDATION_SCENARIOS}
        
        if scenarios_to_inject is None:
            scenarios_to_inject = list(scenarios.keys())

        for scenario_id in scenarios_to_inject:
            if not scenarios: continue
            scenario = scenarios.get(scenario_id)
            if not scenario: continue
            
            vehicle_for_scenario = random.choice(list(self.vehicles.keys()))

            trip_df = self.generate_trip(vehicle_for_scenario, duration_minutes=30)
            if not trip_df.empty:
                trip_df = self.apply_scenario(trip_df, scenario)
                all_trips.append(trip_df)
                applied_scenarios_log.append({
                    "trip_id": trip_df['trip_id'].iloc[0],
                    "scenario": asdict(scenario)
                })

        total_trips_to_generate = trips_per_vehicle * len(self.vehicles)
        start_index = len(applied_scenarios_log)
        for _ in range(start_index, total_trips_to_generate):
            vehicle_id = random.choice(list(self.vehicles.keys()))
            trip_df = self.generate_trip(vehicle_id, duration_minutes=random.randint(15, 60))
            if not trip_df.empty:
                all_trips.append(trip_df)

        if not all_trips:
            self.logger.warning("No trips were generated.")
            return pd.DataFrame(columns=get_base_telemetry_columns()), []

        fleet_df = pd.concat(all_trips, ignore_index=True)
        fleet_df = fleet_df.sort_values(by='timestamp').reset_index(drop=True)
        
        for col, dtype in get_validation_dtypes().items():
            if col in fleet_df.columns:
                try:
                    fleet_df[col] = fleet_df[col].astype(dtype)
                except Exception as e:
                    self.logger.error(f"Failed to cast column {col} to {dtype}: {e}")

        self.logger.info(f"Fleet data generation complete. Total records: {len(fleet_df)}")
        return fleet_df, applied_scenarios_log

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("Sentinel Fleet: Synthetic Data Generator")
    print("=" * 60)

    generator = VehicleTelemetryGenerator()
    fleet_data, ground_truth_log = generator.generate_fleet_dataset(
        trips_per_vehicle=5
    )

    output_filename = 'fleet_telemetry_data.parquet'
    ground_truth_filename = 'fleet_ground_truth.json'
    
    if not fleet_data.empty:
        fleet_data.to_parquet(output_filename, index=False, engine='pyarrow')
        print(f"\n✅ Successfully generated and saved dataset to '{output_filename}'")
        print(f"   - Total Records: {len(fleet_data)}")
        print(f"   - Total Trips: {fleet_data['trip_id'].nunique()}")
        print(f"   - Vehicles: {list(fleet_data['vehicle_id'].unique())}")
    
        with open(ground_truth_filename, 'w') as f:
            json.dump(ground_truth_log, f, indent=2)
        print(f"✅ Successfully saved ground truth log to '{ground_truth_filename}'")
        print(f"   - Total Scenarios Injected: {len(ground_truth_log)}")

        print("\n--- Data Generation Complete ---")
        print("You can now run the main orchestrator, which will use this data.")
    else:
        print("\n❌ No data was generated. Please check the generator settings.")
        
    print("=" * 60)