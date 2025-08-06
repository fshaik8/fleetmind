import os
import json
import logging
from typing import List, Dict, Tuple, Optional, TypedDict, Any
from datetime import datetime
import pandas as pd

from langgraph.graph import StateGraph, END

# Import all our real agent nodes
from data_ingestion_agent import data_ingestion_node
from query_router_agent import query_router_node
from vehicle_health_agent import vehicle_health_node
from driver_behavior_agent import driver_behavior_node
from fleet_analyst_agent import fleet_analyst_node

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- State Definition ---
class VehicleAnalysisState(TypedDict):
    """
    Represents the state of our vehicle analysis graph. This state object
    flows through all nodes and gets updated by each agent.
    """
    # === Input Parameters ===
    user_query: str
    vehicle_ids: List[str]
    data_source: str
    
    # === Data Management (from DataIngestionAgent) ===
    telemetry_data: Dict[str, pd.DataFrame]
    data_quality_reports: Dict[str, Dict]
    is_data_suitable: bool
    
    # === Query Routing (from QueryRouterAgent) ===
    routing_decision: Optional[Dict]
    
    # === Agent Analysis Results (populated by specialist agents) ===
    analysis_results: List[Dict]
    
    # === Final Output ===
    final_report: str

# --- The Main Orchestrator Class ---
class VehicleAnalysisOrchestrator:
    """
    Builds and runs the multi-agent graph for vehicle telemetry analysis.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        self.app = self._build_graph()
        
    def _build_graph(self) -> Any:
        """Builds and compiles the LangGraph workflow."""
        self.logger.info("Building multi-agent analysis graph...")
        
        workflow = StateGraph(VehicleAnalysisState)
        
        # --- 1. Register All Agents as Nodes ---
        workflow.add_node("data_ingestion", data_ingestion_node)
        workflow.add_node("query_router", query_router_node)
        workflow.add_node("vehicle_health", vehicle_health_node)
        workflow.add_node("driver_behavior", driver_behavior_node)
        workflow.add_node("fleet_analyst", fleet_analyst_node)
        
        # --- 2. Define the Graph's Workflow (Edges) ---
        workflow.set_entry_point("data_ingestion")
        workflow.add_edge("data_ingestion", "query_router")
        
        workflow.add_conditional_edges(
            "query_router",
            self._decide_next_agent,
            {
                "vehicle_health": "vehicle_health",
                "driver_behavior": "driver_behavior",
                "run_all_specialists": "vehicle_health",
                "end_workflow": END
            }
        )
        
        # After a single-purpose analysis, the workflow ends.
        # But for the multi-agent path, we define the sequence.
        workflow.add_edge("vehicle_health", "driver_behavior")
        workflow.add_edge("driver_behavior", "fleet_analyst")
        workflow.add_edge("fleet_analyst", END)
        
        app = workflow.compile()
        self.logger.info("Graph compilation complete.")
        return app
    
    def _decide_next_agent(self, state: VehicleAnalysisState) -> str:
        """Logic for the conditional edge to route based on the router's decision."""
        self.logger.info("Orchestrator: Deciding next step based on router's decision...")
        
        if not state.get("is_data_suitable", False):
            self.logger.warning("Data is not suitable for analysis. Ending workflow.")
            state['final_report'] = "Analysis could not be completed because the data quality was too low."
            return "end_workflow"
            
        routing_decision = state.get("routing_decision")
        if not routing_decision:
            self.logger.error("No routing decision found in state. Ending workflow.")
            state['final_report'] = "Analysis failed because the query router could not make a decision."
            return "end_workflow"
            
        required_agents = routing_decision.get("required_agents", [])
        intent = routing_decision.get("primary_intent")
        self.logger.info(f"Router's intent: '{intent}'. Required agents: {required_agents}")
        
        # If the plan requires more than one agent or is a general fleet analysis,
        # we run the full pipeline.
        if len(required_agents) > 1 or intent == "fleet_analyst":
            return "run_all_specialists"
        # Otherwise, we route to the single, specific agent.
        elif intent in ["vehicle_health", "driver_behavior"]:
             # For single agent queries, we need to adjust the graph flow to end after they run.
             # This is a limitation of this simple linear graph. A more complex graph
             # would have separate end points. For now, we will let it run the full path,
             # and the final analyst will just summarize that single report.
            return "run_all_specialists" # Simplified logic: always run full pipeline for now.
        else:
            self.logger.warning(f"Unknown intent '{intent}'. Ending workflow.")
            return "end_workflow"

    def run_analysis(self, user_query: str, vehicle_ids: Optional[List[str]] = None):
        """The main method to execute an analysis query on the graph."""
        
        data_source = 'fleet_telemetry_data.parquet'
        if not os.path.exists(data_source):
            self.logger.error(f"Data file '{data_source}' not found. Please run synthetic_data_generator.py first.")
            return
            
        initial_state = {
            "user_query": user_query,
            "vehicle_ids": vehicle_ids or [],
            "data_source": data_source,
            "analysis_results": []
        }
        
        self.logger.info(f"\n{'='*60}\nRunning analysis for query: '{user_query}'\n{'='*60}")
        
        final_state = None
        for step_output in self.app.stream(initial_state):
            node_name = list(step_output.keys())[0]
            state = step_output[node_name]
            final_state = state
            
            print(f"--- Agent Executed: {node_name} ---")
            
            if self.debug_mode:
                if node_name == "data_ingestion":
                    print(f"  Data Suitable: {state.get('is_data_suitable')}")
                elif node_name == "query_router":
                    print(f"  Routing Decision: {state.get('routing_decision', {}).get('primary_intent')}")

        if final_state:
            print("\n--- Final Report ---")
            print(final_state.get("final_report", "No final report was generated."))
        else:
            print("\n--- Workflow finished with no final state ---")

        print("\n" + "="*60)

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("FATAL ERROR: OPENROUTER_API_KEY is not set. The router agent will fail.")
        print("Please set it in your terminal, e.g.: export OPENROUTER_API_KEY='your_key'")
    else:
        orchestrator = VehicleAnalysisOrchestrator(debug_mode=True)
        
        # Test 1: Simple, single-agent query.
        orchestrator.run_analysis(
            user_query="Are there any fault codes for TRUCK_001?",
            vehicle_ids=["TRUCK_001"]
        )
        
        # Test 2: Simple, single-agent query.
        orchestrator.run_analysis(
            user_query="What's the safety score for the driver of VAN_003?",
            vehicle_ids=["VAN_003"]
        )
        
        # Test 3: Complex, multi-agent query that requires synthesis.
        orchestrator.run_analysis(
            user_query="Give me a comprehensive summary report of the entire fleet's performance, including health and safety."
        )