#
# Sentinel Fleet: A Multi-Agent System for Conversational Telemetry Analysis
#
# query_router_agent.py
#
# ==============================================================================
#
# This module contains the QueryRouterAgent, the "brain" or "dispatcher" of the
# system. It uses a Large Language Model (LLM) to perform intent classification
# on the user's query, determining which specialist agent is best suited to
# handle the request. It's the core of the system's intelligent routing.
#
# ==============================================================================

import os
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

from openai import OpenAI
import pandas as pd

@dataclass
class RoutingDecision:
    """A structured output representing the router's decision."""
    primary_intent: str
    required_agents: List[str]
    analysis_scope: str
    routing_confidence: float
    routing_rationale: str

def clean_response_text(text: str) -> str:
    """Remove triple-back-tick fences if the model wrapped its reply."""
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text

def extract_json_from_text(text: str) -> Optional[str]:
    """Return the first JSON object found in the text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


class QueryRouterAgent:
    """Agent that uses an LLM to classify and route user queries."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initializes the QueryRouterAgent."""
        self.logger = logger or logging.getLogger(__name__)
        self.llm_client = self._initialize_llm()

        self.agent_specialties = {
            'vehicle_health': "Analyzes engine diagnostics, fault codes, maintenance needs, and system health.",
            'driver_behavior': "Evaluates driver performance, safety scores, and driving patterns like hard braking.",
            'fleet_analyst': "Provides fleet-wide insights, comparisons between vehicles, and summary reports."
        }
        self.system_prompt = self._create_system_prompt()

    def _initialize_llm(self) -> Optional[OpenAI]:
        """Initializes the LLM client, configured for OpenRouter.ai."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.logger.warning("OPENROUTER_API_KEY environment variable not set. LLM routing will be disabled.")
            return None
        
        try:
            # --- FINAL POLISH --- Added a 20-second timeout to the client.
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Sentinel Fleet"},
                timeout=20.0,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            return None

    def _create_system_prompt(self) -> str:
        """Creates the structured prompt to guide the LLM's routing decision."""
        return f"""You are an expert query routing agent for a vehicle fleet analysis system. Your job is to analyze a user's query and decide which specialist agent is best suited to answer it.

You must respond ONLY with a single, valid JSON object that strictly follows this format:
{{
    "primary_intent": "The single best matching agent specialty.",
    "required_agents": ["A list containing the names of the required agents."],
    "analysis_scope": "Determine if the query is about a 'single_vehicle', 'multi_vehicle', or 'fleet_wide'.",
    "routing_confidence": "A confidence score from 0.0 to 1.0.",
    "routing_rationale": "A brief explanation for your decision."
}}

Here are the available specialist agents and their capabilities:
{json.dumps(self.agent_specialties, indent=2)}

Analyze the user's query and available data to make the best routing decision. If the query is complex and involves comparison or multiple aspects, you can require multiple agents. If the query is about comparing vehicles, the primary intent should be 'fleet_analyst'.
"""

    def run(self, state: Dict) -> Dict:
        """Main entry point for the agent's execution within LangGraph."""
        self.logger.info("QueryRouterAgent: Analyzing user query for routing...")
        
        user_query = state.get("user_query")
        telemetry_data = state.get("telemetry_data", {})

        if not user_query:
            self.logger.error("No user query found in state.")
            decision = RoutingDecision(
                primary_intent='fleet_analyst', required_agents=['fleet_analyst'],
                analysis_scope='fleet_wide', routing_confidence=0.0,
                routing_rationale="Error: No query provided."
            )
        elif not self.llm_client:
            decision = self._rule_based_fallback_routing(user_query)
        else:
            try:
                human_prompt = f"User Query: \"{user_query}\"\nAvailable Vehicles: {list(telemetry_data.keys())}"
                
                model_to_use = "deepseek/deepseek-chat-v3-0324:free"
                
                response = self.llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": human_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                    raise ValueError("LLM returned an empty response.")

                raw_reply = response.choices[0].message.content
                self.logger.info(f"LLM raw response: {raw_reply}")

                json_str = extract_json_from_text(clean_response_text(raw_reply))
                if not json_str:
                    raise ValueError("No valid JSON object found in the LLM response.")

                decision_json = json.loads(json_str)
                decision = RoutingDecision(**decision_json)
                self.logger.info(f"LLM routing decision: {decision.primary_intent} with confidence {decision.routing_confidence:.2f}")

            except Exception as e:
                self.logger.error(f"LLM routing failed: {e}. Falling back to rule-based method.", exc_info=False)
                decision = self._rule_based_fallback_routing(user_query)

        state['routing_decision'] = asdict(decision)
        return state

    def _rule_based_fallback_routing(self, user_query: str) -> RoutingDecision:
        """A simple, keyword-based routing method for when the LLM fails."""
        self.logger.info("Using rule-based fallback for routing.")
        query_lower = user_query.lower()
        
        if any(kw in query_lower for kw in ['engine', 'fault', 'maintenance', 'dtc', 'health']):
            intent = 'vehicle_health'
        elif any(kw in query_lower for kw in ['driver', 'safety', 'braking', 'speeding']):
            intent = 'driver_behavior'
        else:
            intent = 'fleet_analyst'
            
        return RoutingDecision(
            primary_intent=intent,
            required_agents=[intent],
            analysis_scope='fleet_wide',
            routing_confidence=0.5,
            routing_rationale="Fallback rule-based routing was used."
        )

def query_router_node(state: Dict) -> Dict:
    """LangGraph node function for the Query Router Agent."""
    agent = QueryRouterAgent()
    return agent.run(state)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("Sentinel Fleet: Standalone Test for QueryRouterAgent")
    print("=" * 60)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY is not set. The agent will use its rule-based fallback.")
    
    test_queries = [
        "Check TRUCK_001 for any engine overheating or fault codes.",
        "How is the driver of VAN_003 behaving? Is he braking too hard?",
        "Give me a full summary report of the entire fleet's performance.",
        "Compare the health of TRUCK_001 and TRUCK_002."
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Query ---")
        print(f"'{query}'")
        
        mock_state = {
            "user_query": query,
            "telemetry_data": {
                "TRUCK_001": pd.DataFrame(),
                "TRUCK_002": pd.DataFrame(),
                "VAN_003": pd.DataFrame()
            }
        }
        
        final_state = query_router_node(mock_state)
        decision = final_state.get('routing_decision', {})
        
        print("\n--- Routing Decision ---")
        print(f"  Intent: {decision.get('primary_intent')}")
        print(f"  Required Agents: {decision.get('required_agents')}")
        print(f"  Confidence: {decision.get('routing_confidence')}")
        print(f"  Rationale: {decision.get('routing_rationale')}")
        
    print("\n" + "=" * 60)