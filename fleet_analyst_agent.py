import os
import json
import logging
import pandas as pd
# --- FIX --- Added the missing import for dataclass and asdict.
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# This agent uses an LLM for synthesis, similar to the router.
from openai import OpenAI

@dataclass
class FleetSummary:
    """A structured summary of the entire fleet's status."""
    total_vehicles_analyzed: int
    vehicles_with_critical_issues: int
    vehicles_with_warnings: int
    drivers_needing_coaching: int
    fleet_average_safety_score: float

class FleetAnalystAgent:
    """
    Specialist agent for synthesizing reports and providing fleet-wide insights.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initializes the FleetAnalystAgent."""
        self.logger = logger or logging.getLogger(__name__)
        self.llm_client = self._initialize_llm()

    def _initialize_llm(self) -> Optional[OpenAI]:
        """Initializes the LLM client, configured for OpenRouter.ai."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.logger.warning("OPENROUTER_API_KEY not set. LLM synthesis will be disabled.")
            return None
        
        try:
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Sentinel Fleet"},
                timeout=30.0, # Allow a longer timeout for summary generation
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            return None

    def run(self, state: Dict) -> Dict:
        """Main entry point for the agent's execution within LangGraph."""
        self.logger.info("FleetAnalystAgent: Starting fleet-wide analysis and synthesis...")

        user_query = state.get("user_query", "Generate a fleet summary.")
        analysis_results = state.get("analysis_results", [])
        
        summary_context = self._create_summary_context(analysis_results)
        
        if self.llm_client:
            final_report = self._generate_llm_summary(user_query, summary_context)
        else:
            self.logger.warning("LLM not available. Generating a basic, rule-based summary.")
            final_report = self._generate_rule_based_summary(summary_context)

        state['final_report'] = final_report
        self.logger.info("Fleet analysis and synthesis complete.")
        return state

    def _create_summary_context(self, analysis_results: List[Dict]) -> str:
        """Formats the structured reports from other agents into a text block."""
        if not analysis_results:
            return "No prior analysis was performed by specialist agents."

        context_parts = ["--- CONSOLIDATED ANALYSIS DATA ---\n"]
        for result in analysis_results:
            context_parts.append(json.dumps(result, indent=2))
        
        return "\n".join(context_parts)

    def _generate_llm_summary(self, user_query: str, context: str) -> str:
        """Uses an LLM to write a high-quality, natural language summary."""
        
        system_prompt = """You are an expert fleet analyst AI. Your task is to synthesize a series of structured JSON reports into a single, easy-to-understand executive summary for a fleet manager.
        
- Begin with a direct, top-level answer to the user's original query.
- Use clear, professional language. Do not use jargon.
- Organize the information logically with headings or bullet points.
- Highlight the most critical findings and provide actionable recommendations.
- Do not just repeat the JSON data; interpret it and explain its significance."""

        human_prompt = f"""
Original User Query: "{user_query}"

Please synthesize the following consolidated data into a final report:
{context}
"""
        try:
            model_to_use = "deepseek/deepseek-chat"
            
            response = self.llm_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt},
                ],
                temperature=0.5,
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("LLM returned an empty response.")

        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {e}. Falling back to basic summary.")
            return self._generate_rule_based_summary(context)

    def _generate_rule_based_summary(self, context: str) -> str:
        """A simple fallback to format the context if the LLM fails."""
        report_lines = ["--- Fleet Analysis Summary ---\n"]
        report_lines.append("The following is a summary of the analysis performed:\n")
        report_lines.append(context)
        report_lines.append("\n(Note: This is a basic summary as the AI synthesis model was unavailable.)")
        return "\n".join(report_lines)

def fleet_analyst_node(state: Dict) -> Dict:
    """LangGraph node function for the Fleet Analyst Agent."""
    agent = FleetAnalystAgent()
    return agent.run(state)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("Sentinel Fleet: Standalone Test for FleetAnalystAgent")
    print("=" * 60)
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY is not set. The agent will use its rule-based fallback.")

    mock_state = {
        "user_query": "Give me a summary of my fleet's status.",
        "analysis_results": [
            {
                "agent": "VehicleHealthAgent",
                "vehicle_id": "TRUCK_001",
                "overall_status": "Critical Issues Detected",
                "findings": [
                    {"check_name": "DTC Check", "status": "Critical", "details": "Active fault codes detected: P0217"},
                ]
            },
            {
                "agent": "DriverBehaviorAgent",
                "driver_id": "driver_03",
                "overall_assessment": "Aggressive Driving Pattern Detected",
                "findings": [
                    {"event_type": "hard_accel_event", "event_count": 22, "severity": "High"},
                ]
            },
             {
                "agent": "VehicleHealthAgent",
                "vehicle_id": "VAN_004",
                "overall_status": "All Systems Normal",
                "findings": []
            }
        ]
    }

    print("--- Running agent on mock multi-agent data ---")
    final_state = fleet_analyst_node(mock_state)

    print("\n--- Agent's Final Synthesized Report ---")
    print(final_state.get('final_report'))
    print("\n" + "=" * 60)