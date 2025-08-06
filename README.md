# FleetMind: A Multi-Agent System for Conversational Telemetry Analysis

A multi-agent AI system built with LangGraph for conversational analysis of vehicle telemetry data.

**FleetMind** is a complete, end-to-end multi-agent AI system designed to analyze complex vehicle telemetry data through natural language. Instead of relying on traditional dashboards and manual queries, this project allows a user to ask plain-English questions about a vehicle fleet's health, safety, and performance, and receive a synthesized, intelligent answer.

The system is architected using **LangGraph** to orchestrate a team of specialized AI agents, each with a distinct role, from data validation to intelligent routing and final report synthesis. This project serves as a powerful demonstration of modern agentic AI workflows, data validation, and resilient system design.

---

## Key Features

- **Conversational Interface:** Ask complex, multi-domain questions in natural language (e.g., "Which of my drivers are driving safely but have vehicles that need maintenance?").
- **Multi-Agent Architecture:** A team of specialized agents (`DataIngestion`, `QueryRouter`, `VehicleHealth`, `DriverBehavior`, `FleetAnalyst`) collaborate to solve problems, orchestrated by LangGraph.
- **Intelligent LLM-Powered Routing:** A `QueryRouterAgent` uses an open-source LLM (via OpenRouter) to semantically understand user intent and dispatch tasks to the appropriate specialist(s).
- **Rule-Based Specialist Analysis:** Specialist agents use robust, rule-based logic to perform accurate, data-driven analysis of vehicle health and driver safety.
- **AI-Powered Synthesis:** A final `FleetAnalystAgent` uses an LLM to synthesize the structured reports from multiple specialist agents into a single, coherent executive summary.
- **Synthetic Data & Ground-Truth Validation:** The project includes a comprehensive synthetic data generator that can inject pre-defined fault scenarios, allowing for rigorous, quantitative testing of the agents' analytical accuracy.
- **Resilient Design:** The system features a rule-based fallback for the LLM router, ensuring graceful degradation and high availability during external API failures.

---

## System Architecture

The project follows a stateful, graph-based workflow managed by LangGraph. A query progresses through the system as follows:

1.  **Data Ingestion:** The `DataIngestionAgent` loads and validates the specified vehicle's telemetry data, ensuring its quality.
2.  **Query Routing:** The `QueryRouterAgent` analyzes the user's query and the available data, using an LLM to make an intelligent decision about which specialist agent(s) to activate.
3.  **Specialist Analysis Pipeline:** Based on the router's decision, the orchestrator executes a sequence of specialist agents. For complex queries, it runs a full pipeline:
    - **`VehicleHealthAgent`:** Analyzes the data for fault codes and sensor anomalies.
    - **`DriverBehaviorAgent`:** Analyzes the data for safety events and calculates a performance score.
4.  **Synthesis:** The `FleetAnalystAgent` receives the structured reports from the specialists and uses an LLM to write the final, human-readable report.

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- An API key from [OpenRouter.ai](https://openrouter.ai/) (for LLM-powered routing and synthesis)
- The required Python packages:
  ```bash
  pip install pandas pyarrow openai "langgraph"
  ```

### 2. Configuration

Set your OpenRouter API key as an environment variable. This is crucial for the intelligent agents to function.

**On Linux/macOS:**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

**On Windows (Command Prompt):**
```bash
set OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 3. Generate the Dataset

Before running the main application, you must first generate the synthetic telemetry dataset. This script creates the `fleet_telemetry_data.parquet` file that the system uses.

```bash
python synthetic_data_generator.py
```

### 4. Run the Main Orchestrator

Execute the main orchestrator file to run the pre-defined test queries. This will demonstrate the full, end-to-end multi-agent workflow.

```bash
python graph_orchestrator.py
```
You will see the step-by-step execution as the system processes each query, from data ingestion to routing and final analysis, printing a detailed report for each.

---

## Project Files

- **`vehicle_telemetry_schema.py`**: The **Blueprint**; defines the data structure and ground-truth test scenarios.
- **`synthetic_data_generator.py`**: The **Data Factory**; creates the realistic test dataset with injected faults.
- **`data_ingestion_agent.py`**: Agent 1: The **Quality Inspector** for data validation.
- **`query_router_agent.py`**: Agent 2: The **Dispatcher** for intelligent, LLM-powered routing.
- **`vehicle_health_agent.py`**: Agent 3: The **Maintenance Specialist** for diagnostics.
- **`driver_behavior_agent.py`**: Agent 4: The **Safety Coach** for performance analysis.
- **`fleet_analyst_agent.py`**: Agent 5: The **Chief Analyst** for final report synthesis.
- **`graph_orchestrator.py`**: The **Conductor**; the main application that builds and runs the LangGraph workflow.
