"""
Agentic Orchestrator for Layer Intelligence System.

The orchestrator is the main "brain" that coordinates all components:
- Layer discovery and semantic search
- Query parsing and execution
- Knowledge graph reasoning
- Response generation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import google.generativeai as genai
from decouple import config as env_config

from .models import (
    AgentAction,
    AgentResponse,
    AgentStep,
    LayerSearchResult,
    QueryInsight,
    QueryPlan,
    StructuredQuery,
)
from .layer_catalog import LayerCatalogService, get_layer_catalog_service_sync
from .knowledge_graph import KnowledgeGraphService, get_knowledge_graph_service_sync
from .self_query import SelfQueryRetriever, QueryExecutor, get_self_query_retriever

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context maintained across conversation turns."""
    session_id: str = ""
    location: Optional[dict] = None  # Current geographic focus
    layers_discussed: list[str] = field(default_factory=list)
    recent_queries: list[str] = field(default_factory=list)
    insights_gathered: list[QueryInsight] = field(default_factory=list)
    preferences: dict = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_steps: int = 10
    timeout_seconds: int = 60
    confidence_threshold: float = 0.7
    enable_learning: bool = True
    verbose: bool = False


class AgenticOrchestrator:
    """
    Main agentic orchestrator that processes queries autonomously.

    The orchestrator follows a ReAct-style approach:
    1. Reason about what action to take
    2. Execute the action
    3. Observe the results
    4. Repeat until answer is ready
    """

    def __init__(
        self,
        catalog: Optional[LayerCatalogService] = None,
        graph: Optional[KnowledgeGraphService] = None,
        retriever: Optional[SelfQueryRetriever] = None,
        config: Optional[OrchestratorConfig] = None,
        llm_model: str = "gemini-2.0-flash",
    ):
        self.catalog = catalog or get_layer_catalog_service_sync()
        self.graph = graph or get_knowledge_graph_service_sync()
        self.retriever = retriever or get_self_query_retriever()
        self.config = config or OrchestratorConfig()
        self.llm_model = llm_model

        # Initialize Gemini
        api_key = env_config("GOOGLE_API_KEY", default="")
        if api_key:
            genai.configure(api_key=api_key)

        self._model = None
        self._executor = None

        # Action handlers
        self._action_handlers: dict[AgentAction, Callable] = {
            AgentAction.DISCOVER_LAYERS: self._action_discover_layers,
            AgentAction.QUERY_LAYER: self._action_query_layer,
            AgentAction.ANALYZE_RESULTS: self._action_analyze_results,
            AgentAction.CROSS_REFERENCE: self._action_cross_reference,
            AgentAction.GENERATE_INSIGHT: self._action_generate_insight,
            AgentAction.STORE_LEARNING: self._action_store_learning,
            AgentAction.ASK_CLARIFICATION: self._action_ask_clarification,
        }

    @property
    def model(self):
        """Lazy load the Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(self.llm_model)
        return self._model

    @property
    def executor(self):
        """Lazy load the query executor."""
        if self._executor is None:
            token = env_config("ARCGIS_API_KEY", default="")
            self._executor = QueryExecutor(arcgis_token=token if token else None)
        return self._executor

    async def process(
        self,
        query: str,
        context: Optional[ConversationContext] = None,
    ) -> AgentResponse:
        """
        Process a natural language query and return a response.

        Args:
            query: User's natural language query
            context: Optional conversation context

        Returns:
            AgentResponse with answer and metadata
        """
        start_time = time.time()
        context = context or ConversationContext()
        steps: list[AgentStep] = []
        working_memory: dict = {
            "original_query": query,
            "layers_found": [],
            "query_results": [],
            "insights": [],
        }

        try:
            # Initial planning
            initial_plan = await self._plan_initial_actions(query, context)
            working_memory["initial_plan"] = initial_plan

            # Execute action loop
            for step_num in range(self.config.max_steps):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout_seconds:
                    logger.warning("Orchestrator timeout reached")
                    break

                # Decide next action
                action, reasoning, inputs = await self._decide_next_action(
                    query, context, working_memory, steps
                )

                if action is None:
                    # No more actions needed
                    break

                step_start = time.time()

                # Execute action
                try:
                    handler = self._action_handlers.get(action)
                    if handler:
                        outputs = await handler(inputs, working_memory, context)
                        success = True
                        error = None
                    else:
                        outputs = None
                        success = False
                        error = f"Unknown action: {action}"

                except Exception as e:
                    logger.error(f"Action {action} failed: {e}")
                    outputs = None
                    success = False
                    error = str(e)

                step_duration = int((time.time() - step_start) * 1000)

                # Record step
                steps.append(AgentStep(
                    action=action,
                    reasoning=reasoning,
                    inputs=inputs,
                    outputs=outputs,
                    success=success,
                    error=error,
                    duration_ms=step_duration,
                ))

                if self.config.verbose:
                    logger.info(f"Step {step_num + 1}: {action.value} - {'OK' if success else 'FAIL'}")

            # Generate final answer
            answer = await self._generate_answer(query, working_memory, steps)

            # Calculate confidence
            confidence = self._calculate_confidence(steps, working_memory)

            # Generate suggestions
            suggestions = await self._generate_suggestions(query, working_memory)

            execution_time = int((time.time() - start_time) * 1000)

            return AgentResponse(
                answer=answer,
                steps=steps,
                layers_used=working_memory.get("layers_found", []),
                confidence=confidence,
                suggestions=suggestions,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            execution_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                answer=f"I encountered an error processing your query: {str(e)}",
                steps=steps,
                layers_used=[],
                confidence=0.0,
                suggestions=["Try rephrasing your question", "Check if the layer data is available"],
                execution_time_ms=execution_time,
            )

    async def _plan_initial_actions(
        self,
        query: str,
        context: ConversationContext,
    ) -> dict:
        """Create initial action plan based on query."""
        prompt = f"""Analyze this GIS/real estate query and create an action plan.

QUERY: {query}

CONTEXT:
- Recent layers discussed: {context.layers_discussed[-3:] if context.layers_discussed else 'None'}
- Location focus: {context.location or 'Not specified'}

Determine the query type and required actions:

1. DATA_RETRIEVAL: "Show me...", "List...", "What are..."
   Actions: discover_layers -> query_layer -> generate_insight

2. STATISTICAL: "What is the average...", "Total...", "Count..."
   Actions: discover_layers -> query_layer (with stats) -> analyze_results -> generate_insight

3. COMPARISON: "Compare...", "Difference between...", "How does X compare to Y"
   Actions: discover_layers -> query_layer (multiple) -> cross_reference -> generate_insight

4. SPATIAL: "Near...", "Within...", "In this area..."
   Actions: discover_layers -> query_layer (with geometry) -> generate_insight

5. TREND: "Over time...", "Change in...", "Historical..."
   Actions: discover_layers -> query_layer (time series) -> analyze_results -> generate_insight

Return JSON:
{{
    "query_type": "DATA_RETRIEVAL|STATISTICAL|COMPARISON|SPATIAL|TREND",
    "actions": ["action1", "action2", ...],
    "estimated_layers_needed": 1-3,
    "needs_clarification": true/false,
    "clarification_question": "optional question if needed"
}}

Return ONLY valid JSON."""

        try:
            response = await self.model.generate_content_async(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Initial planning error: {e}")
            return {
                "query_type": "DATA_RETRIEVAL",
                "actions": ["discover_layers", "query_layer", "generate_insight"],
                "estimated_layers_needed": 1,
                "needs_clarification": False,
            }

    async def _decide_next_action(
        self,
        query: str,
        context: ConversationContext,
        working_memory: dict,
        previous_steps: list[AgentStep],
    ) -> tuple[Optional[AgentAction], str, dict]:
        """Decide the next action to take based on current state."""

        # Simple rule-based decision for efficiency
        # (Can be replaced with LLM-based reasoning for more complex cases)

        completed_actions = {s.action for s in previous_steps if s.success}
        failed_actions = {s.action for s in previous_steps if not s.success}

        # Step 1: Discover layers if not done
        if AgentAction.DISCOVER_LAYERS not in completed_actions:
            return (
                AgentAction.DISCOVER_LAYERS,
                "First, find relevant layers for the query",
                {"query": query},
            )

        # Step 2: Query layers if discovered
        if (AgentAction.DISCOVER_LAYERS in completed_actions and
            AgentAction.QUERY_LAYER not in completed_actions and
            working_memory.get("layers_found")):
            return (
                AgentAction.QUERY_LAYER,
                "Execute queries on discovered layers",
                {"query": query},
            )

        # Step 3: Analyze results if we have them
        if (AgentAction.QUERY_LAYER in completed_actions and
            working_memory.get("query_results") and
            AgentAction.ANALYZE_RESULTS not in completed_actions):

            plan = working_memory.get("initial_plan", {})
            if plan.get("query_type") in ["STATISTICAL", "TREND"]:
                return (
                    AgentAction.ANALYZE_RESULTS,
                    "Analyze the query results for insights",
                    {},
                )

        # Step 4: Cross-reference for comparisons
        if (working_memory.get("query_results") and
            len(working_memory.get("layers_found", [])) > 1 and
            AgentAction.CROSS_REFERENCE not in completed_actions):

            plan = working_memory.get("initial_plan", {})
            if plan.get("query_type") == "COMPARISON":
                return (
                    AgentAction.CROSS_REFERENCE,
                    "Cross-reference data between layers",
                    {},
                )

        # Step 5: Generate insight (final step)
        if (AgentAction.QUERY_LAYER in completed_actions and
            AgentAction.GENERATE_INSIGHT not in completed_actions):
            return (
                AgentAction.GENERATE_INSIGHT,
                "Generate final insight from results",
                {},
            )

        # Done
        return None, "", {}

    # ==========================================================================
    # Action Handlers
    # ==========================================================================

    async def _action_discover_layers(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Discover relevant layers for the query."""
        query = inputs.get("query", working_memory["original_query"])

        # Search layers
        results = await self.catalog.search_layers(query, limit=5)

        # Store in working memory
        layer_names = [r.layer.name for r in results]
        working_memory["layers_found"] = layer_names
        working_memory["layer_results"] = [
            {
                "name": r.layer.name,
                "display_name": r.layer.display_name,
                "description": r.layer.description,
                "score": r.similarity_score,
                "url": r.layer.layer_url,
            }
            for r in results
        ]

        # Also check knowledge graph for related layers
        if layer_names:
            suggested = await self.graph.suggest_analysis_layers(
                query, layer_names[:1]
            )
            if suggested:
                for s in suggested:
                    if s not in layer_names:
                        layer_names.append(s)
                        working_memory["layers_found"] = layer_names

        return {
            "layers_found": len(layer_names),
            "top_layer": layer_names[0] if layer_names else None,
            "scores": [r.similarity_score for r in results],
        }

    async def _action_query_layer(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Execute queries on discovered layers."""
        query = inputs.get("query", working_memory["original_query"])

        # Parse query to structured format
        plan = await self.retriever.parse_query(
            query,
            context={"location": context.location} if context.location else None,
        )

        if not plan.queries:
            return {"error": "Could not generate query plan", "features_returned": 0}

        working_memory["query_plan"] = {
            "reasoning": plan.reasoning,
            "query_count": len(plan.queries),
        }

        # Execute queries
        all_results = []
        for structured_query in plan.queries:
            try:
                result = await self.executor.execute(structured_query)
                all_results.append({
                    "layer": structured_query.layer_name,
                    "success": True,
                    "feature_count": len(result.get("features", [])),
                    "data": result,
                })
            except Exception as e:
                all_results.append({
                    "layer": structured_query.layer_name,
                    "success": False,
                    "error": str(e),
                })

        working_memory["query_results"] = all_results

        total_features = sum(
            r.get("feature_count", 0) for r in all_results if r.get("success")
        )

        return {
            "queries_executed": len(all_results),
            "successful": sum(1 for r in all_results if r.get("success")),
            "total_features": total_features,
        }

    async def _action_analyze_results(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Analyze query results for patterns and statistics."""
        results = working_memory.get("query_results", [])

        if not results:
            return {"analysis": "No results to analyze"}

        # Gather data for analysis
        data_summary = []
        for result in results:
            if result.get("success") and result.get("data"):
                features = result["data"].get("features", [])
                if features:
                    # Extract attributes
                    sample_attrs = features[0].get("attributes", {})
                    numeric_fields = [
                        k for k, v in sample_attrs.items()
                        if isinstance(v, (int, float)) and k.lower() not in ["objectid", "fid"]
                    ]

                    # Calculate basic stats
                    stats = {}
                    for field in numeric_fields[:5]:
                        values = [
                            f.get("attributes", {}).get(field)
                            for f in features
                            if f.get("attributes", {}).get(field) is not None
                        ]
                        if values:
                            stats[field] = {
                                "min": min(values),
                                "max": max(values),
                                "avg": sum(values) / len(values),
                                "count": len(values),
                            }

                    data_summary.append({
                        "layer": result["layer"],
                        "feature_count": len(features),
                        "statistics": stats,
                    })

        working_memory["analysis"] = data_summary

        return {
            "layers_analyzed": len(data_summary),
            "has_statistics": any(s.get("statistics") for s in data_summary),
        }

    async def _action_cross_reference(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Cross-reference data between multiple layers."""
        results = working_memory.get("query_results", [])
        successful_results = [r for r in results if r.get("success")]

        if len(successful_results) < 2:
            return {"cross_reference": "Need at least 2 layers for comparison"}

        # Use knowledge graph to understand relationships
        layer_names = [r["layer"] for r in successful_results]
        relationships = []

        for i, name1 in enumerate(layer_names):
            for name2 in layer_names[i+1:]:
                path = await self.graph.find_cross_layer_path(name1, name2)
                if path:
                    relationships.append({
                        "layer1": name1,
                        "layer2": name2,
                        "relationship": path.explanation,
                    })

        working_memory["cross_references"] = relationships

        return {
            "relationships_found": len(relationships),
            "layers_compared": len(layer_names),
        }

    async def _action_generate_insight(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Generate insights from all gathered data."""
        query = working_memory["original_query"]
        results = working_memory.get("query_results", [])
        analysis = working_memory.get("analysis", [])
        cross_refs = working_memory.get("cross_references", [])

        # Prepare data summary for LLM
        data_context = []
        for result in results:
            if result.get("success") and result.get("data"):
                features = result["data"].get("features", [])
                sample = features[:3] if features else []
                data_context.append({
                    "layer": result["layer"],
                    "feature_count": len(features),
                    "sample_data": [f.get("attributes") for f in sample],
                })

        prompt = f"""Generate a comprehensive insight based on the GIS data analysis.

ORIGINAL QUERY: {query}

DATA RETRIEVED:
{json.dumps(data_context, indent=2)}

STATISTICAL ANALYSIS:
{json.dumps(analysis, indent=2) if analysis else 'None'}

CROSS-REFERENCES:
{json.dumps(cross_refs, indent=2) if cross_refs else 'None'}

Generate a clear, informative response that:
1. Directly answers the user's question
2. Includes specific data points and numbers
3. Highlights key patterns or trends
4. Provides geographic context when relevant
5. Suggests related insights they might find useful

Format the response in a conversational but professional tone.
Use bullet points for lists of data.
Include relevant numbers with proper formatting."""

        try:
            response = await self.model.generate_content_async(prompt)
            insight = response.text.strip()
            working_memory["final_insight"] = insight
            return {"insight_generated": True, "length": len(insight)}
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return {"insight_generated": False, "error": str(e)}

    async def _action_store_learning(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Store learned patterns for future queries."""
        if not self.config.enable_learning:
            return {"learning_stored": False}

        # Create insight record
        insight = QueryInsight(
            query=working_memory["original_query"],
            intent=working_memory.get("initial_plan", {}).get("query_type", ""),
            layers_used=working_memory.get("layers_found", []),
            was_successful=bool(working_memory.get("final_insight")),
        )

        # Store in context
        context.insights_gathered.append(insight)

        return {"learning_stored": True, "insight_id": insight.id}

    async def _action_ask_clarification(
        self,
        inputs: dict,
        working_memory: dict,
        context: ConversationContext,
    ) -> dict:
        """Ask user for clarification when needed."""
        question = inputs.get("question", "Could you please provide more details?")
        working_memory["clarification_needed"] = question
        return {"clarification_asked": True, "question": question}

    # ==========================================================================
    # Response Generation
    # ==========================================================================

    async def _generate_answer(
        self,
        query: str,
        working_memory: dict,
        steps: list[AgentStep],
    ) -> str:
        """Generate the final answer from working memory."""
        # Check if we have a generated insight
        if working_memory.get("final_insight"):
            return working_memory["final_insight"]

        # Check if clarification is needed
        if working_memory.get("clarification_needed"):
            return working_memory["clarification_needed"]

        # Fallback: summarize what we found
        layers = working_memory.get("layers_found", [])
        results = working_memory.get("query_results", [])

        if not layers:
            return "I couldn't find any relevant data layers for your query. Could you rephrase your question or provide more details about what type of data you're looking for?"

        if not results or not any(r.get("success") for r in results):
            return f"I found relevant layers ({', '.join(layers[:3])}) but encountered issues retrieving the data. Please try again or check if the data is accessible."

        # Basic summary
        total_features = sum(
            r.get("feature_count", 0) for r in results if r.get("success")
        )

        return f"Found {total_features} features across {len(results)} layer(s): {', '.join(layers[:3])}. Please refine your query for more specific insights."

    def _calculate_confidence(
        self,
        steps: list[AgentStep],
        working_memory: dict,
    ) -> float:
        """Calculate confidence score for the response."""
        score = 0.5  # Base score

        # Increase for successful steps
        successful_steps = sum(1 for s in steps if s.success)
        score += (successful_steps / max(len(steps), 1)) * 0.2

        # Increase for found layers
        if working_memory.get("layers_found"):
            score += 0.1

        # Increase for query results
        results = working_memory.get("query_results", [])
        if results and any(r.get("success") for r in results):
            score += 0.1

        # Increase for generated insight
        if working_memory.get("final_insight"):
            score += 0.1

        return min(score, 1.0)

    async def _generate_suggestions(
        self,
        query: str,
        working_memory: dict,
    ) -> list[str]:
        """Generate follow-up suggestions for the user."""
        layers_found = working_memory.get("layers_found", [])

        suggestions = []

        if layers_found:
            # Suggest related queries based on found layers
            primary_layer = layers_found[0]
            try:
                layer_suggestions = await self.retriever.suggest_queries(
                    primary_layer, limit=2
                )
                suggestions.extend(layer_suggestions)
            except Exception:
                pass

            # Suggest comparison if multiple layers
            if len(layers_found) > 1:
                suggestions.append(
                    f"Compare {layers_found[0]} with {layers_found[1]}"
                )

        # Generic suggestions
        if not suggestions:
            suggestions = [
                "Show me a map of this data",
                "What's the trend over time?",
                "Compare this with nearby areas",
            ]

        return suggestions[:3]


# =============================================================================
# Tool Functions for GIS Agent Integration
# =============================================================================

async def discover_layers_for_query(query: str) -> dict:
    """
    Tool function: Discover relevant GIS layers for a query.

    Use this when you need to find which data layers are relevant
    for answering a user's question about geographic or market data.
    """
    catalog = get_layer_catalog_service_sync()
    results = await catalog.search_layers(query, limit=5)

    return {
        "layers": [
            {
                "name": r.layer.name,
                "display_name": r.layer.display_name,
                "description": r.layer.description,
                "category": r.layer.category,
                "relevance_score": r.similarity_score,
                "url": r.layer.layer_url,
                "fields": [f.name for f in r.layer.fields[:10]],
            }
            for r in results
        ],
        "total_found": len(results),
    }


async def query_layer_natural_language(
    query: str,
    layer_name: Optional[str] = None,
) -> dict:
    """
    Tool function: Query a GIS layer using natural language.

    Automatically converts the natural language query to structured
    ArcGIS query parameters and returns the results.
    """
    orchestrator = get_orchestrator()
    response = await orchestrator.process(query)

    return {
        "answer": response.answer,
        "layers_used": response.layers_used,
        "confidence": response.confidence,
        "suggestions": response.suggestions,
    }


async def get_layer_info(layer_name: str) -> dict:
    """
    Tool function: Get detailed information about a specific layer.

    Returns schema, field descriptions, and suggested queries.
    """
    catalog = get_layer_catalog_service_sync()
    layer = await catalog.get_layer(layer_name)

    if not layer:
        return {"error": f"Layer '{layer_name}' not found"}

    return {
        "name": layer.name,
        "display_name": layer.display_name,
        "description": layer.description,
        "category": layer.category,
        "url": layer.layer_url,
        "geometry_type": layer.geometry_type.value if layer.geometry_type else None,
        "fields": [
            {
                "name": f.name,
                "alias": f.alias,
                "type": f.field_type,
                "description": f.semantic_description,
                "filterable": f.is_filterable,
            }
            for f in layer.fields
        ],
        "common_queries": layer.common_queries,
        "related_layers": layer.related_layers,
    }


# =============================================================================
# Factory Functions
# =============================================================================

_orchestrator: Optional[AgenticOrchestrator] = None


def get_orchestrator(
    force_new: bool = False,
    config: Optional[OrchestratorConfig] = None,
) -> AgenticOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator

    if _orchestrator is None or force_new:
        _orchestrator = AgenticOrchestrator(config=config)

    return _orchestrator


def set_orchestrator(orchestrator: AgenticOrchestrator):
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator
