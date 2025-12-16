# AI Architecture Improvement Proposal

## Current Issues

1. **Slow responses for simple queries** - "hi", "what is D2 segment" take 10+ seconds
2. **No intelligent layer awareness** - AI doesn't know what layers are needed
3. **No multi-step workflow planning** - AI doesn't understand complex tasks like site selection
4. **Context overflow** - Large responses cause API errors on follow-up queries

## Proposed Tiered Response Architecture

### Tier 1: Instant Responses (< 100ms)
**No LLM call required** - Pattern matching + cached responses

```
Queries:
- Greetings: "hi", "hello", "hey"
- Simple capabilities: "what can you do?", "help"
- Common segment lookups: "what is D2 segment?"

Implementation:
- Embedding-based intent classification (cosine similarity)
- Pre-cached response templates
- No network calls to LLM
```

### Tier 2: Fast Execution (< 2s)
**Single tool call with minimal LLM reasoning**

```
Queries:
- Map navigation: "zoom to Dallas", "pan right"
- Simple pins: "add pin on Dallas"
- Layer toggles: "show lifestyle layer"

Implementation:
- Intent classifier identifies action type
- Direct tool execution via workflow templates
- Brief LLM confirmation message
```

### Tier 3: Smart Execution (2-10s)
**Multi-step workflows with LLM planning**

```
Queries:
- "What are nearby lifestyles for 123 Main St?"
- "Create 15-minute drive time polygon"
- "Get demographics for this location"

Implementation:
- GIS Agent handles full workflow
- Geocode → Zoom → Pin → Create Polygon → Get Data
- Structured response with insights
```

### Tier 4: Complex Analysis (10-30s)
**Multi-agent collaboration with planning**

```
Queries:
- "What's the best place to open a car wash in Dallas?"
- "Analyze competition for a coffee shop in this area"
- "Create a market analysis for retail expansion"

Implementation:
- Planning Agent creates task breakdown
- Multiple sub-agents execute in parallel
- Layer availability check BEFORE execution
- Aggregated insights with recommendations
```

## Proposed Architecture Components

### 1. Fast Intent Router (New)

```python
class FastIntentRouter:
    """
    Sub-millisecond intent classification using embeddings.
    Routes 95% of queries instantly, escalates 5% to LLM.
    """

    INTENT_CATEGORIES = {
        "greeting": ["hi", "hello", "hey", "good morning"],
        "capabilities": ["what can you do", "help", "your capabilities"],
        "segment_lookup": ["what is * segment", "tell me about * segment"],
        "map_navigation": ["zoom to *", "pan *", "go to *"],
        "pin_operation": ["add pin *", "remove pin *", "pin on *"],
        "layer_operation": ["show * layer", "hide * layer", "toggle *"],
        "lifestyle_analysis": ["nearby lifestyles", "lifestyle for", "demographics"],
        "site_selection": ["best place for", "where should I open", "site analysis"],
    }

    def classify(self, query: str) -> tuple[str, float]:
        """Returns (intent, confidence) in < 1ms"""
        # Use sentence embeddings + FAISS for fast similarity
        pass
```

### 2. Response Cache (New)

```python
class ResponseCache:
    """
    Cache responses for common queries.
    - Segment definitions (D1, D2, 1A, etc.)
    - LifeMode group descriptions
    - Common capability questions
    """

    SEGMENT_CACHE = {
        "D1": "Savvy Suburbanites - Affluent, educated couples...",
        "D2": "Comfortable Empty Nesters - Upper-middle-class...",
        "D3": "Booming and Consuming - Prosperous families...",
        # ... all 67 segments
    }

    def get_cached_response(self, intent: str, entities: dict) -> Optional[str]:
        pass
```

### 3. Layer Awareness System (New)

```python
class LayerAwarenessSystem:
    """
    Tracks what layers are available on the map and what's needed for each task.
    """

    TASK_LAYER_REQUIREMENTS = {
        "lifestyle_analysis": ["Tapestry Segmentation 2025", "Demographics"],
        "site_selection": ["POI Layer", "Traffic", "Demographics", "Competition"],
        "trade_area": ["Drive Time Service"],
        "demographic_report": ["Demographics", "Census"],
    }

    def check_layer_availability(self, task: str, map_context: dict) -> dict:
        """
        Returns:
        {
            "available": ["Demographics"],
            "missing": ["Tapestry Segmentation 2025"],
            "message": "Add Tapestry 2025 layer for lifestyle analysis"
        }
        """
        pass
```

### 4. Workflow Planner Agent (New)

```python
class WorkflowPlannerAgent(LlmAgent):
    """
    Plans complex multi-step workflows before execution.
    Uses chain-of-thought reasoning to break down tasks.
    """

    WORKFLOW_TEMPLATES = {
        "site_selection": [
            "1. Geocode target area",
            "2. Check required layers (POI, Traffic, Demographics)",
            "3. Create trade area polygon",
            "4. Query POI for existing competitors",
            "5. Analyze demographics",
            "6. Score locations based on criteria",
            "7. Generate recommendations"
        ],
        "lifestyle_report": [
            "1. Geocode address",
            "2. Zoom to location",
            "3. Add location pin",
            "4. Create drive-time polygon",
            "5. Get tapestry segmentation",
            "6. Get demographics",
            "7. Generate insights report"
        ]
    }
```

### 5. Enhanced Root Agent Routing

```python
# Updated routing logic in root_agent.py

def _get_system_instruction(self) -> str:
    return """
## FAST RESPONSE RULES (Answer IMMEDIATELY without tools)

### Greetings
"hi", "hello" → Friendly greeting + brief capability summary

### Segment Questions
"what is [X] segment" → Answer from knowledge:
- D1: Savvy Suburbanites
- D2: Comfortable Empty Nesters
- D3: Booming and Consuming
[... all segments ...]

### Capability Questions
"what can you do" → List capabilities briefly

### LifeMode Questions
"what is LifeMode A" → Answer from knowledge

## QUICK EXECUTION (Route to GIS, execute fast)

### Map Navigation
"zoom to [place]" → GIS: zoom_to_location
"pan [direction]" → GIS: pan_map
"add pin on [place]" → GIS: geocode + add_pin

## SMART EXECUTION (Multi-step workflow)

### Lifestyle Analysis
"nearby lifestyles for [address]" → GIS full workflow

## COMPLEX ANALYSIS (Plan first, then execute)

### Site Selection
"best place for [business] in [area]" → WorkflowPlanner → GIS + Analytics
"""
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Expand QUICK_RESPONSES** in root_agent.py with all common queries
2. **Add segment lookup** to quick responses (all 67 segments)
3. **Improve workflow templates** for zoom/pan/pin operations

### Phase 2: Intent Classification (3-5 days)
1. Build FastIntentRouter with embedding-based classification
2. Add response cache for segments and capabilities
3. Integrate with root agent for instant responses

### Phase 3: Layer Awareness (3-5 days)
1. Build LayerAwarenessSystem
2. Integrate with GIS agent
3. Add user prompts for missing layers

### Phase 4: Workflow Planning (5-7 days)
1. Build WorkflowPlannerAgent
2. Add complex query handling
3. Implement parallel execution where possible

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Greeting response time | 2-3s | < 100ms |
| Segment lookup time | 10-15s | < 500ms |
| Map navigation time | 3-5s | < 1s |
| Lifestyle report time | 30-60s | 10-15s |
| Site selection time | N/A | 20-30s |

## Sources

- [Google ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/)
- [Google ADK Best Practices](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- [Intent Classification in <1ms](https://medium.com/@durgeshrathod.777/intent-classification-in-1ms-how-we-built-a-lightning-fast-classifier-with-embeddings-db76bfb6d964)
- [Speculative Cascades for LLM Inference](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [Hybrid LLM + Intent Classification](https://medium.com/data-science-collective/intent-driven-natural-language-interface-a-hybrid-llm-intent-classification-approach-e1d96ad6f35d)
