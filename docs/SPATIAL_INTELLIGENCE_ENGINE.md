# Spatial Intelligence Engine - Architecture

## Vision

A comprehensive AI system that understands complex GIS queries, plans multi-step spatial workflows, executes operations, queries real layer data, and generates actionable business insights.

---

## Example Query Flow

**User Query:**
> "What are nearby lifestyles for 1101 Coit Rd, Plano, TX 75075, United States for 15 min drive time?"

**AI Understanding:**
```
Intent: Lifestyle/Tapestry segment analysis
Location: 1101 Coit Rd, Plano, TX 75075
Analysis Type: Drive-time trade area
Parameters: 15 minutes
Output: Top segments with insights
```

**AI Plan:**
1. Geocode "1101 Coit Rd, Plano, TX 75075" → get lat/lng
2. Zoom map to location
3. Generate 15-minute drive-time polygon
4. Check if Tapestry 2025 layer is visible → toggle if needed
5. Spatial query: intersect drive-time with Tapestry polygons
6. Aggregate: calculate segment percentages by household count
7. Rank: identify Top 5 segments
8. Generate insights: business recommendations

**AI Response:**
> Based on the 15-minute drive time from 1101 Coit Rd, Plano, TX, here are the top 5 lifestyle segments:
>
> 1. **Savvy Suburbanites (1D)** - 23.4% of households
>    - Affluent, well-educated couples in newer suburbs
>    - High disposable income, tech-savvy consumers
>    - *Business Insight: Premium products, digital marketing effective*
>
> 2. **Professional Pride (1B)** - 18.2% of households
>    - ...

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Frontend)                     │
│  - Natural language input                                        │
│  - Map interaction (click, select)                               │
│  - Layer management                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SPATIAL INTELLIGENCE ENGINE                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   QUERY      │  │   PLANNING   │  │   EXECUTION          │  │
│  │   PARSER     │→ │   ENGINE     │→ │   ORCHESTRATOR       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    TOOL REGISTRY                          │  │
│  │  - GIS Tools (geocode, drive-time, spatial query)        │  │
│  │  - Layer Tools (query features, toggle visibility)        │  │
│  │  - Analytics Tools (aggregate, rank, calculate)           │  │
│  │  - Knowledge Tools (RAG for segment insights)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 INSIGHT GENERATOR                         │  │
│  │  - Business recommendations                               │  │
│  │  - Comparative analysis                                   │  │
│  │  - Market opportunity identification                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  ArcGIS    │  │  Feature   │  │  Knowledge │  │  Customer │ │
│  │  Services  │  │  Layers    │  │  Base      │  │  Data     │ │
│  │  (API)     │  │  (Map)     │  │  (RAG)     │  │  (CRM)    │ │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Query Parser

Understands natural language spatial queries and extracts:

### Intent Classification

| Intent | Example Query | Key Indicators |
|--------|---------------|----------------|
| `LOCATION_ANALYSIS` | "Analyze demographics for Dallas" | analyze, demographics, profile |
| `TRADE_AREA_ANALYSIS` | "15 min drive time from store" | drive time, radius, trade area |
| `SEGMENT_ANALYSIS` | "What lifestyles are in this area" | lifestyle, segment, tapestry |
| `COMPARISON` | "Compare Austin vs Houston" | compare, vs, difference |
| `SITE_SELECTION` | "Best location for coffee shop" | best, optimal, site |
| `GAP_ANALYSIS` | "Market gaps in my trade area" | gap, opportunity, underserved |
| `LAYER_QUERY` | "What is I3 segment" | what is, tell me about |

### Entity Extraction

```python
class SpatialQueryEntities:
    location: str           # "1101 Coit Rd, Plano, TX"
    coordinates: Tuple      # (33.0123, -96.7456)
    distance: float         # 15
    distance_unit: str      # "minutes" | "miles" | "km"
    analysis_type: str      # "drive_time" | "radius" | "polygon"
    layer_name: str         # "Tapestry 2025"
    segment_id: str         # "I3"
    time_period: str        # "2025"
    output_format: str      # "top_5" | "all" | "summary"
```

---

## Component 2: Planning Engine

Creates execution plans based on parsed queries.

### Plan Structure

```python
class ExecutionPlan:
    steps: List[PlanStep]
    dependencies: Dict[str, List[str]]  # step dependencies
    parallel_groups: List[List[str]]    # steps that can run in parallel
    estimated_time: float
    required_layers: List[str]
    required_data: List[str]

class PlanStep:
    id: str
    action: str           # "geocode", "create_drive_time", "query_layer", etc.
    tool: str             # which tool to use
    parameters: Dict
    depends_on: List[str] # previous steps this depends on
    output_key: str       # where to store result
```

### Example Plan: Trade Area Lifestyle Analysis

```python
plan = ExecutionPlan(
    steps=[
        PlanStep(
            id="geocode",
            action="geocode_address",
            tool="gis_tools.geocode",
            parameters={"address": "1101 Coit Rd, Plano, TX 75075"},
            output_key="location"
        ),
        PlanStep(
            id="zoom",
            action="zoom_to_location",
            tool="map_operations.zoom",
            parameters={"location": "${location}"},
            depends_on=["geocode"]
        ),
        PlanStep(
            id="drive_time",
            action="create_drive_time",
            tool="gis_tools.service_area",
            parameters={
                "location": "${location}",
                "time_minutes": 15,
                "travel_mode": "driving"
            },
            depends_on=["geocode"],
            output_key="trade_area"
        ),
        PlanStep(
            id="ensure_layer",
            action="ensure_layer_visible",
            tool="layer_tools.toggle_visibility",
            parameters={"layer_name": "Tapestry Segmentation 2025", "visible": True}
        ),
        PlanStep(
            id="query_segments",
            action="spatial_query",
            tool="layer_tools.query_features",
            parameters={
                "layer": "Tapestry Segmentation 2025",
                "geometry": "${trade_area}",
                "spatial_rel": "intersects",
                "out_fields": ["TSEGCODE", "TSEGNAME", "TLIFENAME", "THHBASE"]
            },
            depends_on=["drive_time", "ensure_layer"],
            output_key="segment_features"
        ),
        PlanStep(
            id="aggregate",
            action="aggregate_segments",
            tool="analytics.aggregate",
            parameters={
                "features": "${segment_features}",
                "group_by": "TSEGCODE",
                "sum_field": "THHBASE",
                "calculate": ["percentage", "rank"]
            },
            depends_on=["query_segments"],
            output_key="segment_stats"
        ),
        PlanStep(
            id="top_5",
            action="get_top_n",
            tool="analytics.rank",
            parameters={
                "data": "${segment_stats}",
                "sort_by": "percentage",
                "n": 5
            },
            depends_on=["aggregate"],
            output_key="top_segments"
        ),
        PlanStep(
            id="enrich_insights",
            action="get_segment_insights",
            tool="knowledge.rag_query",
            parameters={
                "segments": "${top_segments}",
                "query_type": "business_insights"
            },
            depends_on=["top_5"],
            output_key="insights"
        ),
        PlanStep(
            id="generate_response",
            action="format_response",
            tool="response.format",
            parameters={
                "top_segments": "${top_segments}",
                "insights": "${insights}",
                "format": "detailed_with_recommendations"
            },
            depends_on=["enrich_insights"]
        )
    ]
)
```

---

## Component 3: Tool Registry

### GIS Tools (Existing + New)

```python
# Existing tools
geocode_address(address: str) -> Location
get_coordinates_for_address(address: str) -> Coordinates
get_esri_geoenrich_data(lat: float, lon: float) -> Demographics
toggle_layer_visibility(layer_name: str, visible: bool) -> Result

# NEW tools needed
create_drive_time_polygon(
    location: Location,
    time_minutes: int,
    travel_mode: str = "driving"
) -> Polygon

create_radius_buffer(
    location: Location,
    distance: float,
    unit: str = "miles"
) -> Polygon

spatial_intersect(
    geometry: Polygon,
    layer_id: str
) -> FeatureSet
```

### Layer Query Tools (NEW)

```python
async def query_layer_features(
    layer_id: str,
    geometry: Optional[Polygon] = None,
    where_clause: Optional[str] = None,
    out_fields: List[str] = ["*"],
    spatial_relationship: str = "intersects"
) -> FeatureSet:
    """
    Query features from a layer on the map.

    This enables the AI to read actual data from visible layers,
    not just metadata.
    """
    pass

async def get_layer_statistics(
    layer_id: str,
    geometry: Optional[Polygon] = None,
    stat_field: str = None,
    group_by: str = None
) -> Statistics:
    """
    Get aggregated statistics from a layer.
    """
    pass

async def get_visible_layer_extent_features(
    layer_id: str,
    max_features: int = 1000
) -> FeatureSet:
    """
    Get all features currently visible in the map extent.
    """
    pass
```

### Analytics Tools (NEW)

```python
def aggregate_by_field(
    features: FeatureSet,
    group_field: str,
    value_field: str,
    aggregation: str = "sum"  # sum, count, avg, min, max
) -> Dict[str, float]:
    """Aggregate feature values by a grouping field."""
    pass

def calculate_percentages(
    aggregated: Dict[str, float]
) -> Dict[str, float]:
    """Convert counts to percentages."""
    pass

def rank_and_filter(
    data: Dict[str, float],
    top_n: int = 5,
    sort_order: str = "desc"
) -> List[Tuple[str, float]]:
    """Rank items and return top N."""
    pass

def calculate_gap_analysis(
    market_data: Dict[str, float],
    customer_data: Dict[str, float]
) -> Dict[str, GapMetrics]:
    """
    Calculate gap and index metrics.

    Returns:
        segment_id: {
            market_pct: float,
            customer_pct: float,
            gap: float,  # market - customer
            index: float  # customer / market * 100
        }
    """
    pass
```

### Knowledge Tools (NEW)

```python
async def get_segment_description(
    segment_code: str
) -> SegmentDescription:
    """
    Get detailed description of a Tapestry segment from RAG.

    Returns:
        - Overview
        - Demographics (age, income, household composition)
        - Preferences (spending, media, lifestyle)
        - Marketing recommendations
    """
    pass

async def get_business_insights(
    segments: List[str],
    business_type: Optional[str] = None
) -> BusinessInsights:
    """
    Generate business insights based on segment composition.
    """
    pass
```

---

## Component 4: Layer Data Access Framework

### Frontend → Backend Communication

```typescript
// Frontend: New WebSocket message types

// Request layer query from backend
interface LayerQueryRequest {
  type: "LAYER/QUERY_REQUEST";
  payload: {
    request_id: string;
    layer_id: string;
    geometry?: GeoJSON;
    where?: string;
    out_fields?: string[];
    return_geometry?: boolean;
  };
}

// Response with layer data
interface LayerQueryResponse {
  type: "LAYER/QUERY_RESPONSE";
  payload: {
    request_id: string;
    features: Feature[];
    count: number;
    exceeded_limit: boolean;
  };
}

// Send current map context with visible layer data
interface EnhancedMapContext {
  type: "MAP/CONTEXT";
  payload: {
    center: { lat: number; lng: number };
    zoom: number;
    extent: Extent;
    visible_layers: LayerInfo[];
    selected_features?: Feature[];
    // NEW: Include feature data for visible layers in extent
    layer_features?: {
      [layer_id: string]: {
        features: Feature[];
        total_count: number;
        in_extent_count: number;
      };
    };
  };
}
```

### Backend: Layer Query Handler

```python
# New endpoint or WebSocket handler

async def handle_layer_query(
    connection_id: str,
    request: LayerQueryRequest
) -> LayerQueryResponse:
    """
    Receives layer query request, forwards to frontend,
    waits for response, returns to AI.
    """
    # 1. Send query request to frontend
    await manager.send_message(connection_id, {
        "type": "LAYER/QUERY_REQUEST",
        "payload": request.payload
    })

    # 2. Wait for response (with timeout)
    response = await wait_for_layer_response(
        connection_id,
        request.request_id,
        timeout=30
    )

    return response
```

---

## Component 5: Insight Generator

### Segment Insight Template

```python
class SegmentInsight:
    segment_code: str       # "I3"
    segment_name: str       # "Heartland Communities"
    lifemode_group: str     # "Midlife Constants"
    percentage: float       # 23.4
    household_count: int    # 12,450

    # From RAG knowledge base
    overview: str
    demographics: Dict[str, Any]
    preferences: Dict[str, Any]

    # Generated insights
    business_implications: List[str]
    marketing_recommendations: List[str]
    opportunities: List[str]
    challenges: List[str]
```

### Response Format Templates

```python
def format_trade_area_analysis(
    location: str,
    trade_area_type: str,
    top_segments: List[SegmentInsight],
    total_households: int
) -> str:
    """
    Format a comprehensive trade area analysis response.
    """
    response = f"""
## Trade Area Analysis: {location}
**Analysis Type:** {trade_area_type}
**Total Households:** {total_households:,}

### Top 5 Lifestyle Segments

"""
    for i, seg in enumerate(top_segments, 1):
        response += f"""
#### {i}. {seg.segment_name} ({seg.segment_code}) - {seg.percentage:.1f}%
*{seg.lifemode_group}*

{seg.overview}

**Key Characteristics:**
{format_bullet_list(seg.demographics)}

**Business Implications:**
{format_bullet_list(seg.business_implications)}

**Marketing Recommendations:**
{format_bullet_list(seg.marketing_recommendations)}

---
"""

    # Add summary insights
    response += """
### Summary Insights

Based on this segment composition, this trade area shows:
{generate_summary_insights(top_segments)}

### Recommended Actions
{generate_recommendations(top_segments)}
"""
    return response
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create Tapestry segment knowledge base (RAG corpus)
- [ ] Implement layer query WebSocket protocol
- [ ] Build basic query parser with intent classification
- [ ] Add drive-time polygon creation tool

### Phase 2: Core Intelligence (Week 3-4)
- [ ] Build planning engine with step execution
- [ ] Implement analytics tools (aggregate, rank, gap)
- [ ] Create insight generator with templates
- [ ] Integrate RAG for segment descriptions

### Phase 3: Advanced Features (Week 5-6)
- [ ] Add comparison analysis
- [ ] Implement gap analysis
- [ ] Build site selection scoring
- [ ] Add customer data integration

### Phase 4: Polish & Scale (Week 7-8)
- [ ] Performance optimization
- [ ] Caching strategy
- [ ] Error handling & recovery
- [ ] Documentation & testing

---

## Example Queries the System Should Handle

| Query | AI Actions |
|-------|------------|
| "What are the lifestyles in this area?" | Query visible Tapestry layer in current extent |
| "Create 15 min drive time and show segments" | Geocode → Drive time → Query → Display |
| "Compare segments between my two stores" | Query both areas → Calculate → Compare |
| "What's the opportunity gap in Beverly Hills?" | Market data → Customer data → Gap analysis |
| "Best location for a gym in Dallas" | Site selection scoring based on segments |
| "Tell me about I3 Heartland Communities" | RAG query → Segment description |
| "Export segment analysis to report" | Generate formatted report |

---

## Data Requirements

### 1. Tapestry Knowledge Base
- All 61 segment descriptions (2025)
- Demographics per segment
- Preferences and behaviors
- Marketing recommendations
- Historical: 68 segments (2024) for comparison

### 2. Layer Configuration
- Tapestry 2025 polygon layer (block group level)
- Household count field mapping
- Segment code/name fields

### 3. ArcGIS Services
- Geocoding service
- Routing/Service Area service
- GeoEnrichment (backup)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Query understanding accuracy | > 95% |
| Plan execution success rate | > 90% |
| Response time (simple query) | < 3 seconds |
| Response time (complex analysis) | < 15 seconds |
| Insight relevance score | > 4.5/5 user rating |
