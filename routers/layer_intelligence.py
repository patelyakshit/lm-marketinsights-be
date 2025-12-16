"""
Layer Intelligence REST API Router.

Provides REST endpoints for:
- Layer catalog management and sync
- Semantic layer search
- Natural language querying
- Knowledge graph operations
- System health and stats
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/layer-intelligence",
    tags=["Layer Intelligence"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class LayerSearchRequest(BaseModel):
    """Request model for layer search."""
    query: str = Field(..., description="Natural language search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum results to return")


class LayerSearchResult(BaseModel):
    """Single layer search result."""
    name: str
    display_name: str
    description: str
    category: str
    relevance_score: float
    url: str
    fields: list[dict]


class LayerSearchResponse(BaseModel):
    """Response model for layer search."""
    success: bool
    total_found: int
    layers: list[LayerSearchResult]


class NaturalLanguageQueryRequest(BaseModel):
    """Request for natural language query processing."""
    query: str = Field(..., description="Natural language question")
    layer_name: Optional[str] = Field(None, description="Specific layer to query")
    include_steps: bool = Field(False, description="Include execution steps in response")


class NaturalLanguageQueryResponse(BaseModel):
    """Response from natural language query."""
    success: bool
    answer: str
    layers_used: list[str]
    confidence: float
    suggestions: list[str]
    execution_time_ms: int
    steps: Optional[list[dict]] = None


class LayerSyncRequest(BaseModel):
    """Request to sync layers from ArcGIS."""
    group_id: Optional[str] = Field(None, description="ArcGIS group ID to sync")
    item_id: Optional[str] = Field(None, description="Single item ID to sync")
    enrich_semantics: bool = Field(True, description="Generate semantic descriptions")


class LayerSyncResponse(BaseModel):
    """Response from layer sync operation."""
    success: bool
    message: str
    layers_synced: int
    task_id: Optional[str] = None


class LayerDetailResponse(BaseModel):
    """Detailed layer information."""
    success: bool
    layer: Optional[dict] = None
    error: Optional[str] = None


class QuerySuggestionsResponse(BaseModel):
    """Suggested queries for a layer."""
    success: bool
    layer: str
    suggestions: list[str]


class SystemStatsResponse(BaseModel):
    """System statistics response."""
    success: bool
    catalog: dict
    embedding: dict
    knowledge_graph: dict


# =============================================================================
# Layer Search Endpoints
# =============================================================================

@router.post("/search", response_model=LayerSearchResponse)
async def search_layers(request: LayerSearchRequest):
    """
    Search for relevant data layers using semantic similarity.

    Finds layers that match the natural language query based on
    layer descriptions, field names, and semantic tags.
    """
    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()
        results = await catalog.search_layers(request.query, limit=request.limit)

        layers = []
        for r in results:
            layer = r.layer
            layers.append(LayerSearchResult(
                name=layer.name,
                display_name=layer.display_name,
                description=layer.description,
                category=layer.category,
                relevance_score=round(r.similarity_score, 3),
                url=layer.layer_url,
                fields=[
                    {"name": f.name, "alias": f.alias, "type": f.field_type}
                    for f in layer.fields[:10]
                ],
            ))

        return LayerSearchResponse(
            success=True,
            total_found=len(results),
            layers=layers,
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Layer Intelligence service not initialized. Run sync first.",
        )
    except Exception as e:
        logger.error(f"Layer search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers/{layer_name}", response_model=LayerDetailResponse)
async def get_layer_details(layer_name: str):
    """
    Get detailed information about a specific layer.

    Returns full schema, field descriptions, and query suggestions.
    """
    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()
        layer = await catalog.get_layer(layer_name)

        if not layer:
            return LayerDetailResponse(
                success=False,
                error=f"Layer '{layer_name}' not found",
            )

        return LayerDetailResponse(
            success=True,
            layer={
                "name": layer.name,
                "display_name": layer.display_name,
                "description": layer.description,
                "category": layer.category,
                "layer_url": layer.layer_url,
                "geometry_type": layer.geometry_type.value if layer.geometry_type else None,
                "record_count": layer.record_count,
                "fields": [
                    {
                        "name": f.name,
                        "alias": f.alias,
                        "type": f.field_type,
                        "description": f.semantic_description,
                        "is_numeric": f.is_numeric,
                        "is_date": f.is_date,
                        "sample_values": f.sample_values[:5] if f.sample_values else [],
                    }
                    for f in layer.fields
                ],
                "common_queries": layer.common_queries,
                "semantic_tags": layer.semantic_tags,
                "related_layers": layer.related_layers,
            },
        )

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Get layer details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers/{layer_name}/suggestions", response_model=QuerySuggestionsResponse)
async def get_query_suggestions(layer_name: str, limit: int = Query(5, ge=1, le=10)):
    """
    Get suggested natural language queries for a layer.

    Returns example questions that can be answered using this layer's data.
    """
    try:
        from services.layer_intelligence.self_query import get_self_query_retriever

        retriever = get_self_query_retriever()
        suggestions = await retriever.suggest_queries(layer_name, limit=limit)

        return QuerySuggestionsResponse(
            success=True,
            layer=layer_name,
            suggestions=suggestions,
        )

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Query suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Natural Language Query Endpoints
# =============================================================================

@router.post("/query", response_model=NaturalLanguageQueryResponse)
async def natural_language_query(request: NaturalLanguageQueryRequest):
    """
    Process a natural language query against GIS data.

    Automatically identifies relevant layers, converts the query to
    structured format, executes it, and returns analyzed results.
    """
    try:
        from services.layer_intelligence.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        response = await orchestrator.process(request.query)

        result = NaturalLanguageQueryResponse(
            success=True,
            answer=response.answer,
            layers_used=response.layers_used,
            confidence=round(response.confidence, 2),
            suggestions=response.suggestions,
            execution_time_ms=response.execution_time_ms,
        )

        if request.include_steps:
            result.steps = [
                {
                    "action": step.action.value,
                    "reasoning": step.reasoning,
                    "success": step.success,
                    "duration_ms": step.duration_ms,
                }
                for step in response.steps
            ]

        return result

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Natural language query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Sync and Management Endpoints
# =============================================================================

@router.post("/sync", response_model=LayerSyncResponse)
async def sync_layers(request: LayerSyncRequest, background_tasks: BackgroundTasks):
    """
    Sync layers from ArcGIS Online into the catalog.

    Can sync from a group or a single item. Runs in background
    for large syncs to avoid timeout.
    """
    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()

        if request.item_id:
            # Sync single item - run synchronously
            layer = await catalog.sync_single_layer(request.item_id)
            return LayerSyncResponse(
                success=True,
                message=f"Synced layer: {layer.display_name}",
                layers_synced=1,
            )

        elif request.group_id:
            # Sync group - run in background
            import uuid
            task_id = str(uuid.uuid4())

            async def sync_group_task():
                try:
                    layers = await catalog.sync_from_arcgis_group(group_id=request.group_id)
                    logger.info(f"Sync task {task_id} completed: {len(layers)} layers")
                except Exception as e:
                    logger.error(f"Sync task {task_id} failed: {e}")

            background_tasks.add_task(sync_group_task)

            return LayerSyncResponse(
                success=True,
                message="Sync started in background",
                layers_synced=0,
                task_id=task_id,
            )

        else:
            # Sync default group
            from services.layer_intelligence.config import get_config
            config = get_config()
            layers = await catalog.sync_from_arcgis_group(
                group_id=config.arcgis.curated_layers_group_id,
            )
            return LayerSyncResponse(
                success=True,
                message=f"Synced {len(layers)} layers from default group",
                layers_synced=len(layers),
            )

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layers")
async def list_all_layers(
    category: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List all layers in the catalog.

    Optionally filter by category. Supports pagination.
    """
    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service_sync

        catalog = get_layer_catalog_service_sync()

        # Get all layers (we'll implement proper listing later)
        # For now, use a broad search
        results = await catalog.search_layers("*", limit=limit)

        layers = []
        for r in results:
            layer = r.layer
            if category and layer.category != category:
                continue
            layers.append({
                "name": layer.name,
                "display_name": layer.display_name,
                "category": layer.category,
                "description": layer.description[:200] if layer.description else "",
            })

        return {
            "success": True,
            "total": len(layers),
            "offset": offset,
            "limit": limit,
            "layers": layers[offset:offset + limit],
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"List layers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Knowledge Graph Endpoints
# =============================================================================

@router.get("/graph/related/{layer_name}")
async def get_related_layers(
    layer_name: str,
    context: Optional[str] = None,
):
    """
    Find layers related to a given layer.

    Uses the knowledge graph to discover complementary data sources.
    """
    try:
        from services.layer_intelligence.knowledge_graph import get_knowledge_graph_service_sync

        graph = get_knowledge_graph_service_sync()
        suggested = await graph.suggest_analysis_layers(
            context or f"analysis for {layer_name}",
            [layer_name],
        )

        relationships = []
        for related in suggested:
            if related != layer_name:
                path = await graph.find_cross_layer_path(layer_name, related)
                relationships.append({
                    "layer": related,
                    "relationship": path.explanation if path else "Related data",
                })

        return {
            "success": True,
            "source_layer": layer_name,
            "related_layers": relationships,
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Related layers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/path")
async def find_layer_path(
    from_layer: str = Query(..., description="Source layer name"),
    to_layer: str = Query(..., description="Target layer name"),
):
    """
    Find the relationship path between two layers.

    Uses knowledge graph to explain how layers are connected.
    """
    try:
        from services.layer_intelligence.knowledge_graph import get_knowledge_graph_service_sync

        graph = get_knowledge_graph_service_sync()
        path = await graph.find_cross_layer_path(from_layer, to_layer)

        if not path:
            return {
                "success": False,
                "message": f"No path found between {from_layer} and {to_layer}",
            }

        return {
            "success": True,
            "from_layer": from_layer,
            "to_layer": to_layer,
            "explanation": path.explanation,
            "score": path.score,
            "path_length": len(path.nodes),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="Service not initialized")
    except Exception as e:
        logger.error(f"Find path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# System Health and Stats
# =============================================================================

@router.get("/health")
async def health_check():
    """Check if Layer Intelligence services are healthy."""
    status = {
        "catalog": False,
        "embedding": False,
        "knowledge_graph": False,
        "overall": False,
    }

    try:
        from services.layer_intelligence.layer_catalog import get_layer_catalog_service
        catalog = get_layer_catalog_service()
        status["catalog"] = True
    except Exception as e:
        logger.warning(f"Catalog health check failed: {e}")

    try:
        from services.layer_intelligence.embedding_service import get_embedding_service
        embedding = get_embedding_service()
        status["embedding"] = True
    except Exception as e:
        logger.warning(f"Embedding health check failed: {e}")

    try:
        from services.layer_intelligence.knowledge_graph import get_knowledge_graph_service
        graph = get_knowledge_graph_service()
        status["knowledge_graph"] = True
    except Exception as e:
        logger.warning(f"Knowledge graph health check failed: {e}")

    status["overall"] = all([status["catalog"], status["embedding"], status["knowledge_graph"]])

    return {
        "status": "healthy" if status["overall"] else "degraded",
        "services": status,
    }


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get statistics about the Layer Intelligence system."""
    try:
        stats = {
            "catalog": {},
            "embedding": {},
            "knowledge_graph": {},
        }

        try:
            from services.layer_intelligence.embedding_service import get_embedding_service
            embedding = get_embedding_service()
            stats["embedding"] = embedding.cache_stats()
        except Exception:
            pass

        try:
            from services.layer_intelligence.knowledge_graph import get_knowledge_graph_service
            graph = get_knowledge_graph_service()
            stats["knowledge_graph"] = graph.get_stats()
        except Exception:
            pass

        return SystemStatsResponse(
            success=True,
            catalog=stats["catalog"],
            embedding=stats["embedding"],
            knowledge_graph=stats["knowledge_graph"],
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
