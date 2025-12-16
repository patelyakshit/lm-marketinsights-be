"""
Layer Catalog Service - Auto-discovers and indexes ArcGIS layers for AI querying.

This is the core component that:
1. Syncs layers from ArcGIS Online groups
2. Extracts field metadata and generates semantic descriptions
3. Creates vector embeddings for semantic search
4. Maintains an index of all available layers
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from .config import get_config, LayerIntelligenceConfig
from .models import (
    LayerType,
    GeometryType,
    FieldMetadata,
    LayerMetadata,
    LayerSearchResult,
)
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class LayerCatalogService:
    """
    Central service for layer discovery and metadata management.

    Features:
    - Auto-sync from ArcGIS Online groups
    - Semantic search using vector embeddings
    - Field-level metadata with AI-generated descriptions
    - Query template generation
    """

    def __init__(
        self,
        config: Optional[LayerIntelligenceConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_client: Optional[Any] = None,
    ):
        self.config = config or get_config()
        self.embedding_service = embedding_service or get_embedding_service()

        # Qdrant client for vector storage
        self._qdrant = qdrant_client
        self._qdrant_initialized = False

        # In-memory layer cache
        self._layers: dict[str, LayerMetadata] = {}
        self._layers_by_item_id: dict[str, str] = {}  # arcgis_item_id -> name

        # ArcGIS connection
        self._gis = None

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self):
        """Initialize the service and vector collections."""
        await self._init_qdrant()
        logger.info("LayerCatalogService initialized")

    async def _init_qdrant(self):
        """Initialize Qdrant collections."""
        if self._qdrant_initialized:
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self._qdrant is None:
                self._qdrant = QdrantClient(
                    host=self.config.qdrant.host,
                    port=self.config.qdrant.port,
                    api_key=self.config.qdrant.api_key,
                    https=self.config.qdrant.https,
                )

            # Create layer catalog collection
            collections = self._qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.qdrant.layer_collection not in collection_names:
                self._qdrant.create_collection(
                    collection_name=self.config.qdrant.layer_collection,
                    vectors_config=VectorParams(
                        size=self.embedding_service.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.config.qdrant.layer_collection}")

            # Create field collection
            if self.config.qdrant.field_collection not in collection_names:
                self._qdrant.create_collection(
                    collection_name=self.config.qdrant.field_collection,
                    vectors_config=VectorParams(
                        size=self.embedding_service.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.config.qdrant.field_collection}")

            self._qdrant_initialized = True

        except ImportError:
            logger.warning("qdrant-client not installed. Vector search will be disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")

    def _get_gis(self):
        """Get or create ArcGIS GIS connection."""
        if self._gis is None:
            try:
                from arcgis.gis import GIS

                self._gis = GIS(
                    self.config.arcgis.portal_url,
                    self.config.arcgis.username,
                    self.config.arcgis.password,
                )
                logger.info(f"Connected to ArcGIS as: {self._gis.users.me.username}")
            except Exception as e:
                logger.error(f"Failed to connect to ArcGIS: {e}")
                raise

        return self._gis

    # =========================================================================
    # Layer Sync
    # =========================================================================

    async def sync_from_arcgis_group(
        self,
        group_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> list[LayerMetadata]:
        """
        Sync all layers from an ArcGIS Online group.

        Args:
            group_id: ArcGIS group ID (uses default from config if not provided)
            force_refresh: Force re-sync even if already cached

        Returns:
            List of synced layer metadata
        """
        group_id = group_id or self.config.arcgis.curated_layers_group_id
        logger.info(f"Syncing layers from ArcGIS group: {group_id}")

        try:
            gis = self._get_gis()
            group = gis.groups.get(group_id)

            if not group:
                logger.error(f"Group not found: {group_id}")
                return []

            items = group.content()
            logger.info(f"Found {len(items)} items in group")

            synced_layers = []
            for item in items:
                try:
                    layer = await self.sync_single_layer(item.id, force_refresh=force_refresh)
                    if layer:
                        synced_layers.append(layer)
                except Exception as e:
                    logger.error(f"Error syncing item {item.id}: {e}")
                    continue

            logger.info(f"Successfully synced {len(synced_layers)} layers")
            return synced_layers

        except Exception as e:
            logger.error(f"Error syncing from group: {e}")
            return []

    async def sync_single_layer(
        self,
        item_id: str,
        force_refresh: bool = False,
    ) -> Optional[LayerMetadata]:
        """
        Sync a single layer by its ArcGIS item ID.

        Args:
            item_id: ArcGIS item ID
            force_refresh: Force re-sync even if cached

        Returns:
            Layer metadata or None if sync failed
        """
        # Check cache
        if item_id in self._layers_by_item_id and not force_refresh:
            layer_name = self._layers_by_item_id[item_id]
            return self._layers.get(layer_name)

        try:
            gis = self._get_gis()
            item = gis.content.get(item_id)

            if not item:
                logger.warning(f"Item not found: {item_id}")
                return None

            # Extract metadata based on item type
            layer = await self._extract_layer_metadata(item)

            if layer:
                # Generate semantic enrichment
                layer = await self._enrich_layer_semantics(layer)

                # Generate embeddings and index
                await self._index_layer(layer)

                # Store in cache
                self._layers[layer.name] = layer
                self._layers_by_item_id[item_id] = layer.name

                logger.info(f"Synced layer: {layer.name}")

            return layer

        except Exception as e:
            logger.error(f"Error syncing layer {item_id}: {e}")
            return None

    async def _extract_layer_metadata(self, item) -> Optional[LayerMetadata]:
        """Extract metadata from an ArcGIS item."""
        try:
            # Determine layer type
            type_mapping = {
                "Feature Service": LayerType.FEATURE,
                "Feature Layer": LayerType.FEATURE,
                "Map Service": LayerType.MAP_IMAGE,
                "Vector Tile Layer": LayerType.VECTOR_TILE,
                "Tile Layer": LayerType.TILE,
                "Group Layer": LayerType.GROUP,
                "Scene Layer": LayerType.SCENE,
            }

            layer_type = type_mapping.get(item.type, LayerType.FEATURE)

            # Generate unique name from title
            name = self._generate_layer_name(item.title)

            layer = LayerMetadata(
                id=str(uuid4()),
                arcgis_item_id=item.id,
                name=name,
                display_name=item.title,
                layer_url=item.url or "",
                portal_url=f"{self.config.arcgis.portal_url}/home/item.html?id={item.id}",
                owner=item.owner,
                layer_type=layer_type,
                description=item.description or item.snippet or "",
                semantic_tags=item.tags or [],
                last_synced=datetime.utcnow(),
            )

            # Extract fields for feature services
            if layer_type == LayerType.FEATURE and item.url:
                layer = await self._extract_feature_service_metadata(layer, item)

            return layer

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None

    async def _extract_feature_service_metadata(
        self,
        layer: LayerMetadata,
        item,
    ) -> LayerMetadata:
        """Extract detailed metadata from a feature service."""
        try:
            from arcgis.features import FeatureLayer

            # Get the feature layer (usually index 0)
            if hasattr(item, 'layers') and item.layers:
                fl = item.layers[0]
            else:
                fl = FeatureLayer(f"{item.url}/0")

            # Get layer properties
            props = fl.properties

            # Geometry type
            geom_mapping = {
                "esriGeometryPoint": GeometryType.POINT,
                "esriGeometryPolyline": GeometryType.POLYLINE,
                "esriGeometryPolygon": GeometryType.POLYGON,
                "esriGeometryMultipoint": GeometryType.MULTIPOINT,
                "esriGeometryEnvelope": GeometryType.ENVELOPE,
            }
            layer.geometry_type = geom_mapping.get(
                props.get("geometryType", ""),
                GeometryType.POLYGON
            )

            # Extract fields
            skip_fields = {
                "objectid", "oid", "fid",
                "shape", "shape_length", "shape_area",
                "globalid", "created_date", "last_edited_date",
                "created_user", "last_edited_user",
            }

            fields = []
            for f in props.get("fields", []):
                field_name = f.get("name", "")
                if field_name.lower() in skip_fields:
                    continue

                field_meta = FieldMetadata(
                    name=field_name,
                    alias=f.get("alias", field_name),
                    field_type=f.get("type", "esriFieldTypeString"),
                )
                fields.append(field_meta)

            layer.fields = fields

            # Get extent
            if "extent" in props:
                layer.extent = props["extent"]

            # Get record count (if available)
            try:
                count_result = fl.query(where="1=1", return_count_only=True)
                layer.record_count = count_result
            except:
                pass

        except Exception as e:
            logger.warning(f"Error extracting feature service metadata: {e}")

        return layer

    def _generate_layer_name(self, title: str) -> str:
        """Generate a unique layer name from title."""
        import re

        # Convert to lowercase and replace spaces/special chars
        name = title.lower()
        name = re.sub(r'[^a-z0-9]+', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')

        # Ensure uniqueness
        base_name = name
        counter = 1
        while name in self._layers:
            name = f"{base_name}_{counter}"
            counter += 1

        return name

    # =========================================================================
    # Semantic Enrichment
    # =========================================================================

    async def _enrich_layer_semantics(self, layer: LayerMetadata) -> LayerMetadata:
        """
        Enrich layer with AI-generated semantic descriptions.

        Uses LLM to generate:
        - Enhanced layer description
        - Category classification
        - Semantic tags
        - Common query examples
        - Field descriptions
        """
        try:
            # Generate field descriptions
            for field in layer.fields:
                field.semantic_description = self._generate_field_description(
                    field, layer
                )
                field.related_concepts = self._extract_field_concepts(field)

            # Categorize layer
            layer.category = self._categorize_layer(layer)

            # Generate common queries
            layer.common_queries = self._generate_common_queries(layer)

            # Generate query templates
            layer.query_templates = self._generate_query_templates(layer)

        except Exception as e:
            logger.warning(f"Error enriching layer semantics: {e}")

        return layer

    def _generate_field_description(
        self,
        field: FieldMetadata,
        layer: LayerMetadata,
    ) -> str:
        """Generate a semantic description for a field."""
        name_lower = field.name.lower()
        alias_lower = (field.alias or "").lower()

        # Common field patterns and descriptions
        patterns = {
            # Rental/Housing
            "median": "Median value",
            "rent": "Rental price or rent-related metric",
            "apt": "Apartment-related data",
            "house": "House/single-family home data",
            "bed": "Bedroom count or bedroom-specific metric",
            "0bed": "Studio apartment (0 bedroom)",
            "1bed": "One-bedroom unit",
            "2bed": "Two-bedroom unit",
            "3bed": "Three-bedroom unit",
            "4bed": "Four-bedroom unit",
            "change": "Change or trend over time",
            "percent": "Percentage value",

            # Demographics
            "pop": "Population count",
            "totpop": "Total population",
            "hh": "Household count",
            "income": "Income level",
            "medinc": "Median income",
            "medhinc": "Median household income",
            "age": "Age-related metric",

            # Tapestry
            "tseg": "Tapestry segment",
            "tlife": "Tapestry LifeMode group",
            "thhbase": "Tapestry household base count",

            # Geographic
            "name": "Geographic area name",
            "geoid": "Geographic identifier (FIPS code)",
            "msa": "Metropolitan Statistical Area",
            "state": "State name or code",
            "county": "County name or code",
            "lat": "Latitude coordinate",
            "lon": "Longitude coordinate",
            "lng": "Longitude coordinate",
        }

        # Build description from patterns
        desc_parts = []
        for pattern, description in patterns.items():
            if pattern in name_lower or pattern in alias_lower:
                desc_parts.append(description)

        if desc_parts:
            return ". ".join(desc_parts[:3])

        # Default based on type
        if field.is_numeric:
            return f"Numeric value for {field.alias or field.name}"
        elif field.is_date:
            return f"Date/time value for {field.alias or field.name}"
        else:
            return f"Text/category value for {field.alias or field.name}"

    def _extract_field_concepts(self, field: FieldMetadata) -> list[str]:
        """Extract related concepts from a field."""
        name_lower = field.name.lower()
        concepts = []

        concept_mapping = {
            "rent": ["rental_market", "housing", "affordability"],
            "income": ["economics", "wealth", "purchasing_power"],
            "pop": ["demographics", "population", "density"],
            "age": ["demographics", "life_stage"],
            "hh": ["households", "demographics", "housing"],
            "tseg": ["consumer_behavior", "lifestyle", "segmentation"],
            "traffic": ["transportation", "accessibility"],
            "crime": ["safety", "security"],
            "school": ["education", "families"],
            "business": ["commercial", "economy"],
            "retail": ["commercial", "shopping"],
        }

        for pattern, related in concept_mapping.items():
            if pattern in name_lower:
                concepts.extend(related)

        return list(set(concepts))

    def _categorize_layer(self, layer: LayerMetadata) -> str:
        """Categorize a layer based on its content."""
        name_lower = layer.name.lower()
        desc_lower = layer.description.lower()
        tags_lower = " ".join(layer.semantic_tags).lower()
        all_text = f"{name_lower} {desc_lower} {tags_lower}"

        categories = {
            "residential": ["rent", "housing", "apartment", "house", "dwelling", "residential"],
            "demographics": ["population", "census", "demographic", "income", "age", "tapestry"],
            "commercial": ["business", "retail", "store", "commercial", "poi", "places"],
            "transportation": ["traffic", "road", "transit", "transportation", "accessibility"],
            "environmental": ["environment", "climate", "weather", "pollution", "green"],
            "boundaries": ["boundary", "border", "zone", "district", "region"],
            "infrastructure": ["utility", "infrastructure", "service", "facility"],
        }

        for category, keywords in categories.items():
            if any(kw in all_text for kw in keywords):
                return category

        return "other"

    def _generate_common_queries(self, layer: LayerMetadata) -> list[str]:
        """Generate example natural language queries for a layer."""
        queries = []

        # Based on category
        if layer.category == "residential":
            queries.extend([
                f"What is the median rent in [location]?",
                f"Show areas with rent above $[amount]",
                f"Compare rental prices between [location1] and [location2]",
            ])
        elif layer.category == "demographics":
            queries.extend([
                f"What is the population of [location]?",
                f"Show demographics for [location]",
                f"What is the median income in [location]?",
            ])

        # Based on specific fields
        for field in layer.fields[:5]:
            if field.is_numeric:
                queries.append(f"What is the {field.alias or field.name} in [location]?")

        return queries[:5]

    def _generate_query_templates(self, layer: LayerMetadata) -> dict[str, str]:
        """Generate SQL-like query templates for common operations."""
        templates = {}

        # Basic query
        out_fields = [f.name for f in layer.fields[:5]]
        templates["basic"] = f"SELECT {', '.join(out_fields)} FROM {layer.name}"

        # Numeric field queries
        numeric_fields = layer.get_numeric_fields()
        if numeric_fields:
            f = numeric_fields[0]
            templates["filter_numeric"] = f"{f.name} > [value]"
            templates["statistics"] = f"SELECT AVG({f.name}), MIN({f.name}), MAX({f.name})"

        # Text field queries
        text_fields = layer.get_text_fields()
        if text_fields:
            f = text_fields[0]
            templates["filter_text"] = f"{f.name} LIKE '%[value]%'"

        return templates

    # =========================================================================
    # Indexing
    # =========================================================================

    async def _index_layer(self, layer: LayerMetadata):
        """Index a layer in the vector database."""
        if not self._qdrant_initialized:
            await self._init_qdrant()

        if self._qdrant is None:
            return

        try:
            from qdrant_client.models import PointStruct

            # Create embedding text for layer
            embed_text = self._create_layer_embed_text(layer)

            # Generate embedding
            embedding = await self.embedding_service.embed_single(embed_text)

            # Upsert to Qdrant
            point = PointStruct(
                id=layer.id,
                vector=embedding,
                payload={
                    "name": layer.name,
                    "display_name": layer.display_name,
                    "description": layer.description,
                    "category": layer.category,
                    "semantic_tags": layer.semantic_tags,
                    "field_names": [f.name for f in layer.fields],
                    "layer_url": layer.layer_url,
                    "arcgis_item_id": layer.arcgis_item_id,
                },
            )

            self._qdrant.upsert(
                collection_name=self.config.qdrant.layer_collection,
                points=[point],
            )

            layer.embedding_id = layer.id

            # Also index individual fields
            await self._index_layer_fields(layer)

        except Exception as e:
            logger.error(f"Error indexing layer: {e}")

    async def _index_layer_fields(self, layer: LayerMetadata):
        """Index layer fields for field-level search."""
        if self._qdrant is None:
            return

        try:
            from qdrant_client.models import PointStruct

            points = []
            for field in layer.fields:
                # Create embedding text for field
                embed_text = f"{field.alias or field.name}: {field.semantic_description}"

                # Generate embedding
                embedding = await self.embedding_service.embed_single(embed_text)

                field_id = f"{layer.name}_{field.name}"
                point = PointStruct(
                    id=field_id,
                    vector=embedding,
                    payload={
                        "layer_name": layer.name,
                        "field_name": field.name,
                        "field_alias": field.alias,
                        "field_type": field.field_type,
                        "description": field.semantic_description,
                        "is_numeric": field.is_numeric,
                        "related_concepts": field.related_concepts,
                    },
                )
                points.append(point)

                field.embedding_id = field_id

            if points:
                self._qdrant.upsert(
                    collection_name=self.config.qdrant.field_collection,
                    points=points,
                )

        except Exception as e:
            logger.error(f"Error indexing fields: {e}")

    def _create_layer_embed_text(self, layer: LayerMetadata) -> str:
        """Create text for layer embedding."""
        parts = [
            layer.display_name,
            layer.description,
            layer.category,
            " ".join(layer.semantic_tags),
        ]

        # Add field information
        field_info = []
        for f in layer.fields[:10]:
            field_info.append(f"{f.alias or f.name}: {f.semantic_description}")

        if field_info:
            parts.append(" ".join(field_info))

        return " ".join(filter(None, parts))

    # =========================================================================
    # Search
    # =========================================================================

    async def search_layers(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.3,
    ) -> list[LayerSearchResult]:
        """
        Find layers relevant to a natural language query.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            category: Optional category filter
            min_score: Minimum similarity score

        Returns:
            List of LayerSearchResult sorted by relevance
        """
        results = []

        # Try vector search first
        if self._qdrant_initialized and self._qdrant:
            results = await self._vector_search(query, limit, category, min_score)

        # Fall back to keyword search if no results
        if not results:
            results = self._keyword_search(query, limit, category)

        return results

    async def _vector_search(
        self,
        query: str,
        limit: int,
        category: Optional[str],
        min_score: float,
    ) -> list[LayerSearchResult]:
        """Perform vector similarity search."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)

            # Build filter
            filter_conditions = None
            if category:
                filter_conditions = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category),
                        )
                    ]
                )

            # Search
            search_results = self._qdrant.search(
                collection_name=self.config.qdrant.layer_collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
                query_filter=filter_conditions,
            )

            # Convert to LayerSearchResult
            results = []
            for hit in search_results:
                layer_name = hit.payload.get("name")
                layer = self._layers.get(layer_name)

                if not layer:
                    # Construct minimal layer from Qdrant payload
                    layer = LayerMetadata(
                        name=hit.payload.get("name", "unknown"),
                        display_name=hit.payload.get("display_name", hit.payload.get("name", "Unknown")),
                        description=hit.payload.get("description", ""),
                        layer_url=hit.payload.get("layer_url", ""),
                        arcgis_item_id=hit.payload.get("arcgis_item_id", ""),
                        category=hit.payload.get("category", "other"),
                        semantic_tags=hit.payload.get("semantic_tags", []),
                        fields=[],  # Fields not stored in payload
                    )
                    # Cache it for subsequent requests
                    self._layers[layer_name] = layer

                # Find suggested fields based on query
                suggested_fields = await self._suggest_fields(layer, query)

                results.append(LayerSearchResult(
                    layer=layer,
                    similarity_score=hit.score,
                    match_reason=f"Matched on: {', '.join(hit.payload.get('semantic_tags', [])[:3])}",
                    suggested_fields=suggested_fields,
                ))

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        limit: int,
        category: Optional[str],
    ) -> list[LayerSearchResult]:
        """Perform keyword-based search as fallback."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for layer in self._layers.values():
            # Skip if category doesn't match
            if category and layer.category != category:
                continue

            # Calculate relevance score
            score = 0.0

            # Check name match
            if query_lower in layer.name.lower():
                score += 0.5
            if query_lower in layer.display_name.lower():
                score += 0.4

            # Check tag matches
            for tag in layer.semantic_tags:
                if tag.lower() in query_lower or query_lower in tag.lower():
                    score += 0.2

            # Check description
            if layer.description:
                desc_words = set(layer.description.lower().split())
                overlap = len(query_words & desc_words)
                score += overlap * 0.1

            # Check field names
            for field in layer.fields:
                if query_lower in field.name.lower() or query_lower in (field.alias or "").lower():
                    score += 0.15

            if score > 0:
                results.append(LayerSearchResult(
                    layer=layer,
                    similarity_score=min(score, 1.0),
                    match_reason="Keyword match",
                    suggested_fields=[],
                ))

        # Sort by score and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]

    async def _suggest_fields(self, layer: LayerMetadata, query: str) -> list[str]:
        """Suggest relevant fields for a query."""
        query_lower = query.lower()
        suggested = []

        # Score each field
        field_scores = []
        for field in layer.fields:
            score = 0

            # Check name/alias match
            if query_lower in field.name.lower():
                score += 1.0
            if field.alias and query_lower in field.alias.lower():
                score += 0.8

            # Check description match
            if field.semantic_description:
                if query_lower in field.semantic_description.lower():
                    score += 0.5

            # Check concept match
            for concept in field.related_concepts:
                if concept in query_lower or query_lower in concept:
                    score += 0.3

            if score > 0:
                field_scores.append((field.name, score))

        # Sort and return top fields
        field_scores.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in field_scores[:5]]

    # =========================================================================
    # Layer Access
    # =========================================================================

    async def get_layer(self, name: str) -> Optional[LayerMetadata]:
        """Get a layer by name."""
        return self._layers.get(name)

    async def get_layer_by_item_id(self, item_id: str) -> Optional[LayerMetadata]:
        """Get a layer by ArcGIS item ID."""
        layer_name = self._layers_by_item_id.get(item_id)
        if layer_name:
            return self._layers.get(layer_name)
        return None

    async def get_all_layers(self) -> list[LayerMetadata]:
        """Get all cached layers."""
        return list(self._layers.values())

    async def get_layers_by_category(self, category: str) -> list[LayerMetadata]:
        """Get all layers in a category."""
        return [l for l in self._layers.values() if l.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return list(set(l.category for l in self._layers.values() if l.category))

    # =========================================================================
    # Context Generation
    # =========================================================================

    def get_ai_context(self, layers: Optional[list[LayerMetadata]] = None) -> str:
        """
        Generate context string for AI agent prompts.

        Args:
            layers: Specific layers to include (all if None)

        Returns:
            Formatted context string
        """
        layers = layers or list(self._layers.values())

        if not layers:
            return "No layers currently available."

        parts = ["## Available Data Layers\n"]

        # Group by category
        by_category = {}
        for layer in layers:
            cat = layer.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(layer)

        for category, category_layers in by_category.items():
            parts.append(f"\n### {category.title()}\n")
            for layer in category_layers:
                parts.append(layer.to_context_string())
                parts.append("")

        return "\n".join(parts)


# =============================================================================
# Factory Functions
# =============================================================================

_catalog_service: Optional[LayerCatalogService] = None


async def get_layer_catalog_service(
    force_new: bool = False,
) -> LayerCatalogService:
    """Get or create the global layer catalog service."""
    global _catalog_service

    if _catalog_service is None or force_new:
        _catalog_service = LayerCatalogService()
        await _catalog_service.initialize()

    return _catalog_service


def get_layer_catalog_service_sync() -> LayerCatalogService:
    """Get the layer catalog service (sync version, must be initialized)."""
    global _catalog_service

    if _catalog_service is None:
        _catalog_service = LayerCatalogService()

    return _catalog_service
