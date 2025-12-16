"""
Knowledge Graph Service - Manages layer relationships for multi-hop reasoning.

Supports multiple backends:
- NetworkX (default, in-memory, good for development)
- FalkorDB (production, Redis-based, ultra-low latency)
- Neo4j (enterprise, full-featured)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from collections import defaultdict

from .config import get_config, LayerIntelligenceConfig
from .models import (
    RelationshipType,
    GraphNode,
    GraphRelationship,
    ReasoningPath,
    LayerMetadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base Graph Backend
# =============================================================================

class GraphBackend(ABC):
    """Abstract base class for graph backends."""

    @abstractmethod
    async def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        pass

    @abstractmethod
    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """Add a relationship between nodes."""
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphNode, GraphRelationship]]:
        """Get neighboring nodes."""
        pass

    @abstractmethod
    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[ReasoningPath]:
        """Find paths between two nodes."""
        pass

    @abstractmethod
    async def query(self, cypher: str) -> list[dict]:
        """Execute a Cypher-like query."""
        pass


# =============================================================================
# NetworkX Backend (Development)
# =============================================================================

class NetworkXBackend(GraphBackend):
    """NetworkX-based graph backend for development."""

    def __init__(self):
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
            self._nodes: dict[str, GraphNode] = {}
        except ImportError:
            raise ImportError("networkx not installed. Run: pip install networkx")

    async def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        self.graph.add_node(
            node.id,
            node_type=node.node_type,
            name=node.name,
            **node.properties,
        )
        self._nodes[node.id] = node
        return True

    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """Add a relationship between nodes."""
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            relationship_type=relationship.relationship_type.value,
            **relationship.properties,
        )
        return True

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    async def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphNode, GraphRelationship]]:
        """Get neighboring nodes."""
        results = []

        if direction in ["outgoing", "both"]:
            for successor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, successor)
                if relationship_type and edge_data.get("relationship_type") != relationship_type.value:
                    continue

                node = self._nodes.get(successor)
                if node:
                    rel = GraphRelationship(
                        source_id=node_id,
                        target_id=successor,
                        relationship_type=RelationshipType(edge_data.get("relationship_type")),
                        properties=edge_data,
                    )
                    results.append((node, rel))

        if direction in ["incoming", "both"]:
            for predecessor in self.graph.predecessors(node_id):
                edge_data = self.graph.get_edge_data(predecessor, node_id)
                if relationship_type and edge_data.get("relationship_type") != relationship_type.value:
                    continue

                node = self._nodes.get(predecessor)
                if node:
                    rel = GraphRelationship(
                        source_id=predecessor,
                        target_id=node_id,
                        relationship_type=RelationshipType(edge_data.get("relationship_type")),
                        properties=edge_data,
                    )
                    results.append((node, rel))

        return results

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[ReasoningPath]:
        """Find paths between two nodes."""
        import networkx as nx

        paths = []

        try:
            # Find all simple paths up to max_hops
            for path in nx.all_simple_paths(self.graph, start_id, end_id, cutoff=max_hops):
                nodes = []
                relationships = []

                for i, node_id in enumerate(path):
                    node = self._nodes.get(node_id)
                    if node:
                        nodes.append(node)

                    if i > 0:
                        prev_id = path[i - 1]
                        edge_data = self.graph.get_edge_data(prev_id, node_id)
                        if edge_data:
                            rel = GraphRelationship(
                                source_id=prev_id,
                                target_id=node_id,
                                relationship_type=RelationshipType(edge_data.get("relationship_type")),
                            )
                            relationships.append(rel)

                if nodes:
                    reasoning_path = ReasoningPath(
                        nodes=nodes,
                        relationships=relationships,
                        explanation=self._generate_path_explanation(nodes, relationships),
                        score=1.0 / len(path),  # Shorter paths score higher
                    )
                    paths.append(reasoning_path)

        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass

        return sorted(paths, key=lambda p: p.score, reverse=True)

    def _generate_path_explanation(
        self,
        nodes: list[GraphNode],
        relationships: list[GraphRelationship],
    ) -> str:
        """Generate natural language explanation of path."""
        if not nodes:
            return ""

        parts = [nodes[0].name]
        for i, rel in enumerate(relationships):
            parts.append(f"--[{rel.relationship_type.value}]-->")
            if i + 1 < len(nodes):
                parts.append(nodes[i + 1].name)

        return " ".join(parts)

    async def query(self, cypher: str) -> list[dict]:
        """Execute a simple Cypher-like query (limited support)."""
        # NetworkX doesn't support Cypher, return empty
        logger.warning("Cypher queries not supported in NetworkX backend")
        return []


# =============================================================================
# FalkorDB Backend (Production)
# =============================================================================

class FalkorDBBackend(GraphBackend):
    """FalkorDB-based graph backend for production."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "layer_intelligence",
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self._client = None
        self._graph = None

    def _get_client(self):
        """Get or create FalkorDB client."""
        if self._client is None:
            try:
                from falkordb import FalkorDB
                self._client = FalkorDB(host=self.host, port=self.port)
                self._graph = self._client.select_graph(self.graph_name)
            except ImportError:
                raise ImportError("falkordb not installed. Run: pip install falkordb")
        return self._graph

    async def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        graph = self._get_client()

        props_str = ", ".join(
            f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
            for k, v in node.properties.items()
        )

        cypher = f"""
        MERGE (n:{node.node_type} {{id: "{node.id}"}})
        SET n.name = "{node.name}"
        """
        if props_str:
            cypher += f", {props_str}"

        try:
            graph.query(cypher)
            return True
        except Exception as e:
            logger.error(f"FalkorDB add_node error: {e}")
            return False

    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """Add a relationship between nodes."""
        graph = self._get_client()

        cypher = f"""
        MATCH (a {{id: "{relationship.source_id}"}})
        MATCH (b {{id: "{relationship.target_id}"}})
        MERGE (a)-[r:{relationship.relationship_type.value}]->(b)
        """

        try:
            graph.query(cypher)
            return True
        except Exception as e:
            logger.error(f"FalkorDB add_relationship error: {e}")
            return False

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        graph = self._get_client()

        cypher = f'MATCH (n {{id: "{node_id}"}}) RETURN n'

        try:
            result = graph.query(cypher)
            if result.result_set:
                row = result.result_set[0]
                node_data = row[0].properties
                return GraphNode(
                    id=node_id,
                    node_type=node_data.get("node_type", "unknown"),
                    name=node_data.get("name", ""),
                    properties=node_data,
                )
        except Exception as e:
            logger.error(f"FalkorDB get_node error: {e}")

        return None

    async def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing",
    ) -> list[tuple[GraphNode, GraphRelationship]]:
        """Get neighboring nodes."""
        graph = self._get_client()

        rel_pattern = f":{relationship_type.value}" if relationship_type else ""

        if direction == "outgoing":
            cypher = f'MATCH (a {{id: "{node_id}"}})-[r{rel_pattern}]->(b) RETURN b, type(r)'
        elif direction == "incoming":
            cypher = f'MATCH (a {{id: "{node_id}"}})<-[r{rel_pattern}]-(b) RETURN b, type(r)'
        else:
            cypher = f'MATCH (a {{id: "{node_id}"}})-[r{rel_pattern}]-(b) RETURN b, type(r)'

        results = []
        try:
            result = graph.query(cypher)
            for row in result.result_set:
                node_data = row[0].properties
                rel_type = row[1]

                node = GraphNode(
                    id=node_data.get("id", ""),
                    node_type=node_data.get("node_type", "unknown"),
                    name=node_data.get("name", ""),
                    properties=node_data,
                )

                rel = GraphRelationship(
                    source_id=node_id,
                    target_id=node.id,
                    relationship_type=RelationshipType(rel_type),
                )

                results.append((node, rel))

        except Exception as e:
            logger.error(f"FalkorDB get_neighbors error: {e}")

        return results

    async def find_path(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[ReasoningPath]:
        """Find paths between two nodes."""
        graph = self._get_client()

        cypher = f"""
        MATCH path = (a {{id: "{start_id}"}})-[*1..{max_hops}]-(b {{id: "{end_id}"}})
        RETURN path
        LIMIT 5
        """

        paths = []
        try:
            result = graph.query(cypher)
            for row in result.result_set:
                path_data = row[0]
                # Parse path data and create ReasoningPath
                # (FalkorDB returns path as a list of nodes and edges)
                # Implementation depends on FalkorDB response format
                pass

        except Exception as e:
            logger.error(f"FalkorDB find_path error: {e}")

        return paths

    async def query(self, cypher: str) -> list[dict]:
        """Execute a Cypher query."""
        graph = self._get_client()

        try:
            result = graph.query(cypher)
            return [dict(zip(result.header, row)) for row in result.result_set]
        except Exception as e:
            logger.error(f"FalkorDB query error: {e}")
            return []


# =============================================================================
# Knowledge Graph Service
# =============================================================================

class KnowledgeGraphService:
    """
    Main service for managing the knowledge graph.

    Provides:
    - Layer relationship management
    - Multi-hop reasoning
    - Cross-layer analysis suggestions
    - Concept mapping
    """

    def __init__(
        self,
        config: Optional[LayerIntelligenceConfig] = None,
        backend: Optional[GraphBackend] = None,
    ):
        self.config = config or get_config()

        # Initialize backend
        if backend:
            self._backend = backend
        else:
            backend_type = self.config.knowledge_graph.backend
            if backend_type == "falkordb":
                self._backend = FalkorDBBackend(
                    host=self.config.knowledge_graph.falkordb_host,
                    port=self.config.knowledge_graph.falkordb_port,
                    graph_name=self.config.knowledge_graph.falkordb_graph_name,
                )
            else:
                self._backend = NetworkXBackend()

        # Concept cache for quick lookups
        self._concepts: dict[str, GraphNode] = {}

        # Layer node cache
        self._layer_nodes: dict[str, str] = {}  # layer_name -> node_id

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self):
        """Initialize the knowledge graph with domain knowledge."""
        await self._seed_domain_concepts()
        logger.info("KnowledgeGraphService initialized")

    async def _seed_domain_concepts(self):
        """Seed the graph with domain concepts and relationships."""
        # Domain concepts
        concepts = [
            ("rental_market", "Rental Market", "Market for rental properties"),
            ("housing_demand", "Housing Demand", "Demand for housing"),
            ("affordability", "Affordability", "Housing affordability metric"),
            ("consumer_behavior", "Consumer Behavior", "Consumer preferences and behavior"),
            ("demographics", "Demographics", "Population characteristics"),
            ("income_level", "Income Level", "Household income metrics"),
            ("population", "Population", "Population counts and density"),
            ("lifestyle", "Lifestyle", "Consumer lifestyle patterns"),
            ("transportation", "Transportation", "Transportation and accessibility"),
            ("commercial", "Commercial", "Commercial and business activity"),
            ("safety", "Safety", "Safety and crime metrics"),
            ("education", "Education", "Educational facilities and attainment"),
        ]

        for concept_id, name, description in concepts:
            node = GraphNode(
                id=f"concept_{concept_id}",
                node_type="Concept",
                name=name,
                properties={"description": description},
            )
            await self._backend.add_node(node)
            self._concepts[concept_id] = node

        # Concept relationships
        concept_relationships = [
            ("rental_market", RelationshipType.CORRELATES_WITH, "housing_demand"),
            ("rental_market", RelationshipType.CORRELATES_WITH, "affordability"),
            ("affordability", RelationshipType.CORRELATES_WITH, "income_level"),
            ("consumer_behavior", RelationshipType.CORRELATES_WITH, "lifestyle"),
            ("consumer_behavior", RelationshipType.CORRELATES_WITH, "income_level"),
            ("demographics", RelationshipType.CONTAINS, "population"),
            ("demographics", RelationshipType.CONTAINS, "income_level"),
        ]

        for source, rel_type, target in concept_relationships:
            rel = GraphRelationship(
                source_id=f"concept_{source}",
                target_id=f"concept_{target}",
                relationship_type=rel_type,
            )
            await self._backend.add_relationship(rel)

    # =========================================================================
    # Layer Management
    # =========================================================================

    async def add_layer(self, layer: LayerMetadata):
        """Add a layer to the knowledge graph."""
        # Create layer node
        node = GraphNode(
            id=f"layer_{layer.name}",
            node_type="Layer",
            name=layer.display_name,
            properties={
                "layer_name": layer.name,
                "category": layer.category,
                "description": layer.description,
            },
        )
        await self._backend.add_node(node)
        self._layer_nodes[layer.name] = node.id

        # Create field nodes
        for field in layer.fields:
            field_node = GraphNode(
                id=f"field_{layer.name}_{field.name}",
                node_type="Field",
                name=field.alias or field.name,
                properties={
                    "field_name": field.name,
                    "layer_name": layer.name,
                    "field_type": field.field_type,
                    "description": field.semantic_description,
                },
            )
            await self._backend.add_node(field_node)

            # Link field to layer
            rel = GraphRelationship(
                source_id=node.id,
                target_id=field_node.id,
                relationship_type=RelationshipType.HAS_FIELD,
            )
            await self._backend.add_relationship(rel)

            # Link field to concepts
            for concept in field.related_concepts:
                concept_id = f"concept_{concept}"
                if concept in self._concepts:
                    rel = GraphRelationship(
                        source_id=field_node.id,
                        target_id=concept_id,
                        relationship_type=RelationshipType.INDICATES,
                    )
                    await self._backend.add_relationship(rel)

        # Link layer to concepts based on category
        category_concepts = {
            "residential": ["rental_market", "housing_demand"],
            "demographics": ["demographics", "population", "income_level"],
            "commercial": ["commercial", "consumer_behavior"],
            "transportation": ["transportation"],
        }

        for concept in category_concepts.get(layer.category, []):
            if concept in self._concepts:
                rel = GraphRelationship(
                    source_id=node.id,
                    target_id=f"concept_{concept}",
                    relationship_type=RelationshipType.MEASURES,
                )
                await self._backend.add_relationship(rel)

        logger.info(f"Added layer to knowledge graph: {layer.name}")

    async def add_layer_relationship(
        self,
        layer1_name: str,
        relationship_type: RelationshipType,
        layer2_name: str,
    ):
        """Add a relationship between two layers."""
        node1_id = self._layer_nodes.get(layer1_name)
        node2_id = self._layer_nodes.get(layer2_name)

        if not node1_id or not node2_id:
            logger.warning(f"Layer not found: {layer1_name} or {layer2_name}")
            return

        rel = GraphRelationship(
            source_id=node1_id,
            target_id=node2_id,
            relationship_type=relationship_type,
        )
        await self._backend.add_relationship(rel)

    # =========================================================================
    # Reasoning
    # =========================================================================

    async def suggest_analysis_layers(
        self,
        query: str,
        available_layers: Optional[list[LayerMetadata]] = None,
    ) -> list[tuple[str, str, float]]:
        """
        Suggest layers for a complex analysis query.

        Uses graph traversal to find layers that can answer the query
        by connecting concepts.

        Args:
            query: Natural language query
            available_layers: Optional list of available layers

        Returns:
            List of (layer_name, reason, score) tuples
        """
        # Extract concepts from query
        query_concepts = self._extract_concepts_from_query(query)

        # Find layers connected to these concepts
        layer_scores = defaultdict(lambda: {"score": 0.0, "reasons": []})

        for concept in query_concepts:
            concept_id = f"concept_{concept}"

            # Find layers that measure this concept
            neighbors = await self._backend.get_neighbors(
                concept_id,
                direction="incoming",
            )

            for node, rel in neighbors:
                if node.node_type == "Layer":
                    layer_name = node.properties.get("layer_name")
                    if layer_name:
                        layer_scores[layer_name]["score"] += 0.5
                        layer_scores[layer_name]["reasons"].append(
                            f"Measures {concept}"
                        )

                elif node.node_type == "Field":
                    layer_name = node.properties.get("layer_name")
                    if layer_name:
                        layer_scores[layer_name]["score"] += 0.3
                        layer_scores[layer_name]["reasons"].append(
                            f"Has field indicating {concept}"
                        )

        # Also check for correlated concepts
        for concept in query_concepts:
            concept_id = f"concept_{concept}"
            neighbors = await self._backend.get_neighbors(
                concept_id,
                relationship_type=RelationshipType.CORRELATES_WITH,
            )

            for node, rel in neighbors:
                correlated_concept = node.name.lower().replace(" ", "_")

                # Find layers for correlated concept
                corr_neighbors = await self._backend.get_neighbors(
                    node.id,
                    direction="incoming",
                )

                for corr_node, corr_rel in corr_neighbors:
                    if corr_node.node_type == "Layer":
                        layer_name = corr_node.properties.get("layer_name")
                        if layer_name:
                            layer_scores[layer_name]["score"] += 0.2
                            layer_scores[layer_name]["reasons"].append(
                                f"Related via {node.name}"
                            )

        # Convert to sorted list
        results = []
        for layer_name, data in layer_scores.items():
            reason = "; ".join(data["reasons"][:3])
            results.append((layer_name, reason, data["score"]))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]

    def _extract_concepts_from_query(self, query: str) -> list[str]:
        """Extract concept keywords from a query."""
        query_lower = query.lower()

        concept_keywords = {
            "rental_market": ["rent", "rental", "lease", "apartment", "housing cost"],
            "housing_demand": ["housing", "homes", "demand", "supply"],
            "affordability": ["afford", "affordable", "cost", "price", "expensive"],
            "consumer_behavior": ["consumer", "customer", "shopping", "buying", "behavior"],
            "demographics": ["demographic", "population", "people", "residents"],
            "income_level": ["income", "salary", "wage", "earning", "wealth"],
            "population": ["population", "people", "residents", "inhabitants"],
            "lifestyle": ["lifestyle", "living", "segment", "tapestry"],
            "transportation": ["traffic", "transit", "commute", "transportation"],
            "commercial": ["business", "commercial", "retail", "store", "shop"],
            "safety": ["crime", "safety", "safe", "dangerous"],
            "education": ["school", "education", "college", "university"],
        }

        found_concepts = []
        for concept, keywords in concept_keywords.items():
            if any(kw in query_lower for kw in keywords):
                found_concepts.append(concept)

        return found_concepts

    async def find_cross_layer_path(
        self,
        layer1_name: str,
        layer2_name: str,
    ) -> Optional[ReasoningPath]:
        """Find a reasoning path between two layers."""
        node1_id = self._layer_nodes.get(layer1_name)
        node2_id = self._layer_nodes.get(layer2_name)

        if not node1_id or not node2_id:
            return None

        paths = await self._backend.find_path(node1_id, node2_id, max_hops=4)
        return paths[0] if paths else None

    async def explain_layer_relationship(
        self,
        layer1_name: str,
        layer2_name: str,
    ) -> str:
        """Generate explanation of how two layers relate."""
        path = await self.find_cross_layer_path(layer1_name, layer2_name)

        if not path:
            return f"No direct relationship found between {layer1_name} and {layer2_name}."

        return f"Relationship: {path.explanation}"

    # =========================================================================
    # Query
    # =========================================================================

    async def get_related_layers(
        self,
        layer_name: str,
        relationship_type: Optional[RelationshipType] = None,
    ) -> list[str]:
        """Get layers related to the specified layer."""
        node_id = self._layer_nodes.get(layer_name)
        if not node_id:
            return []

        neighbors = await self._backend.get_neighbors(
            node_id,
            relationship_type=relationship_type,
            direction="both",
        )

        related = []
        for node, rel in neighbors:
            if node.node_type == "Layer":
                related.append(node.properties.get("layer_name", ""))

        return [r for r in related if r]

    async def get_layer_concepts(self, layer_name: str) -> list[str]:
        """Get concepts associated with a layer."""
        node_id = self._layer_nodes.get(layer_name)
        if not node_id:
            return []

        neighbors = await self._backend.get_neighbors(
            node_id,
            relationship_type=RelationshipType.MEASURES,
        )

        concepts = []
        for node, rel in neighbors:
            if node.node_type == "Concept":
                concepts.append(node.name)

        return concepts


# =============================================================================
# Factory Functions
# =============================================================================

_kg_service: Optional[KnowledgeGraphService] = None


async def get_knowledge_graph_service(
    force_new: bool = False,
) -> KnowledgeGraphService:
    """Get or create the global knowledge graph service."""
    global _kg_service

    if _kg_service is None or force_new:
        _kg_service = KnowledgeGraphService()
        await _kg_service.initialize()

    return _kg_service


def get_knowledge_graph_service_sync() -> KnowledgeGraphService:
    """Get the knowledge graph service (sync version)."""
    global _kg_service

    if _kg_service is None:
        _kg_service = KnowledgeGraphService()

    return _kg_service
