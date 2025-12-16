from typing import Optional, Any
from typing_extensions import override
import logging

import qdrant_client

from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval
from google.adk.tools.tool_context import ToolContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config.config import (
    QDRANT_DB_PORT,
    QDRANT_DB_URL,
    VERTEX_RAG_CORPUS_ID,
)
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class QdrantRetrieval(LlamaIndexRetrieval):

    def __init__(
        self,
        *,
        name: str,
        description: str,
        collection_name: str,
        embedding_model: Optional[BaseEmbedding] = None,
        top_k: int = 20,
    ):
        """Tool to which uses qdrant as retriever
        Args:
            name: str -> name of the tool
            description: str -> Description of the tool to use in LLM calls
            collection_name: str -> Name of the Qdrant collection
            embedding_model: Optional[BaseEmbedding] -> Embedding model to use
            top_k: int -> Number of top results to retrieve (default: 20)
        Raises:
            ValueError -> when given collection does not exists in qdrant.
        """

        self._q_client = qdrant_client.QdrantClient(
            url=QDRANT_DB_URL, port=QDRANT_DB_PORT
        )

        if not self._q_client.collection_exists(collection_name=collection_name):
            raise ValueError(f"collection '{collection_name}' does not exists.")

        # Configure QdrantVectorStore to read 'page_content' field instead of 'text'
        # This is compatible with LangChain-indexed collections
        self._vector_store = QdrantVectorStore(
            client=self._q_client,
            collection_name=collection_name,
            enable_hybrid=False,
            text_key="page_content",  # Map to your LangChain field
            metadata_key="metadata",  # Map to your metadata field
        )
        self._retriever = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store, embed_model=embedding_model
        ).as_retriever(similarity_top_k=top_k)

        super().__init__(name=name, description=description, retriever=self._retriever)

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        try:
            results = self.retriever.retrieve(args["query"])

            if not results:
                logger.warning("No results found for query")
                return "No relevant information found in the knowledge base."

            # Get the first result's text
            response = results[0].text if results[0].text else results[0].get_content()

            if not response:
                logger.warning("Retrieved result has no text content")
                return "Retrieved data has no text content."

            return response

        except Exception as e:
            logger.error(f"Error retrieving from Qdrant: {e}", exc_info=True)
            return "Unable to retrieve information from the knowledge base."


async def retrieve_tapestry_insights_vertex_rag(query: str) -> str:
    """
    Retrieve tapestry segment insights using Vertex AI RAG Engine.

    Args:
        query: The query string to search for tapestry segment information

    Returns:
        Complete text response from Vertex AI RAG Engine
    """
    try:
        logger.info(
            f"Retrieving tapestry insights via Vertex AI RAG for query: {query[:100]}..."
        )

        # Create client with vertexai=True (uses existing credentials via ADC)
        client = genai.Client(vertexai=True)

        # Parse RAG corpus ID from config
        # Format: projects/PROJECT_ID/locations/LOCATION/ragCorpora/CORPUS_ID
        corpus_parts = VERTEX_RAG_CORPUS_ID.split("/")
        if len(corpus_parts) < 6 or corpus_parts[4] != "ragCorpora":
            raise ValueError(f"Invalid RAG corpus ID format: {VERTEX_RAG_CORPUS_ID}")

        # Configure contents with user query
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]

        # Configure retrieval tool with Vertex RAG Store
        tools = [
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_rag_store=types.VertexRagStore(
                        rag_resources=[
                            types.VertexRagStoreRagResource(
                                rag_corpus=VERTEX_RAG_CORPUS_ID
                            )
                        ],
                    )
                )
            )
        ]

        # Configure generation settings
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=65535,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
            tools=tools,
            # Note: ThinkingConfig may not be supported in current API version
            # Removed thinking_config as it causes validation errors
        )

        # Stream response and collect text
        model = "gemini-2.5-pro"
        response_text = ""

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content:
                continue

            # Try to get text from chunk.text first (simpler API)
            if hasattr(chunk, "text") and chunk.text:
                response_text += chunk.text
                continue

            # Fallback: extract text from parts
            if chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

        if not response_text:
            logger.warning("No response text received from Vertex AI RAG")
            return "No information found for the query."

        logger.info(f"Successfully retrieved insights: {len(response_text)} characters")
        return response_text.strip()

    except Exception as e:
        logger.error(
            f"Error retrieving tapestry insights via Vertex AI RAG: {e}", exc_info=True
        )
        return (
            f"Unable to retrieve information from Vertex AI RAG Engine. Error: {str(e)}"
        )
