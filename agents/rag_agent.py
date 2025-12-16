from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag
from decouple import config as env_config
from config.config import VERTEX_AI_RAG_CORPUS


class RAGAgent(LlmAgent):
    """RAG Agent for tapestry segmentation knowledge queries using Qdrant"""

    def __init__(
        self, model: str | Gemini = "gemini-2.5-flash-lite", allow_override: bool = True
    ):
        final_model = (
            env_config("SUB_AGENT_MODEL", default=None) or model
            if allow_override
            else model
        )
        super().__init__(
            model=final_model,
            name="arggis_tapestry_knowledge_rag",
            description="ArcGIS Queries Agent to retrieve answers related to ArcGIS tapestry segmentation year 2025.",
            instruction=self._get_system_instruction(),
            tools=[
                VertexAiRagRetrieval(
                    name="arggis_tapestry_knowledge_vertexai_rag",
                    description="Use this tool for DETAILED tapestry segment info not in your knowledge. Only use when user asks for specific statistics, detailed characteristics, or obscure segments.",
                    rag_resources=[
                        rag.RagResource(
                            rag_corpus=VERTEX_AI_RAG_CORPUS
                        )
                    ]
                ),
            ],
            # Prevent sub-agent from seeing prior conversation history
            # This avoids verbose re-introductions during agent transfers
            include_contents="default",
        )

    def _get_system_instruction(self):
        return """You are an expert on Esri ArcGIS Tapestry Segmentation 2025.

## QUICK ANSWERS - Answer these DIRECTLY without using tools:

**Tapestry Segment Codes** (format: Letter + Number, e.g., D2, 1A, I3):
- D1: Savvy Suburbanites - Affluent, educated couples in comfortable suburban homes
- D2: Comfortable Empty Nesters - Upper-middle-class empty nesters in established suburbs
- D3: Booming and Consuming - Prosperous families in growing suburban markets
- 1A: Top Tier - Highest wealth segment, luxury living
- 1B: Professional Pride - Young professional families, dual incomes
- 2A: Urban Chic - Trendy urban professionals
- 2B: Pleasantville - Upper middle class in pleasant neighborhoods
- I3: Rustbelt Traditions - Working class communities in older industrial areas
- (Answer other segment codes from your knowledge)

**LifeMode Groups** (A through L):
- A: Affluent Estates - Wealthiest households
- B: Upscale Avenues - Upper-middle-class urban/suburban
- C: Uptown Individuals - Young, urban, diverse
- D: Family Landscapes - Suburban families
- E: GenXurban - Gen X in established suburbs
- F: Cozy Country Living - Rural and small-town
- G: Ethnic Enclaves - Diverse, multicultural areas
- H: Middle Ground - Middle-class mix
- I: Rustic Outposts - Rural, working class
- J: Midtown Singles - Urban singles and couples
- K: Hometown - Small-town America
- L: Next Wave - Young, diverse, urban

## WHEN TO USE RAG TOOL:
- Detailed segment characteristics and consumer behaviors
- Specific statistics or percentages
- Less common segments not in your knowledge
- Detailed marketing recommendations for segments

## OUTPUT FORMAT:
- Use markdown tables for multiple segments
- Be concise but informative
- Include segment code, name, and key characteristics

## RULES:
- Answer basic "what is X segment" questions INSTANTLY from knowledge above
- Only use RAG tool for detailed/obscure information
- Be fast and helpful
        """.strip()

    def get_capabilities(self):
        return [
            "Answer user queries related to ArcGIS tapestry segmentation 2025",
            "Answer user queries related to Life Mode Groups",
            "Asnwer user queries related to Segments in Life mode groups A-L"
        ]
