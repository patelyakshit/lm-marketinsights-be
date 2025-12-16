"""
Spatial-RAG ADK Tools

Tools for performing hybrid spatial-RAG searches that combine
location-based queries with semantic RAG lookups.
"""

import logging
from typing import Optional

from google.adk.tools import FunctionTool

from utils.spatial_rag import (
    spatial_rag_search,
    SpatialRAGSearch,
    SpatialContext,
    SpatialQueryType,
)

logger = logging.getLogger(__name__)

# Store GIS executor reference for tool use
_gis_executor = None


def set_gis_executor(executor):
    """Set the GIS executor for spatial-RAG tools."""
    global _gis_executor
    _gis_executor = executor


async def get_location_intelligence(
    location: str,
    include_demographics: bool = True,
    include_tapestry: bool = True,
    include_poi: bool = False
) -> str:
    """
    Get comprehensive intelligence for a location combining spatial data and RAG insights.

    This tool geocodes the location, fetches demographics and tapestry data,
    and optionally retrieves relevant segment insights from the knowledge base.

    Args:
        location: Address, city, ZIP code, or location description (e.g., "Downtown Dallas, TX" or "90210")
        include_demographics: Include demographic statistics (population, income, etc.)
        include_tapestry: Include tapestry segmentation data
        include_poi: Include nearby points of interest

    Returns:
        Formatted location intelligence with spatial data and insights

    Example:
        result = await get_location_intelligence("1600 Pennsylvania Ave, Washington DC")
        # Returns demographics, tapestry segment, and relevant insights
    """
    try:
        from tools.gis_tools import gis_executor

        # Use the global executor or passed one
        executor = _gis_executor or gis_executor

        # Perform spatial-RAG search
        result = await spatial_rag_search(
            query=f"location intelligence for {location}",
            gis_executor=executor,
            include_demographics=include_demographics,
            include_poi=include_poi,
            include_tapestry=include_tapestry
        )

        if not result.success:
            return f"Could not find location: {location}. Error: {result.error or 'Unknown'}"

        # Build response
        response_parts = []

        ctx = result.spatial_context
        if ctx.address:
            response_parts.append(f"**Location:** {ctx.address}")
            response_parts.append(f"**Coordinates:** ({ctx.latitude:.6f}, {ctx.longitude:.6f})")

        if result.combined_context:
            response_parts.append("")
            response_parts.append(result.combined_context)

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error getting location intelligence: {e}")
        return f"Error retrieving location intelligence: {str(e)}"


async def analyze_location_for_marketing(
    location: str,
    business_type: Optional[str] = None
) -> str:
    """
    Analyze a location for marketing purposes, including consumer segments and demographics.

    Combines spatial data with tapestry insights to provide marketing-relevant intelligence
    for targeting specific locations.

    Args:
        location: Address, city, ZIP code, or area description
        business_type: Optional business type for more targeted insights (e.g., "restaurant", "retail")

    Returns:
        Marketing analysis with consumer segments, demographics, and targeting recommendations

    Example:
        result = await analyze_location_for_marketing("Beverly Hills, CA", "luxury retail")
    """
    try:
        from tools.gis_tools import gis_executor
        from tools.rag_tools import retrieve_tapestry_insights_vertex_rag

        executor = _gis_executor or gis_executor

        # First get spatial data
        result = await spatial_rag_search(
            query=f"marketing analysis for {location}",
            gis_executor=executor,
            include_demographics=True,
            include_poi=True,
            include_tapestry=True
        )

        if not result.success:
            return f"Could not analyze location: {location}. Error: {result.error or 'Unknown'}"

        response_parts = []
        response_parts.append(f"## Marketing Analysis: {result.spatial_context.address or location}")
        response_parts.append("")

        # Add demographics
        if "demographics" in result.spatial_data:
            demo = result.spatial_data["demographics"]
            features = demo.get("features", [])
            if features:
                attrs = features[0].get("attributes", {})
                response_parts.append("### Demographics")

                income = attrs.get("MEDHINC_CY")
                if income:
                    response_parts.append(f"- Median Household Income: ${income:,.0f}")

                pop = attrs.get("TOTPOP_CY")
                if pop:
                    response_parts.append(f"- Total Population: {pop:,.0f}")

                hh = attrs.get("TOTHH_CY")
                if hh:
                    response_parts.append(f"- Total Households: {hh:,.0f}")

                response_parts.append("")

        # Add tapestry segment
        if "tapestry" in result.spatial_data:
            tap = result.spatial_data["tapestry"]
            features = tap.get("features", [])
            if features:
                attrs = features[0].get("attributes", {})
                seg_name = attrs.get("TSEGNAME", "")
                life_name = attrs.get("TLIFENAME", "")

                if seg_name:
                    response_parts.append("### Consumer Segment")
                    response_parts.append(f"- **Tapestry Segment:** {seg_name}")
                    if life_name:
                        response_parts.append(f"- **LifeMode Group:** {life_name}")
                    response_parts.append("")

                    # Get detailed segment insights from RAG
                    try:
                        rag_query = f"What are the characteristics, preferences, and marketing strategies for the {seg_name} tapestry segment?"
                        if business_type:
                            rag_query += f" Focus on {business_type} businesses."

                        insights = await retrieve_tapestry_insights_vertex_rag(rag_query)
                        if insights and "Unable to retrieve" not in insights:
                            response_parts.append("### Segment Insights")
                            response_parts.append(insights)
                            response_parts.append("")
                    except Exception as e:
                        logger.warning(f"Could not retrieve RAG insights: {e}")

        # Add POI summary
        if "poi" in result.spatial_data:
            poi = result.spatial_data["poi"]
            results = poi.get("results", [])
            if results:
                response_parts.append("### Nearby Competition")
                categories = {}
                for place in results[:20]:
                    cat = place.get("categories", [{}])[0].get("label", "Other")
                    categories[cat] = categories.get(cat, 0) + 1

                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                    response_parts.append(f"- {cat}: {count} locations")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error analyzing location for marketing: {e}")
        return f"Error analyzing location: {str(e)}"


async def compare_locations(
    location1: str,
    location2: str
) -> str:
    """
    Compare two locations side-by-side for demographics and consumer segments.

    Useful for site selection, market comparison, and expansion planning.

    Args:
        location1: First location (address, city, or ZIP)
        location2: Second location (address, city, or ZIP)

    Returns:
        Side-by-side comparison of both locations

    Example:
        result = await compare_locations("Austin, TX", "Denver, CO")
    """
    try:
        from tools.gis_tools import gis_executor

        executor = _gis_executor or gis_executor
        search = SpatialRAGSearch(executor)

        # Fetch data for both locations in parallel
        import asyncio
        results = await asyncio.gather(
            spatial_rag_search(f"data for {location1}", executor, True, False, True),
            spatial_rag_search(f"data for {location2}", executor, True, False, True),
            return_exceptions=True
        )

        result1, result2 = results

        if isinstance(result1, Exception) or not result1.success:
            return f"Could not analyze location 1: {location1}"
        if isinstance(result2, Exception) or not result2.success:
            return f"Could not analyze location 2: {location2}"

        # Build comparison table
        response_parts = []
        response_parts.append("## Location Comparison")
        response_parts.append("")
        response_parts.append(f"| Metric | {result1.spatial_context.address or location1} | {result2.spatial_context.address or location2} |")
        response_parts.append("|--------|-------|-------|")

        # Compare demographics
        def get_demo_value(result, field):
            demo = result.spatial_data.get("demographics", {})
            features = demo.get("features", [])
            if features:
                return features[0].get("attributes", {}).get(field)
            return None

        def get_tapestry_value(result, field):
            tap = result.spatial_data.get("tapestry", {})
            features = tap.get("features", [])
            if features:
                return features[0].get("attributes", {}).get(field)
            return None

        # Population
        pop1 = get_demo_value(result1, "TOTPOP_CY")
        pop2 = get_demo_value(result2, "TOTPOP_CY")
        response_parts.append(f"| Population | {pop1:,.0f if pop1 else 'N/A'} | {pop2:,.0f if pop2 else 'N/A'} |")

        # Median Income
        inc1 = get_demo_value(result1, "MEDHINC_CY")
        inc2 = get_demo_value(result2, "MEDHINC_CY")
        response_parts.append(f"| Median Income | ${inc1:,.0f if inc1 else 'N/A'} | ${inc2:,.0f if inc2 else 'N/A'} |")

        # Households
        hh1 = get_demo_value(result1, "TOTHH_CY")
        hh2 = get_demo_value(result2, "TOTHH_CY")
        response_parts.append(f"| Households | {hh1:,.0f if hh1 else 'N/A'} | {hh2:,.0f if hh2 else 'N/A'} |")

        # Tapestry Segment
        seg1 = get_tapestry_value(result1, "TSEGNAME")
        seg2 = get_tapestry_value(result2, "TSEGNAME")
        response_parts.append(f"| Tapestry Segment | {seg1 or 'N/A'} | {seg2 or 'N/A'} |")

        # LifeMode Group
        life1 = get_tapestry_value(result1, "TLIFENAME")
        life2 = get_tapestry_value(result2, "TLIFENAME")
        response_parts.append(f"| LifeMode Group | {life1 or 'N/A'} | {life2 or 'N/A'} |")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error comparing locations: {e}")
        return f"Error comparing locations: {str(e)}"


# Create ADK FunctionTools
get_location_intelligence_tool = FunctionTool(get_location_intelligence)
analyze_location_for_marketing_tool = FunctionTool(analyze_location_for_marketing)
compare_locations_tool = FunctionTool(compare_locations)

# Export all tools
SPATIAL_RAG_TOOLS = [
    get_location_intelligence_tool,
    analyze_location_for_marketing_tool,
    compare_locations_tool,
]
