import json
import logging
from typing import Optional

from google.adk.tools import FunctionTool

from tools.salesforce_tools import salesforce_tools

logger = logging.getLogger(__name__)


# Tool 1: Get All Objects
def get_all_objects() -> str:
    """Retrieve all available Salesforce objects in the organization.
    Returns compact list with name and label only.
    For detailed metadata, use get_all_fields_for_object on specific objects."""
    try:
        result = salesforce_tools.get_all_objects()
        # Use compact JSON (no indent) to reduce payload size for WebSocket frames
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_all_objects tool: {e}")
        return json.dumps({"success": False, "error": str(e)})


get_all_objects_tool = FunctionTool(get_all_objects)


# Tool 2: Get All Fields for Object
def get_all_fields_for_object(object_name: str) -> str:
    """Get comprehensive field information for a specific Salesforce object"""
    try:
        result = salesforce_tools.get_all_fields_for_object(object_name)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_all_fields_for_object tool: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "object_name": object_name}
        )


get_all_fields_for_object_tool = FunctionTool(get_all_fields_for_object)


# Tool 3: Query SOQL
def query_soql(soql_query: str) -> str:
    """Execute SOQL queries against Salesforce"""
    try:
        result = salesforce_tools.query_soql(soql_query)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in query_soql tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})


query_soql_tool = FunctionTool(query_soql)


# Tool 4: Query SOSL
def query_sosl(sosl_query: str) -> str:
    """Execute SOSL searches across multiple Salesforce objects"""
    try:
        result = salesforce_tools.query_sosl(sosl_query)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in query_sosl tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": sosl_query})


query_sosl_tool = FunctionTool(query_sosl)


# Tool 5: Get Record Count
def get_record_count(object_name: str, where_clause: Optional[str] = None) -> str:
    """Count records matching criteria without data retrieval"""
    try:
        result = salesforce_tools.get_record_count(object_name, where_clause)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_record_count tool: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "object_name": object_name}
        )


get_record_count_tool = FunctionTool(get_record_count)


# Tool 6: Query with Pagination
def query_with_pagination(soql_query: str, batch_size: int = 2000) -> str:
    """Handle large result sets with automatic pagination"""
    try:
        result = salesforce_tools.query_with_pagination(soql_query, batch_size)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in query_with_pagination tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})


query_with_pagination_tool = FunctionTool(query_with_pagination)


# Tool 7: Aggregate Query
def aggregate_query(
    object_name: str,
    aggregate_fields: str,
    group_by_fields: Optional[str] = None,
    where_clause: Optional[str] = None,
) -> str:
    """Execute aggregate functions (COUNT, SUM, AVG, MIN, MAX)"""
    try:
        # Parse comma-separated strings to lists
        agg_fields = [f.strip() for f in aggregate_fields.split(",")]
        group_fields = (
            [f.strip() for f in group_by_fields.split(",")] if group_by_fields else None
        )

        result = salesforce_tools.aggregate_query(
            object_name, agg_fields, group_fields, where_clause
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in aggregate_query tool: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "object_name": object_name}
        )


aggregate_query_tool = FunctionTool(aggregate_query)


# Tool 8: Get Object Relationships
def get_object_relationships(object_name: str, relationship_depth: int = 2) -> str:
    """Map parent-child and lookup relationships between objects"""
    try:
        result = salesforce_tools.get_object_relationships(
            object_name, relationship_depth
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in get_object_relationships tool: {e}")
        return json.dumps(
            {"success": False, "error": str(e), "object_name": object_name}
        )


get_object_relationships_tool = FunctionTool(get_object_relationships)


# Tool 9: Export Query to CSV
async def export_query_to_csv(soql_query: str, include_headers: bool = True) -> str:
    """Export query results directly to CSV format"""
    try:
        result = await salesforce_tools.export_query_to_csv(soql_query, include_headers)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in export_query_to_csv tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})


export_query_to_csv_tool = FunctionTool(export_query_to_csv)


# Tool 10: Validate SOQL Syntax
def validate_soql_syntax(soql_query: str) -> str:
    """Validate SOQL syntax before execution"""
    try:
        result = salesforce_tools.validate_soql_syntax(soql_query)
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in validate_soql_syntax tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})


validate_soql_syntax_tool = FunctionTool(validate_soql_syntax)


async def export_query_to_geojson(
    soql_query: str, use_pagination: bool = False, max_records: int = 2000
) -> str:
    """
    Execute SOQL query, geocode address fields, and return GeoJSON FeatureCollection.

    Automatically detects address fields (BillingStreet, ShippingCity, MailingState, etc.),
    geocodes them using ArcGIS, and creates map-ready GeoJSON output.

    IMPORTANT: Query structure
    - DEFAULT: Use FIELDS(ALL) to automatically include all fields including addresses
    - Use pagination for large datasets (>200 records)

    Args:
        soql_query: SOQL query with address fields (use FIELDS(ALL) for automatic detection)
        use_pagination: Set to True for large datasets (automatically fetches all records up to max_records)
        max_records: Maximum records to geocode (default 2000, max 50000 for performance)

    Returns:
        JSON string with GeoJSON FeatureCollection ready for mapping

    Examples:
        # Small dataset (uses LIMIT in query)
        export_query_to_geojson("SELECT FIELDS(ALL) FROM Account LIMIT 200")

        # Large dataset (uses pagination)
        export_query_to_geojson("SELECT FIELDS(ALL) FROM Account WHERE BillingCity != null", use_pagination=True, max_records=5000)

        # Custom fields
        export_query_to_geojson("SELECT FIELDS(ALL) FROM Store__c LIMIT 100")
    """
    try:
        result = await salesforce_tools.export_query_to_geojson(
            soql_query, use_pagination, max_records
        )
        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in export_query_to_geojson tool: {e}")
        return json.dumps({"success": False, "error": str(e), "query": soql_query})


export_query_to_geojson_tool = FunctionTool(export_query_to_geojson)


async def geocode_existing_records(records: str) -> str:
    """
    Geocode already-fetched Salesforce records and return GeoJSON FeatureCollection.

    Use this when records are already available from query_soql or query_with_pagination.
    This avoids double querying and improves performance.

    Args:
        records: List of Salesforce record dictionaries in json string format

    Returns:
        JSON string with GeoJSON FeatureCollection

    Example:
        # After fetching records
        result = query_soql("SELECT FIELDS(ALL) FROM Account LIMIT 100")
        records = result["data"]["records"]
        geojson = geocode_existing_records(records)
    """
    try:
        records_json = json.loads(records)
        result = await salesforce_tools.geocode_existing_records(records_json)

        return json.dumps(result, indent=0)
    except Exception as e:
        logger.error(f"Error in geocode_existing_records: {e}")
        return json.dumps({"success": False, "error": str(e)})


geocode_existing_records_tool = FunctionTool(geocode_existing_records)


# Collection of all Salesforce tools for easy registration
SALESFORCE_ADK_TOOLS = [
    get_all_objects_tool,
    get_all_fields_for_object_tool,
    query_soql_tool,
    query_sosl_tool,
    get_record_count_tool,
    query_with_pagination_tool,
    aggregate_query_tool,
    get_object_relationships_tool,
    export_query_to_csv_tool,
    validate_soql_syntax_tool,
    export_query_to_geojson_tool,
    geocode_existing_records_tool,
]


# Optimized schema generation for Salesforce ADK tools
def _extract_tool_schema(tool):
    """Extract clean schema from a single ADK tool"""
    try:
        declaration = tool._get_declaration()
        schema = {
            "name": declaration.name,
            "description": declaration.description,
            "parameters": {},
        }

        # Extract parameters if they exist
        if hasattr(declaration, "parameters") and declaration.parameters:
            params = declaration.parameters
            schema["parameters"] = {
                "properties": {},
                "required": getattr(params, "required", []),
            }

            # Extract property schemas
            if hasattr(params, "properties"):
                for prop_name, prop_schema in params.properties.items():
                    # Extract type safely
                    prop_type = "string"  # default
                    if hasattr(prop_schema, "type"):
                        type_str = str(prop_schema.type).lower()
                        if "string" in type_str:
                            prop_type = "string"
                        elif "integer" in type_str or "int" in type_str:
                            prop_type = "integer"
                        elif "boolean" in type_str or "bool" in type_str:
                            prop_type = "boolean"
                        elif "object" in type_str:
                            prop_type = "object"

                    schema["parameters"]["properties"][prop_name] = {
                        "type": prop_type,
                        "nullable": getattr(prop_schema, "nullable", False),
                    }

        return schema

    except Exception as e:
        logger.warning(f"Failed to extract schema for tool: {e}")
        return {
            "name": getattr(tool, "__name__", "unknown_tool"),
            "description": "Tool schema extraction failed",
            "parameters": {},
        }


# Generate optimized tool schemas
SALESFORCE_ADK_TOOLS_PROMPT_DETAILS = [
    _extract_tool_schema(tool) for tool in SALESFORCE_ADK_TOOLS
]

logger.info(f"Initialized {len(SALESFORCE_ADK_TOOLS)} Salesforce ADK tools")
