import json
from typing import List, Dict


class LLMPromptFormatter:
    """Converts JSON tool definitions to LLM-friendly prompt formats"""

    def __init__(self, json_data):
        if isinstance(json_data, str):
            self.tools = json.loads(json_data)
        else:
            self.tools = json_data

    def to_system_prompt(self) -> str:
        """Generate a system prompt format for tool definitions"""
        output = []
        output.append("You have access to the following tools:")
        output.append("")

        for tool in self.tools:
            output.append(f"## {tool['name']}")
            output.append(f"**Purpose:** {tool['description']}")

            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_fields = set(parameters.get("required", []))

            if properties:
                output.append("**Parameters:**")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    is_required = param_name in required_fields
                    nullable = param_info.get("nullable", False)

                    requirement = "REQUIRED" if is_required else "OPTIONAL"
                    nullable_info = " (can be null)" if nullable else ""

                    output.append(
                        f"- {param_name} ({param_type}): {requirement}{nullable_info}"
                    )
            else:
                output.append("**Parameters:** None")

            output.append("")

        output.append(
            "Always use the exact tool names and parameter names as specified above."
        )
        return "\n".join(output)

    def to_function_calling_format(self) -> str:
        """Generate function calling instructions for LLMs"""
        output = []
        output.append("# Available Functions")
        output.append("")
        output.append(
            "You can call the following functions to help answer user queries:"
        )
        output.append("")

        for tool in self.tools:
            # Function signature
            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_fields = set(parameters.get("required", []))

            # Create function signature
            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "str")
                if param_name in required_fields:
                    params.append(f"{param_name}: {param_type}")
                else:
                    params.append(f"{param_name}: Optional[{param_type}] = None")

            signature = f"def {tool['name']}({', '.join(params)}):"

            output.append(f"```python")
            output.append(signature)
            output.append(f'    """{tool["description"]}"""')
            output.append("    pass")
            output.append("```")
            output.append("")

        output.append(
            "To use these functions, call them with the appropriate parameters based on the user's request."
        )
        return "\n".join(output)

    def to_xml_schema(self) -> str:
        """Generate XML schema format for tool definitions"""
        output = []
        output.append("<tools>")

        for tool in self.tools:
            output.append(f'  <tool name="{tool["name"]}">')
            output.append(f'    <description>{tool["description"]}</description>')

            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_fields = set(parameters.get("required", []))

            if properties:
                output.append("    <parameters>")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    is_required = str(param_name in required_fields).lower()
                    nullable = str(param_info.get("nullable", False)).lower()

                    output.append(
                        f'      <parameter name="{param_name}" type="{param_type}" required="{is_required}" nullable="{nullable}" />'
                    )
                output.append("    </parameters>")
            else:
                output.append("    <parameters>None</parameters>")

            output.append("  </tool>")

        output.append("</tools>")
        return "\n".join(output)

    def to_usage_examples(self) -> str:
        """Generate usage examples for each tool"""
        output = []
        output.append("# Tool Usage Examples")
        output.append("")
        output.append("Here are examples of how to use each available tool:")
        output.append("")

        for tool in self.tools:
            output.append(f"## {tool['name']}")
            output.append(f"*{tool['description']}*")
            output.append("")

            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_fields = set(parameters.get("required", []))

            if properties:
                # Generate realistic examples
                examples = self._generate_examples(
                    tool["name"], properties, required_fields
                )

                output.append("**Examples:**")
                for i, example in enumerate(examples, 1):
                    output.append(f"{i}. `{example}`")

                output.append("")
                output.append("**Required parameters:**")
                req_params = [p for p in properties.keys() if p in required_fields]
                if req_params:
                    for param in req_params:
                        output.append(f"- {param}")
                else:
                    output.append("- None")

                opt_params = [p for p in properties.keys() if p not in required_fields]
                if opt_params:
                    output.append("")
                    output.append("**Optional parameters:**")
                    for param in opt_params:
                        output.append(f"- {param}")
            else:
                output.append("**Example:**")
                output.append(f"1. `{tool['name']}()`")
                output.append("")
                output.append("**Parameters:** None required")

            output.append("")

        return "\n".join(output)

    def to_decision_tree(self) -> str:
        """Generate a decision tree format for tool selection"""
        output = []
        output.append("# Tool Selection Guide")
        output.append("")
        output.append("Use this guide to choose the right tool for your task:")
        output.append("")

        # Categorize tools
        categories = {
            "query": [],
            "object_info": [],
            "relationship": [],
            "export": [],
            "utility": [],
        }

        for tool in self.tools:
            name = tool["name"].lower()
            desc = tool["description"].lower()

            if "query" in name or "soql" in name or "sosl" in name:
                categories["query"].append(tool)
            elif "object" in name or "field" in name:
                categories["object_info"].append(tool)
            elif "relationship" in name:
                categories["relationship"].append(tool)
            elif "export" in name or "csv" in name:
                categories["export"].append(tool)
            else:
                categories["utility"].append(tool)

        category_descriptions = {
            "query": "When you need to retrieve or search data",
            "object_info": "When you need information about Salesforce objects or fields",
            "relationship": "When you need to understand object relationships",
            "export": "When you need to export data to files",
            "utility": "For validation, counting, or other utility functions",
        }

        for category, tools in categories.items():
            if tools:
                output.append(f"## {category_descriptions[category]}:")
                for tool in tools:
                    parameters = tool.get("parameters", {})
                    properties = parameters.get("properties", {})
                    required_count = len(
                        [
                            p
                            for p in properties.keys()
                            if p in tool.get("parameters", {}).get("required", [])
                        ]
                    )

                    complexity = "Simple" if required_count <= 1 else "Complex"
                    output.append(
                        f"- **{tool['name']}** ({complexity}): {tool['description']}"
                    )
                output.append("")

        return "\n".join(output)

    def to_conversational_format(self) -> str:
        """Generate a conversational format explaining tools naturally"""
        output = []
        output.append("# Available Salesforce Tools")
        output.append("")
        output.append("I have access to several Salesforce tools that can help you:")
        output.append("")

        for i, tool in enumerate(self.tools, 1):
            output.append(f"**{i}. {tool['name'].replace('_', ' ').title()}**")
            output.append(f"I can {tool['description'].lower()}")

            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_fields = set(parameters.get("required", []))

            if properties:
                required_params = [p for p in properties.keys() if p in required_fields]
                optional_params = [
                    p for p in properties.keys() if p not in required_fields
                ]

                if required_params:
                    req_list = ", ".join(f"'{p}'" for p in required_params)
                    output.append(f"To use this, you'll need to provide: {req_list}")

                if optional_params:
                    opt_list = ", ".join(f"'{p}'" for p in optional_params)
                    output.append(f"Optionally, you can also specify: {opt_list}")
            else:
                output.append("This tool doesn't require any parameters.")

            output.append("")

        output.append(
            "Just tell me what you'd like to do with Salesforce data, and I'll use the appropriate tool!"
        )
        return "\n".join(output)

    def _generate_examples(
        self, tool_name: str, properties: Dict, required_fields: set
    ) -> List[str]:
        """Generate realistic examples for tool usage"""
        examples = []

        # Create base example with all required parameters
        base_params = []
        for param_name, param_info in properties.items():
            if param_name in required_fields:
                param_type = param_info.get("type", "string")
                example_value = self._get_example_value(param_name, param_type)
                base_params.append(
                    f'{param_name}="{example_value}"'
                    if param_type == "string"
                    else f"{param_name}={example_value}"
                )

        if base_params:
            examples.append(f"{tool_name}({', '.join(base_params)})")
        else:
            examples.append(f"{tool_name}()")

        # Create example with optional parameters if they exist
        optional_params = [p for p in properties.keys() if p not in required_fields]
        if optional_params and base_params:
            # Add one optional parameter
            opt_param = optional_params[0]
            opt_info = properties[opt_param]
            opt_type = opt_info.get("type", "string")
            opt_value = self._get_example_value(opt_param, opt_type)

            if opt_type == "string":
                all_params = base_params + [f'{opt_param}="{opt_value}"']
            else:
                all_params = base_params + [f"{opt_param}={opt_value}"]

            examples.append(f"{tool_name}({', '.join(all_params)})")

        return examples

    def _get_example_value(self, param_name: str, param_type: str) -> str:
        """Get realistic example values for parameters"""
        examples = {
            "soql_query": "SELECT Id, Name FROM Account LIMIT 10",
            "sosl_query": "FIND {Acme} IN ALL FIELDS RETURNING Account(Id, Name)",
            "object_name": "Account",
            "where_clause": "CreatedDate = TODAY",
            "batch_size": "1000",
            "aggregate_fields": "COUNT(Id), SUM(Amount)",
            "group_by_fields": "Type, Industry",
            "relationship_depth": "2",
            "file_path": "/path/to/export.csv",
            "include_headers": "True",
        }

        if param_name in examples:
            return examples[param_name]

        # Fallback based on type
        if param_type == "string":
            return "example_value"
        elif param_type == "integer":
            return "100"
        elif param_type == "boolean":
            return "True"
        else:
            return "example_value"


def convert_to_llm_prompt(json_data, format_type="system_prompt"):
    """
    Convert JSON tool definitions to LLM-friendly formats.

    Args:
        json_data: JSON string or parsed data containing tool definitions
        format_type: Type of format to generate
            - "system_prompt": System prompt format
            - "function_calling": Function calling format with signatures
            - "xml_schema": XML schema format
            - "usage_examples": Detailed usage examples
            - "decision_tree": Decision tree for tool selection
            - "conversational": Natural conversational format

    Returns:
        str: Formatted prompt text
    """
    formatter = LLMPromptFormatter(json_data)

    formats = {
        "system_prompt": formatter.to_system_prompt,
        "function_calling": formatter.to_function_calling_format,
        "xml_schema": formatter.to_xml_schema,
        "usage_examples": formatter.to_usage_examples,
        "decision_tree": formatter.to_decision_tree,
        "conversational": formatter.to_conversational_format,
    }

    if format_type not in formats:
        raise ValueError(
            f"Unknown format_type: {format_type}. Available: {list(formats.keys())}"
        )

    return formats[format_type]()
