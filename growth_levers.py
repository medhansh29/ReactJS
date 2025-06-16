import json
import os
from typing import List, Dict, Any, Optional
import uuid
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# --- Configuration ---
# Your OpenAI API Key.
API_KEY = os.getenv('OPENAI_API_KEY', None)
# Directory where your product_data.json, customer_data.json, and campaign_performance.json are located.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_store')

# File paths for input data and campaign strategy output
PRODUCT_DATA_PATH = os.path.join(DATA_DIR, 'Product_Data.json')
CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, 'Customer_Data.json')
CAMPAIGN_PERFORMANCE_PATH = os.path.join(DATA_DIR, 'Campaign_Performance.json')
CAMPAIGN_STRATEGY_PATH = os.path.join(os.path.dirname(__file__), 'Campaign_Strategy.json')

# --- Pydantic Models for Structured Output ---

class GrowthLever(BaseModel):
    """Represents a suggested growth lever with its type, details, and rationale."""
    type: str = Field(description="The type of growth lever (e.g., 'Marketing Channel Expansion', 'Product Feature Enhancement').")
    details: str = Field(description="Specific details and actions for implementing this growth lever.")
    rationale: str = Field(description="Explanation of why this growth lever is suitable, referencing audience types and other data.")
    exact_discount_percentage: Optional[float] = Field(
        default=None,
        description="The exact discount percentage (e.g., 10.5 for 10.5%). Set to null if not applicable."
    )

class GrowthLevers(BaseModel):
    """A list of suggested growth levers."""
    growth_levers: List[GrowthLever] = Field(description="A list of distinct growth levers.")

# --- Helper Functions ---

def load_json(file_path: str) -> Any:
    """Loads and parses a JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"Warning: File not found at {file_path}. Returning empty data.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return {}

def save_json(file_path: str, data: Dict) -> None:
    """Saves data to a JSON file with pretty printing."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")

async def call_llm_for_growth_levers(
    prompt_text: str,
    api_key: str,
    all_data_context: Dict,
    current_growth_levers_for_llm: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Calls the OpenAI LLM for growth lever generation/modification."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key) if api_key else None)
    parser = PydanticOutputParser(pydantic_object=GrowthLevers)
    format_instructions = parser.get_format_instructions()

    data_context_str = f"""
Product Data: {json.dumps(all_data_context.get('product_data', []), indent=2)}
Customer Data: {json.dumps(all_data_context.get('customer_data', []), indent=2)}
Campaign Performance Data: {json.dumps(all_data_context.get('campaign_performance', []), indent=2)}
Generated Audience Types: {json.dumps(all_data_context.get('audience_types', []), indent=2)}
    """

    template_variables = {
        "data_context_str": data_context_str,
        "format_instructions_var": format_instructions,
    }

    if current_growth_levers_for_llm:
        full_prompt_template_str = """
        Based on the following data and the current growth lever suggestions, please provide an updated list.
        The user's request for modification is: "{user_modification_prompt}"

        Current Growth Levers:
        {current_growth_levers_json}

        {data_context_str}

        Modify the growth levers based on the user's request. Maintain the JSON array format as specified by the output instructions.
        If a lever needs to be removed, omit it. If a new lever is requested, add it.
        If an existing lever needs its type, details, or rationale changed, update it.
        Ensure the output strictly adheres to the JSON schema, including setting 'exact_discount_percentage' to null if no specific discount is applicable.

        {format_instructions_var}
        """
        template_variables["user_modification_prompt"] = prompt_text
        template_variables["current_growth_levers_json"] = json.dumps(current_growth_levers_for_llm, indent=2)
    else:
        full_prompt_template_str = """
        Based on the provided product data, customer data, campaign performance, and the generated audience types,
        suggest 3-5 distinct growth levers. If the user has a specific idea for a growth lever, incorporate it
        and suggest related ones. For each lever, provide:
        - A concise 'type' (e.g., 'Targeted Ad Campaign', 'Product Bundling', 'Customer Loyalty Program').
        - 'Details' outlining specific actions or strategies.
        - 'Rationale' explaining why this lever is suitable for the business and target audiences,
          referencing the provided data.
        - If the growth lever involves a specific discount, include 'exact_discount_percentage' (e.g., 10.5 for 10.5%). Set to null if not applicable.

        User's specific idea (if any): "{user_initial_prompt}"

        {data_context_str}

        Ensure the output strictly adheres to the JSON schema.

        {format_instructions_var}
        """
        template_variables["user_initial_prompt"] = prompt_text

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [gl.model_dump() for gl in response.growth_levers]
    except Exception as e:
        print(f"Error during LLM growth lever call: {e}")
        return []

async def process_growth_levers(
    user_prompt: str,
    is_modification: bool = False,
    current_growth_levers_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates or modifies growth levers and updates campaign_strategy.json.
    Returns the updated 'Growth Levers' dictionary.
    """
    product_data = load_json(PRODUCT_DATA_PATH)
    customer_data = load_json(CUSTOMER_DATA_PATH)
    campaign_performance = load_json(CAMPAIGN_PERFORMANCE_PATH)
    
    current_campaign_strategy = load_json(CAMPAIGN_STRATEGY_PATH)
    if not isinstance(current_campaign_strategy, dict):
        current_campaign_strategy = {}
    if "Audience types" not in current_campaign_strategy:
        current_campaign_strategy["Audience types"] = {}
    if "Growth Levers" not in current_campaign_strategy:
        current_campaign_strategy["Growth Levers"] = {}

    all_data_context = {
        'product_data': product_data,
        'customer_data': customer_data,
        'campaign_performance': campaign_performance,
        'audience_types': current_campaign_strategy["Audience types"]
    }

    # Convert current_growth_levers_data from Dict[str, Dict] to List[Dict] for LLM if modifying
    growth_levers_for_llm = None
    if is_modification and current_growth_levers_data:
        growth_levers_for_llm = [
            {
                "id": gl_id,
                "type": gl_data["Type"],
                "details": gl_data["Details"],
                "rationale": gl_data["Rationale"],
                "exact_discount_percentage": gl_data.get("exact_discount_percentage")
            }
            for gl_id, gl_data in current_growth_levers_data.items()
        ]

    suggested_growth_levers_raw = await call_llm_for_growth_levers(
        prompt_text=user_prompt,
        api_key=API_KEY or "",
        all_data_context=all_data_context,
        current_growth_levers_for_llm=growth_levers_for_llm
    )

    if suggested_growth_levers_raw:
        new_growth_levers_dict = {}
        for gl in suggested_growth_levers_raw:
            gl_id = gl.get('id', str(uuid.uuid4())[:8]) # Preserve ID if LLM provides it
            new_growth_levers_dict[gl_id] = {
                "Type": gl['type'],
                "Details": gl['details'],
                "Rationale": gl['rationale'],
                "exact_discount_percentage": gl.get('exact_discount_percentage')
            }
        current_campaign_strategy["Growth Levers"] = new_growth_levers_dict
        save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
        return current_campaign_strategy["Growth Levers"]
    return {}

# No if __name__ == "__main__": block here. This script is now a module.
