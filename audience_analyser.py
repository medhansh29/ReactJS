import json
import os
from typing import List, Dict, Any, Optional
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# --- Configuration ---
# Your OpenAI API Key.
API_KEY = os.getenv('OPENAI_API_KEY', None)
# Directory where your product_data.json, customer_data.json, and campaign_performance.json are located.
# This assumes your 'data_store' folder is one level UP from where this script is located.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_store')

# File paths for input and output data
PRODUCT_DATA_PATH = os.path.join(DATA_DIR, 'Product_Data.json')
CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, 'Customer_Data.json')
CAMPAIGN_PERFORMANCE_PATH = os.path.join(DATA_DIR, 'Campaign_Performance.json')
CAMPAIGN_STRATEGY_PATH = os.path.join(os.path.dirname(__file__), 'Campaign_Strategy.json')

# --- Pydantic Models for Structured Output ---

class AudienceType(BaseModel):
    """Represents a suggested audience type with its rationale."""
    type: str = Field(description="A descriptive and concise name for the audience type (e.g., 'Eco-conscious Urban Professionals').")
    rationale: str = Field(description="A detailed explanation of why this audience is suitable, referencing product, customer, and campaign data.")

class AudienceTypes(BaseModel):
    """A list of suggested audience types."""
    audiences: List[AudienceType] = Field(description="A list of distinct audience types.")

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

def calculate_audience_size(audience_type: Dict[str, str], customer_data: List[Dict]) -> int:
    """Heuristically calculates the audience size based on customer data and audience rationale."""
    if not customer_data:
        return 0

    type_lower = audience_type['type'].lower()
    rationale_lower = audience_type['rationale'].lower()
    count = 0

    for customer in customer_data:
        age = customer.get('age', 0)
        if not isinstance(age, (int, float)): age = 0

        match_score = 0
        if ('young' in type_lower or 'young' in rationale_lower) and age < 35: match_score += 1
        if (('adults' in type_lower or 'adults' in rationale_lower or 'professional' in type_lower) and 25 <= age <= 55): match_score += 1
        if (('older' in type_lower or 'older' in rationale_lower or 'senior' in type_lower) and age >= 50): match_score += 1
        
        customer_interests_lower = [i.lower() for i in customer.get('interests', [])]
        interest_keywords = ['tech', 'fitness', 'sustainable', 'eco', 'home', 'coffee', 'music', 'outdoors', 'yoga', 'wellness', 'gaming', 'travel', 'cooking', 'reading', 'photography', 'sports']
        if any(keyword in type_lower or keyword in rationale_lower for keyword in interest_keywords if keyword in customer_interests_lower): match_score += 1

        if 'urban' in type_lower and customer.get('location', '').lower() == 'urban': match_score += 1
        if 'suburban' in type_lower and customer.get('location', '').lower() == 'suburban': match_score += 1
        if 'rural' in type_lower and customer.get('location', '').lower() == 'rural': match_score += 1

        if (('female' in type_lower or 'women' in rationale_lower) and customer.get('gender', '').lower() == 'female'): match_score += 1
        if (('male' in type_lower or 'men' in rationale_lower) and customer.get('gender', '').lower() == 'male'): match_score += 1

        customer_segment_lower = customer.get('CustomerSegment', '').lower()
        if ('high-value' in type_lower or 'high-value' in rationale_lower) and customer_segment_lower == 'high-value': match_score += 1
        if ('loyalist' in type_lower or 'loyal' in rationale_lower) and customer_segment_lower == 'loyalist': match_score += 1
        if ('new customer' in type_lower or 'new user' in rationale_lower) and customer_segment_lower == 'new': match_score += 1
        if ('regular' in type_lower or 'returning' in rationale_lower) and customer_segment_lower == 'regular': match_score += 1

        num_orders = customer.get('NumberOfOrders', 0)
        if ('frequent buyer' in type_lower or 'high orders' in rationale_lower) and num_orders >= 10: match_score += 1

        if match_score >= 2:
            count += 1
    return count

async def call_llm_for_audiences(prompt_text: str, current_audiences_for_llm: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Calls the OpenAI LLM for audience generation/modification."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(API_KEY) if API_KEY else None)
    parser = PydanticOutputParser(pydantic_object=AudienceTypes)
    format_instructions = parser.get_format_instructions()

    product_data_json = json.dumps(load_json(PRODUCT_DATA_PATH), indent=2)
    customer_data_json = json.dumps(load_json(CUSTOMER_DATA_PATH), indent=2)
    campaign_performance_json = json.dumps(load_json(CAMPAIGN_PERFORMANCE_PATH), indent=2)

    template_variables = {
        "format_instructions": format_instructions,
        "product_data": product_data_json,
        "customer_data": customer_data_json,
        "campaign_performance": campaign_performance_json
    }

    if current_audiences_for_llm:
        full_prompt_template = """
        The user wants to modify the current audience suggestions. Here are the current audience types:
        {current_audiences_json}

        The user's request for modification is: "{user_modification_prompt}"

        Based on this, please provide an updated list of audience types.
        Maintain the JSON array format as specified by the output instructions.
        If a type needs to be removed, omit it. If a new type is requested, add it.
        If an existing type needs its rationale or name changed, update it.
        Ensure the output strictly adheres to the JSON schema.
        For each audience, provide a descriptive name for the 'type' field.

        {format_instructions}
        """
        template_variables["current_audiences_json"] = json.dumps(current_audiences_for_llm, indent=2)
        template_variables["user_modification_prompt"] = prompt_text
    else:
        full_prompt_template = """
        Analyze the provided product data, customer data, and past campaign performance below.
        Based on the user's input "{user_initial_prompt}", suggest 3-5 distinct audience types.
        For each type, provide a descriptive name and a detailed rationale explaining why this audience is suitable,
        referencing the provided data. The output must strictly adhere to the JSON schema below.

        Product Data:
        {product_data}

        Customer Data:
        {customer_data}

        Campaign Performance Data:
        {campaign_performance}

        {format_instructions}
        """
        template_variables["user_initial_prompt"] = prompt_text

    prompt = ChatPromptTemplate.from_template(full_prompt_template)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [aud.model_dump() for aud in response.audiences]
    except Exception as e:
        print(f"Error during LLM audience call: {e}")
        return []

async def process_audiences(
    user_prompt: str,
    is_modification: bool = False,
    current_audiences_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates or modifies audience types and updates campaign_strategy.json.
    Returns the updated 'Audience types' dictionary.
    """
    customer_data = load_json(CUSTOMER_DATA_PATH)
    
    # Load existing strategy to preserve other sections
    current_campaign_strategy = load_json(CAMPAIGN_STRATEGY_PATH)
    if not isinstance(current_campaign_strategy, dict):
        current_campaign_strategy = {}
    if "Audience types" not in current_campaign_strategy:
        current_campaign_strategy["Audience types"] = {}
    
    # Convert current_audiences_data from Dict[str, Dict] to List[Dict] for LLM if modifying
    audiences_for_llm = None
    if is_modification and current_audiences_data:
        audiences_for_llm = [
            {"type": name, "rationale": data["Rationale"]}
            for name, data in current_audiences_data.items()
        ]

    suggested_audiences_raw = await call_llm_for_audiences(
        prompt_text=user_prompt,
        current_audiences_for_llm=audiences_for_llm
    )

    if suggested_audiences_raw:
        current_campaign_strategy_flat = {}
        for audience in suggested_audiences_raw:
            size = calculate_audience_size(audience, customer_data)
            current_campaign_strategy_flat[audience['type']] = {
                "Size": size,
                "Rationale": audience['rationale']
            }
        
        # Re-index for the final output structure, ensuring unique numbered keys
        formatted_for_save = {
            f"Type {idx + 1}": current_campaign_strategy_flat[key]
            for idx, key in enumerate(current_campaign_strategy_flat.keys())
        }
        current_campaign_strategy["Audience types"] = formatted_for_save
        save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
        return current_campaign_strategy["Audience types"]
    return {}

# No if __name__ == "__main__": block here. This script is now a module.
