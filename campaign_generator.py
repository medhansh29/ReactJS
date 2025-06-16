import json
import os
from typing import List, Dict, Any, Optional
import uuid
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError

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

class CampaignIdea(BaseModel):
    """Represents a generated campaign idea."""
    title: str = Field(alias="Title", description="A descriptive title for the campaign idea.")
    growth_lever_used_id: Optional[str] = Field(alias="Growth lever used with ID of growth lever", description="The ID of the growth lever used for this campaign. Null if not provided.")
    discount_percentage: Optional[float] = Field(
        default=None,
        alias="Discount percentage(if applicable)",
        description="The exact discount percentage for this campaign (e.g., 10.5 for 10.5%). Null if not applicable."
    )
    target_audience_type_number: int = Field(alias="Target audience with audience type number", description="The number of the target audience type (e.g., 1 for 'Type 1').")
    audience_size: int = Field(alias="Audience size", description="The size of the target audience.")
    details: str = Field(alias="Details", description="Specific details and actions for this campaign.")
    rationale: str = Field(alias="Rationale", description="Explanation of why this campaign is effective for the chosen audience and growth lever.")

class CampaignIdeas(BaseModel):
    """A list of generated campaign ideas."""
    campaign_ideas: List[CampaignIdea] = Field(description="A list of 2-3 distinct campaign ideas.")

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

async def call_llm_for_campaign_ideas(
    api_key: str,
    all_data_context: Dict,
    user_prompt: str = "",
    selected_audience: Optional[Dict[str, Any]] = None,
    selected_growth_lever: Optional[Dict[str, Any]] = None,
    current_campaign_ideas_for_llm: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM to generate or modify campaign ideas."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key) if api_key else None)
    parser = PydanticOutputParser(pydantic_object=CampaignIdeas)
    format_instructions = parser.get_format_instructions()

    data_context_str = f"""
Product Data: {json.dumps(all_data_context.get('product_data', []), indent=2)}
Customer Data: {json.dumps(all_data_context.get('customer_data', []), indent=2)}
Campaign Performance Data: {json.dumps(all_data_context.get('campaign_performance', []), indent=2)}
"""
    
    template_variables = {
        "data_context_str": data_context_str,
        "format_instructions_var": format_instructions,
        "user_prompt": user_prompt,
    }

    if current_campaign_ideas_for_llm: # Modification mode
        full_prompt_template_str = """
        Based on the following data and the current campaign ideas, please provide an UPDATED list of campaign ideas.
        The user's request for modification is: "{user_prompt}"

        Current Campaign Ideas:
        {current_campaign_ideas_json}

        Additional Context Data:
        {data_context_str}

        Modify the campaign ideas based on the user's request. Maintain the JSON array format as specified by the output instructions.
        If an idea needs to be removed (e.g., if the user prompt implies removal of a specific ID), omit it from the returned list.
        If a new idea is requested, add it.
        If an existing idea needs its fields changed, update it.
        Ensure the output strictly adheres to the JSON schema.
        
        {format_instructions_var}
        """
        template_variables["current_campaign_ideas_json"] = json.dumps(current_campaign_ideas_for_llm, indent=2)
    else: # Generation mode
        discount_to_pass = selected_growth_lever.get('exact_discount_percentage') if selected_growth_lever else None
        if discount_to_pass is not None:
            try:
                discount_to_pass = float(discount_to_pass)
            except (ValueError, TypeError):
                discount_to_pass = None

        full_prompt_template_str = """
        Based on the following data, and the selected audience and growth lever, generate 2-3 compelling campaign ideas.
        If the user has a specific request, integrate it.

        Selected Audience:
        Type Number: {target_audience_number}
        Rationale: {audience_rationale}
        Size: {audience_size}

        Selected Growth Lever:
        ID: {growth_lever_id}
        Type: {growth_lever_type}
        Details: {growth_lever_details}
        Rationale: {growth_lever_rationale}
        Discount Percentage: {growth_lever_discount}

        Additional Context Data:
        {data_context_str}

        User's Specific Request (if any): "{user_prompt}"

        Please generate 2-3 campaign ideas in the following JSON array format.
        Ensure that for EACH campaign idea, the 'Growth lever used with ID of growth lever' field strictly matches the 'growth_lever_id' provided in the input.
        For each campaign idea, provide:
        - A concise 'Title' for the campaign.
        - The 'Growth lever used with ID of growth lever' (which is the 'growth_lever_id' passed).
        - The 'Discount percentage(if applicable)' (null if not applicable, derived from 'growth_lever_discount').
        - The 'Target audience with audience type number' (e.g., 1 for 'Type 1', derived from 'target_audience_number').
        - The 'Audience size'.
        - 'Details' outlining the campaign strategy and execution steps.
        - 'Rationale' explaining why this campaign is effective for the specific audience and growth lever, referencing data.

        {format_instructions_var}
        """
        template_variables.update(
            **{
                "target_audience_number": selected_audience.get('number') if selected_audience else None,
                "audience_rationale": selected_audience.get('Rationale') if selected_audience else None,
                "audience_size": selected_audience.get('Size') if selected_audience else None,
                "growth_lever_id": selected_growth_lever.get('ID') if selected_growth_lever else None,
                "growth_lever_type": selected_growth_lever.get('Type') if selected_growth_lever else None,
                "growth_lever_details": selected_growth_lever.get('Details') if selected_growth_lever else None,
                "growth_lever_rationale": selected_growth_lever.get('Rationale') if selected_growth_lever else None,
                "growth_lever_discount": discount_to_pass,
            }
        )

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [idea.model_dump(by_alias=True) for idea in response.campaign_ideas]
    except ValidationError as e:
        print(f"Validation Error during LLM campaign idea call: {e.errors()}")
        print(f"LLM probably did not return the expected JSON format. Raw error: {e}")
        return []
    except Exception as e:
        print(f"Error during LLM campaign idea call: {e}")
        return []

async def process_campaign_ideas(
    action: str, # "generate" or "modify"
    campaign_ids_for_generation: Optional[List[Dict[str, Any]]] = None, # For generation
    user_prompt: str = "", # For initial generation prompt or modification prompt
) -> Dict[str, Any]:
    """
    Generates or modifies campaign ideas and updates campaign_strategy.json.
    Returns the updated 'Campaign ideas' dictionary.
    
    Args:
        action (str): "generate" to create new ideas, "modify" to update existing.
        campaign_ids_for_generation (Optional[List[Dict[str, Any]]]): List of dicts {audience_num, gl_id} for generation.
        user_prompt (str): User's prompt for generation or modification.
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
    if "Campaign ideas" not in current_campaign_strategy:
        current_campaign_strategy["Campaign ideas"] = {}

    all_data_context = {
        'product_data': product_data,
        'customer_data': customer_data,
        'campaign_performance': campaign_performance,
        'audience_types': current_campaign_strategy["Audience types"],
        'growth_levers': current_campaign_strategy["Growth Levers"]
    }

    if action == "generate":
        if not campaign_ids_for_generation:
            return {} # No combinations provided

        generated_ideas_total = {}
        for combo in campaign_ids_for_generation:
            audience_num = combo['audience_type_number']
            gl_id = combo['growth_lever_id']
            
            selected_audience_data = all_data_context["audience_types"].get(f"Type {audience_num}")
            selected_growth_lever_data = all_data_context["growth_levers"].get(gl_id)

            if not selected_audience_data or not selected_growth_lever_data:
                print(f"Warning: Skipping invalid combination (Audience {audience_num}, GL {gl_id}).")
                continue
            
            # Add 'number' field for LLM prompt consistency
            selected_audience_data['number'] = audience_num 

            generated_campaign_ideas = await call_llm_for_campaign_ideas(
                selected_audience=selected_audience_data,
                selected_growth_lever=selected_growth_lever_data,
                api_key=API_KEY or "",
                all_data_context=all_data_context,
                user_prompt=user_prompt
            )
            
            if generated_campaign_ideas:
                for idea in generated_campaign_ideas:
                    campaign_id = str(uuid.uuid4())[:8]
                    generated_ideas_total[campaign_id] = idea
        
        current_campaign_strategy["Campaign ideas"].update(generated_ideas_total) # Add new ideas
        save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
        return current_campaign_strategy["Campaign ideas"]

    elif action == "modify":
        if not current_campaign_strategy["Campaign ideas"]:
            print("No campaign ideas to modify.")
            return {}
        
        # Convert existing ideas to a list for LLM input
        campaign_ideas_for_llm = []
        for camp_id, camp_data in current_campaign_strategy["Campaign ideas"].items():
            temp_data = camp_data.copy()
            temp_data['id'] = camp_id # Add the actual ID for LLM context
            campaign_ideas_for_llm.append(temp_data)

        updated_campaign_ideas_raw = await call_llm_for_campaign_ideas(
            api_key=API_KEY or "",
            all_data_context=all_data_context, # Ensure all_data_context is passed
            user_prompt=user_prompt,
            current_campaign_ideas_for_llm=campaign_ideas_for_llm
        )

        if updated_campaign_ideas_raw is not None: # LLM successfully returned a list (could be empty for removals)
            new_campaign_ideas_dict = {}
            for idea in updated_campaign_ideas_raw:
                # Use the ID returned by LLM or generate a new one if it's truly new and ID is missing
                # Prefer LLM's returned 'id' if it exists, otherwise generate for new ones.
                # If LLM *modified* an existing one, it *should* return the existing ID.
                camp_id_from_llm = idea.get('id')
                if camp_id_from_llm and camp_id_from_llm in current_campaign_strategy["Campaign ideas"]:
                    # Preserve original ID if it was an existing one being modified
                    new_campaign_ideas_dict[camp_id_from_llm] = idea
                else:
                    # Treat as a new idea or a misidentified modification, generate new ID
                    new_campaign_ideas_dict[str(uuid.uuid4())[:8]] = idea
            
            current_campaign_strategy["Campaign ideas"] = new_campaign_ideas_dict
            save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
            return current_campaign_strategy["Campaign ideas"]
        else:
            print("LLM call for modification failed or returned unexpected data.")
            return {} # Indicate failure/no change

    return {} # Default return for unsupported action

# No if __name__ == "__main__": block here. This script is now a module.
