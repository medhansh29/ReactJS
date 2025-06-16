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
# This assumes your 'data_store' folder is one level UP from where flow_generator.py is located.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_store')

# File paths for input data and campaign strategy output
PRODUCT_DATA_PATH = os.path.join(DATA_DIR, 'Product_Data.json')
CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, 'Customer_Data.json')
CAMPAIGN_PERFORMANCE_PATH = os.path.join(DATA_DIR, 'Campaign_Performance.json')
CAMPAIGN_STRATEGY_PATH = os.path.join(os.path.dirname(__file__), 'Campaign_Strategy.json') # campaign_strategy.json will be in the same dir as this script

# --- Pydantic Models for Structured Output ---

class Flow(BaseModel):
    """Represents a generated marketing flow."""
    id: str = Field(description="Unique ID for the flow. This ID should be preserved during modification.")
    campaign_id_used: str = Field(alias="Campaign ID used", description="The ID of the campaign this flow belongs to.")
    final_flow_title: str = Field(alias="Final Flow title", description="A descriptive title for the generated flow.")
    details: str = Field(alias="Details", description="Detailed steps, content, and rationale for the flow.")
    channels: List[str] = Field(alias="Channels", description="List of marketing channels suggested and detailed for this flow (e.g., 'Email', 'Social Media', 'In-app Notification').")

class FlowsList(BaseModel):
    """A list of generated marketing flows."""
    flows: List[Flow] = Field(description="A list of marketing flows.")

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

async def call_llm_for_flows(
    api_key: str,
    all_data_context: Dict,
    user_prompt: str = "",
    campaign_data: Optional[Dict[str, Any]] = None,
    preferred_channels: Optional[List[str]] = None,
    current_flows_for_llm: Optional[List[Dict[str, Any]]] = None # New parameter for modification
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM to generate or modify marketing flows."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key) if api_key else None)
    parser = PydanticOutputParser(pydantic_object=FlowsList)
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

    if current_flows_for_llm: # Modification mode
        full_prompt_template_str = """
        Based on the following data and the current marketing flows, please provide an UPDATED list of flows.
        The user's request for modification is: "{user_prompt}"

        Current Flows:
        {current_flows_json}

        Additional Context Data:
        {data_context_str}
        Campaign Ideas for Context:
        {campaign_ideas_context}

        Modify the flows based on the user's request. Maintain the JSON array format as specified by the output instructions.
        If a flow needs to be removed (e.g., if the user prompt implies removal of a specific ID), omit it from the returned list.
        If a new flow is requested, add it (ensure it has a campaign_id_used).
        If an existing flow needs its fields changed, update it.
        The 'id' field in the returned JSON object for an existing flow MUST be preserved.
        Ensure the output strictly adheres to the JSON schema.
        
        {format_instructions_var}
        """
        template_variables["current_flows_json"] = json.dumps(current_flows_for_llm, indent=2)
        template_variables["campaign_ideas_context"] = json.dumps(all_data_context.get('campaign_ideas', {}), indent=2)
    else: # Generation mode
        full_prompt_template_str = """
        Based on the following campaign idea and preferred marketing channels, generate a detailed marketing flow.
        Focus on creating a single, comprehensive flow that outlines steps and content across the specified channels.

        Campaign Idea:
        ID: {campaign_id}
        Title: {campaign_title}
        Details: {campaign_details}
        Rationale: {campaign_rationale}
        Target Audience: Type {target_audience_type_number} (Size: {audience_size})
        Growth Lever Used: {growth_lever_id}
        Discount Percentage: {discount_percentage}

        Preferred Channels for Flow: {preferred_channels_str}

        Additional Context Data:
        {data_context_str}

        User's Specific Request (if any): "{user_prompt}"

        Please generate one marketing flow in the following JSON array format.
        The 'id' field should be a newly generated unique ID.
        Ensure 'Campaign ID used' matches '{campaign_id}'.
        'Final Flow title' should be a concise name for this specific flow.
        'Details' should contain step-by-step actions, content suggestions, and how channels are integrated.
        'Channels' should list the specific channels used in this flow.
        
        {format_instructions_var}
        """
        template_variables.update({
            "campaign_id": campaign_data.get('id', 'N/A') if campaign_data else 'N/A',
            "campaign_title": campaign_data.get('Title', 'N/A') if campaign_data else 'N/A',
            "campaign_details": campaign_data.get('Details', 'N/A') if campaign_data else 'N/A',
            "campaign_rationale": campaign_data.get('Rationale', 'N/A') if campaign_data else 'N/A',
            "target_audience_type_number": campaign_data.get('Target audience with audience type number', 'N/A') if campaign_data else 'N/A',
            "audience_size": campaign_data.get('Audience size', 'N/A') if campaign_data else 'N/A',
            "growth_lever_id": campaign_data.get('Growth lever used with ID of growth lever', 'N/A') if campaign_data else 'N/A',
            "discount_percentage": campaign_data.get('Discount percentage(if applicable)', 'N/A') if campaign_data else 'N/A',
            "preferred_channels_str": ", ".join(preferred_channels) if preferred_channels else "None Specified"
        })

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [flow.model_dump(by_alias=True) for flow in response.flows]
    except ValidationError as e:
        print(f"Validation Error during LLM flow call: {e.errors()}")
        print(f"LLM probably did not return the expected JSON format. Raw error: {e}")
        return []
    except Exception as e:
        print(f"Error during LLM flow call: {e}")
        return []

async def process_flows(
    action: str, # "generate" or "modify"
    campaign_id_for_generation: Optional[str] = None, # For generation
    preferred_channels_for_generation: Optional[List[str]] = None, # For generation
    user_prompt: str = "", # For initial generation prompt or modification prompt
) -> Dict[str, Any]:
    """
    Generates or modifies marketing flows and updates campaign_strategy.json.
    Returns the updated 'Flows' dictionary.
    """
    product_data = load_json(PRODUCT_DATA_PATH)
    customer_data = load_json(CUSTOMER_DATA_PATH)
    campaign_performance = load_json(CAMPAIGN_PERFORMANCE_PATH)
    
    current_campaign_strategy = load_json(CAMPAIGN_STRATEGY_PATH)
    if not isinstance(current_campaign_strategy, dict):
        current_campaign_strategy = {}
    if "Campaign ideas" not in current_campaign_strategy:
        current_campaign_strategy["Campaign ideas"] = {}
    if "Flows" not in current_campaign_strategy:
        current_campaign_strategy["Flows"] = {}

    all_data_context = {
        'product_data': product_data,
        'customer_data': customer_data,
        'campaign_performance': campaign_performance,
        'campaign_ideas': current_campaign_strategy["Campaign ideas"] # Provide all campaign ideas for context
    }

    if action == "generate":
        if not campaign_id_for_generation:
            return {} # No campaign ID provided

        campaign_data = current_campaign_strategy["Campaign ideas"].get(campaign_id_for_generation)
        if not campaign_data:
            print(f"Error: Campaign ID '{campaign_id_for_generation}' not found.")
            return {}
        
        generated_flows = await call_llm_for_flows(
            api_key=API_KEY or "",
            all_data_context=all_data_context,
            user_prompt=user_prompt,
            campaign_data=campaign_data,
            preferred_channels=preferred_channels_for_generation
        )
        
        if generated_flows:
            for flow_item in generated_flows:
                # Ensure the flow has an ID, generate if LLM somehow missed it or it's new
                if 'id' not in flow_item or not flow_item['id']:
                    flow_item['id'] = str(uuid.uuid4())[:8]
                
                # Ensure Campaign ID used is correct if LLM got it wrong or missed it
                if flow_item.get('Campaign ID used') != campaign_id_for_generation:
                     print(f"Warning: LLM returned 'Campaign ID used' as '{flow_item.get('Campaign ID used')}' for flow, forcing to '{campaign_id_for_generation}'.")
                     flow_item['Campaign ID used'] = campaign_id_for_generation

                current_campaign_strategy["Flows"][flow_item['id']] = flow_item
            
            save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
            return current_campaign_strategy["Flows"]
        else:
            print(f"Could not generate flows for Campaign ID {campaign_id_for_generation}.")
            return {}

    elif action == "modify":
        if not current_campaign_strategy["Flows"]:
            print("No flows to modify.")
            return {}
        
        # Convert existing flows to a list for LLM input
        flows_for_llm = []
        for flow_id, flow_data in current_campaign_strategy["Flows"].items():
            temp_data = flow_data.copy()
            temp_data['id'] = flow_id # Add the actual ID for LLM context
            flows_for_llm.append(temp_data)

        updated_flows_list = await call_llm_for_flows(
            api_key=API_KEY or "",
            all_data_context=all_data_context,
            user_prompt=user_prompt,
            current_flows_for_llm=flows_for_llm
        )

        if updated_flows_list is not None:
            new_flows_dict = {}
            for flow_item in updated_flows_list:
                flow_id_from_llm = flow_item.get('id')
                if flow_id_from_llm and flow_id_from_llm in current_campaign_strategy["Flows"]:
                    new_flows_dict[flow_id_from_llm] = flow_item
                else: # New flow or ID was not preserved, generate new ID
                    new_flows_dict[str(uuid.uuid4())[:8]] = flow_item
            
            current_campaign_strategy["Flows"] = new_flows_dict
            save_json(CAMPAIGN_STRATEGY_PATH, current_campaign_strategy)
            return current_campaign_strategy["Flows"]
        else:
            print("LLM call for modification failed or returned unexpected data.")
            return {}

    return {} # Default return for unsupported action

# No if __name__ == "__main__": block here. This script is now a module.
