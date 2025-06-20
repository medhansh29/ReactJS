import json
import os
from typing import List, Dict, Any, Optional
import uuid
import asyncio
from datetime import datetime
import random # Import random for inventing numbers

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from supabase import create_client, Client
from fastapi import HTTPException

# --- Configuration ---
API_KEY = os.getenv('OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_CLIENT_ANON_KEY')

# Add a check to ensure they are loaded
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_CLIENT_ANON_KEY environment variables must be set.")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

# Initialize Supabase client (only used for read-only contextual data for generation)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class FlowDetail(BaseModel):
    """Represents the detailed steps or structure of a campaign flow."""
    steps: List[str] = Field(description="A list of steps in the campaign flow (e.g., 'Email 1: Welcome', 'In-App Message: Discount Offer').")

class FlowCampaign(BaseModel):
    """Represents a suggested campaign flow combining audience, growth lever, and product."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), alias="flow_id", description="Unique ID for the campaign flow.")
    name: str = Field(description="A concise and descriptive name for the campaign flow (e.g., 'New User Onboarding for Eco-conscious Shoppers').")
    description: str = Field(description="A brief description of the campaign's purpose and target.")
    target_audience_id: str = Field(description="The ID of the target audience for this campaign.")
    growth_lever_id: str = Field(description="The ID of the primary growth lever utilized in this campaign.")
    product_context: str = Field(description="Contextual information about the product relevant to this campaign (e.g., 'Sustainable fashion line').")
    flow_detail: FlowDetail = Field(description="Detailed steps or structure of the campaign flow.")
    estimated_reach: Optional[int] = Field(None, description="Estimated number of users reached by this campaign.")
    expected_ctr: Optional[float] = Field(None, description="Expected Click-Through Rate (e.g., 0.03 for 3%).")
    expected_conversion_rate: Optional[float] = Field(None, description="Expected Conversion Rate (e.g., 0.015 for 1.5%).")
    # Added for UI consistency
    audience_name: Optional[str] = Field(None, description="Name of the target audience (for display purposes, fetched from DB).")
    growth_lever_type: Optional[str] = Field(None, description="Type of the growth lever (for display purposes, fetched from DB).")


class SuggestedFlowsOutput(BaseModel):
    """Represents the output structure for suggested campaign flows."""
    suggested_flows: List[FlowCampaign] = Field(description="A list of generated campaign flows.")
    status: str = Field("success", description="Status of the generation process (success or error).")
    message: Optional[str] = Field(None, description="Additional message or error details.")

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=SecretStr(API_KEY))
parser = PydanticOutputParser(pydantic_object=SuggestedFlowsOutput)

campaign_flow_template = """
You are an expert campaign strategist. Your task is to generate and refine comprehensive campaign flows by synergizing audience insights, growth levers, and product context.

Current Date and Time: {current_datetime}

Here is the user's request: "{user_prompt}"

---
Available Audiences (for contextual understanding and to pick IDs from):
{available_audiences_json}

Available Growth Levers (for contextual understanding and to pick IDs from):
{available_growth_levers_json}

Existing Campaign Flows in Session (for context and modification if applicable):
{current_campaign_flows_json}
---

Based on the user's prompt and the provided data, perform the following action:
{action_instructions}

When generating or updating, ensure each campaign flow has:
- 'flow_id': A unique identifier (if generating, create a new UUID; if updating, use the existing one).
- 'name': A concise name for the campaign flow.
- 'description': A brief description.
- 'target_audience_id': **Crucially, this must be one of the 'id' values from the `available_audiences`**.
- 'growth_lever_id': **Crucially, this must be one of the 'growth_lever_id' values from the `available_growth_levers`**.
- 'product_context': Specific product context.
- 'flow_detail': An object with 'steps' (a list of actions).
- 'estimated_reach': An invented numerical value.
- 'expected_ctr': An invented numerical float (e.g., 0.03).
- 'expected_conversion_rate': An invented numerical float (e.g., 0.015).

If you are asked to generate, generate new, distinct campaign flows. Do not regenerate existing ones unless explicitly asked to modify them.
If you are asked to modify, ensure you use the exact 'flow_id' provided.

{format_instructions}
"""

campaign_flow_prompt = ChatPromptTemplate.from_template(template=campaign_flow_template)

# --- Helper Functions to Fetch Data from Supabase ---
async def fetch_audiences_from_supabase():
    """Fetches all audience data from Supabase."""
    try:
        response = supabase.table("audience_store").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching audiences from Supabase: {e}")
        return []

async def fetch_growth_levers_from_supabase():
    """Fetches all growth lever data from Supabase."""
    try:
        response = supabase.table("growth_levers_store").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching growth levers from Supabase: {e}")
        return []

# --- Main Orchestration Function ---
async def orchestrate_campaign_actions(
    user_prompt: str,
    current_campaign_flows: List[Dict[str, Any]],
    action_type: str,
    flow_id_to_affect: Optional[str] = None,
    updated_by_user_id: Optional[str] = None,
    action_finalize: Optional[str] = None # Added for finalize action
) -> List[Dict[str, Any]]:
    """
    Orchestrates actions related to campaign flows: generate (suggest), update, delete, or finalize (save to DB).

    Args:
        user_prompt (str): The prompt from the user.
        current_campaign_flows (List[Dict[str, Any]]): The current list of campaign flows from UI session memory.
        action_type (str): The type of action to perform ('generate' -> 'suggest', 'update_singular', 'delete_singular', 'finalize').
        flow_id_to_affect (Optional[str]): The ID of the campaign flow to affect for update/delete.
        updated_by_user_id (Optional[str]): User ID for tracking who finalized the data.
        action_finalize (Optional[str]): Specific action for finalize ('overwrite').

    Returns:
        List[Dict[str, Any]]: The updated list of campaign flow records for UI session memory.
    """
    updated_flows_list = [FlowCampaign(**flow).model_dump(by_alias=True) if not isinstance(flow, FlowCampaign) else flow for flow in current_campaign_flows]
    current_datetime = datetime.now().isoformat()
    action_instructions = ""

    # Fetch available audiences and growth levers for contextualization
    available_audiences = await fetch_audiences_from_supabase()
    available_growth_levers = await fetch_growth_levers_from_supabase()

    if action_type == "generate": # This will now act as 'suggest'
        action_instructions = """
        Generate new, distinct campaign flows based on the user's prompt. Do not regenerate existing ones.
        Each generated campaign flow must use one of the provided 'target_audience_id' and 'growth_lever_id' from the available lists.
        Focus on creating synergistic combinations. Propose at least 3-5 new, diverse campaign flows.
        """
        chain = campaign_flow_prompt | llm | parser
        response_obj = await chain.invoke({
            "user_prompt": user_prompt,
            "available_audiences_json": json.dumps(available_audiences, indent=2),
            "available_growth_levers_json": json.dumps(available_growth_levers, indent=2),
            "current_campaign_flows_json": json.dumps(current_campaign_flows, indent=2),
            "action_instructions": action_instructions,
            "format_instructions": parser.get_format_instructions(),
            "current_datetime": current_datetime
        })

        if response_obj.status == "success" and response_obj.suggested_flows:
            new_flows = [flow.model_dump(by_alias=True) for flow in response_obj.suggested_flows]
            existing_ids = {flow.get("flow_id") for flow in updated_flows_list if flow.get("flow_id")}
            for new_flow in new_flows:
                if new_flow.get("flow_id") and new_flow["flow_id"] not in existing_ids:
                    updated_flows_list.append(new_flow)
        else:
            print(f"LLM did not return new campaign flows: {response_obj.message}")


    elif action_type == "update_singular":
        action_instructions = f"""
        Modify the campaign flow with ID '{flow_id_to_affect}' based on the user's prompt.
        Only return the modified campaign flow object. Ensure its ID remains '{flow_id_to_affect}'.
        """
        flow_to_modify = next((flow for flow in updated_flows_list if flow.get("flow_id") == flow_id_to_affect), None)
        if not flow_to_modify:
            raise HTTPException(status_code=404, detail=f"Campaign flow with ID {flow_id_to_affect} not found.")

        chain = campaign_flow_prompt | llm | parser
        response_obj = await chain.invoke({
            "user_prompt": user_prompt,
            "available_audiences_json": json.dumps(available_audiences, indent=2),
            "available_growth_levers_json": json.dumps(available_growth_levers, indent=2),
            "current_campaign_flows_json": json.dumps([flow_to_modify], indent=2), # Pass only the relevant flow
            "action_instructions": action_instructions,
            "format_instructions": parser.get_format_instructions(),
            "current_datetime": current_datetime
        })

        if response_obj.status == "success" and response_obj.suggested_flows:
            modified_flow = response_obj.suggested_flows[0].model_dump(by_alias=True)
            updated_flows_list = [
                modified_flow if flow.get("flow_id") == flow_id_to_affect else flow
                for flow in updated_flows_list
            ]
        else:
            print(f"LLM did not return a modified campaign flow for ID {flow_id_to_affect}: {response_obj.message}. Original list returned.")


    elif action_type == "delete_singular":
        if flow_id_to_affect:
            updated_flows_list = [
                flow for flow in updated_flows_list if flow.get("flow_id") != flow_id_to_affect
            ]
        else:
            raise ValueError("Campaign Flow ID is required for delete_singular action.")

    elif action_type == "finalize":
        if not updated_by_user_id:
            raise ValueError("User ID is required for finalize action.")

        try:
            data_to_save = []
            for flow in updated_flows_list:
                data_to_save.append({
                    "flow_id": flow.get("flow_id"),
                    "name": flow.get("name"),
                    "description": flow.get("description"),
                    "target_audience_id": flow.get("target_audience_id"),
                    "growth_lever_id": flow.get("growth_lever_id"),
                    "product_context": flow.get("product_context"),
                    "flow_detail": json.dumps(flow.get("flow_detail", {})), # Store JSON as string
                    "estimated_reach": flow.get("estimated_reach"),
                    "expected_ctr": flow.get("expected_ctr"),
                    "expected_conversion_rate": flow.get("expected_conversion_rate"),
                    "user_id": updated_by_user_id,
                    "timestamp": datetime.now().isoformat()
                })

            # --- IMPORTANT: Saving to 'campaigns' table as requested ---
            # Ensure your Supabase project has a table named 'campaigns'
            # with the appropriate schema for these flow campaigns.
            response = supabase.table("campaigns").upsert(data_to_save, on_conflict="flow_id").execute()

            if response.data:
                print(f"Successfully saved {len(response.data)} campaign flows to Supabase 'campaigns' table.")
            else:
                print("No data returned after upsert to 'campaigns' table, check Supabase operation.")

        except Exception as e:
            print(f"An unexpected error occurred during finalize action: {e}")
            raise # Re-raise to ensure error is propagated if necessary

    # Enrich with audience_name and growth_lever_type for UI display before returning
    final_formatted_flows = []
    audience_map = {aud.get("id"): aud.get("name") for aud in available_audiences}
    growth_lever_map = {gl.get("growth_lever_id"): gl.get("type") for gl in available_growth_levers}

    for flow in updated_flows_list:
        formatted_flow = {
            "id": flow.get("flow_id"), # Ensure 'id' is present for UI consistency
            "flow_id": flow.get("flow_id"),
            "name": flow.get("name"),
            "description": flow.get("description"),
            "target_audience_id": flow.get("target_audience_id"),
            "growth_lever_id": flow.get("growth_lever_id"),
            "product_context": flow.get("product_context"),
            "flow_detail": flow.get("flow_detail"),
            "estimated_reach": flow.get("estimated_reach"),
            "expected_ctr": flow.get("expected_ctr"),
            "expected_conversion_rate": flow.get("expected_conversion_rate"),
            "audience_name": audience_map.get(flow.get("target_audience_id"), "N/A"),
            "growth_lever_type": growth_lever_map.get(flow.get("growth_lever_id"), "N/A"),
        }
        final_formatted_flows.append(formatted_flow)

    return final_formatted_flows