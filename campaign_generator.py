import json
import os
from typing import List, Dict, Any, Optional
import uuid
import asyncio

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from supabase import create_client, Client

# --- Configuration ---
API_KEY = os.getenv('OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_CLIENT_ANON_KEY')

# Add a check to ensure they are loaded
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_CLIENT_ANON_KEY environment variables must be set.")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set.")

# Initialize Supabase client (only used for read-only contextual data)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class CampaignIdea(BaseModel):
    """Represents a generated campaign idea, matching the 'campaigns' table schema (for UI memory)."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the campaign idea.")
    name: str = Field(description="A descriptive name for the campaign idea.")
    description: str = Field(description="Specific details and actions for this campaign.")
    hypothesis: str = Field(description="Explanation of why this campaign is effective for the chosen audience, growth lever, and product.")
    target_cohort: str = Field(description="The title of the target audience for this campaign.")
    lever_config: str = Field(description="The type of the growth lever used for this campaign.")
    product: str = Field(description="The name of the product this campaign is targeting.")
    discount_percentage: Optional[float] = Field(
        default=None,
        description="The exact discount percentage for this campaign (e.g., 10.5 for 10.5%). Null if not applicable."
    )
    audience_id: str = Field(description="The ID of the target audience used for this campaign.")
    growth_lever_id: str = Field(description="The ID of the growth lever used for this campaign.")
    product_id: str = Field(description="The ID of the product targeted by this campaign.")


class CampaignIdeas(BaseModel):
    """A list of generated campaign ideas."""
    campaign_ideas: List[CampaignIdea] = Field(description="A list of 1-2 distinct campaign ideas.")

# --- Supabase Helper Functions (Only fetch for contextual read-only data) ---

async def fetch_from_supabase(table_name: str, id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches data from a specified Supabase table, optionally by ID.
    This function is retained only for fetching *read-only* contextual data.
    """
    try:
        query = supabase.from_(table_name).select('*')
        if id:
            query = query.eq('id', id)
        response = query.execute()
        if response and hasattr(response, 'data'):
            return response.data
        return []
    except Exception as e:
        return []

# Removed insert_into_campaigns_store
# Removed update_campaigns_store
# Removed delete_from_supabase (as it was only used for campaigns in this file)

# --- LLM Call Function ---

async def call_llm_for_campaign_ideas(
    api_key: str,
    selected_audience_data: Dict[str, Any], # Now mandatory, passed from UI memory
    selected_growth_lever_data: Dict[str, Any], # Now mandatory, passed from UI memory
    selected_product_data: Dict[str, Any], # Now mandatory, passed from UI memory
    user_prompt: str = "",
    current_campaign_ideas_for_llm: Optional[List[Dict[str, Any]]] = None # Current state from UI memory
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM to generate or modify campaign ideas with minimal, relevant data."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key) if api_key else None)
    parser = PydanticOutputParser(pydantic_object=CampaignIdeas)
    format_instructions = parser.get_format_instructions()
    
    template_variables = {
        "format_instructions_var": format_instructions,
        "user_prompt": user_prompt,
    }

    if current_campaign_ideas_for_llm: # Modification mode
        full_prompt_template_str = """
        Based on the current campaign ideas and the user's request, please provide an UPDATED list of campaign ideas.
        The user's request for modification is: "{user_prompt}"

        Current Campaign Ideas (only the one(s) to be modified):
        {current_campaign_ideas_json}
        
        Ensure the output strictly adheres to the JSON schema, preserving the 'id' of existing campaigns and setting it to null for new ones if any are created.
        If the user implies removal of a specific ID, omit it from the returned list.
        If a new idea is requested, add it (with a null 'id').
        If an existing idea needs its fields changed, update it.
        
        {format_instructions_var}
        """
        # Ensure 'id' is included when passing current campaigns to LLM for modification
        campaigns_with_id_for_llm = [
            {k: v for k, v in camp.items()} # Copy all items including 'id'
            for camp in current_campaign_ideas_for_llm
        ]
        template_variables["current_campaign_ideas_json"] = json.dumps(campaigns_with_id_for_llm, indent=2)
    else: # Generation mode
        # These fields are guaranteed to be present as they are passed from UI's memory
        audience_title = selected_audience_data.get('title', 'N/A')
        growth_lever_type = selected_growth_lever_data.get('type', 'N/A')
        product_name = selected_product_data.get('name', 'N/A')
        discount_percentage = selected_growth_lever_data.get('exact_discount_percentage')

        full_prompt_template_str = """
        Based on the following *specifically selected* audience, growth lever, and product details, generate 1-2 compelling campaign ideas.
        If the user has a specific request, integrate it.

        Selected Audience:
        Title: {audience_title}
        Rationale: {audience_rationale}
        Size: {audience_size}
        Audience ID: {audience_id}

        Selected Growth Lever:
        Type: {growth_lever_type}
        Details: {growth_lever_details}
        Rationale: {growth_lever_rationale}
        Discount Percentage: {discount_percentage}
        Growth Lever ID: {growth_lever_id}

        Selected Product:
        Name: {product_name}
        Product ID: {product_id}
        Product Description: {product_description}

        User's Specific Request (if any): "{user_prompt}"

        Please generate 1-2 campaign ideas in the following JSON array format.
        For each campaign idea, ensure:
        - 'id' field is omitted or set to null (will be generated by the system if new).
        - 'name': A concise title.
        - 'description': Specific details and execution steps.
        - 'hypothesis': Why this campaign will be effective, referencing the provided audience, lever, and product details.
        - 'target_cohort': Must be exactly "{audience_title}".
        - 'lever_config': Must be exactly "{growth_lever_type}".
        - 'product': Must be exactly "{product_name}".
        - 'discount_percentage': Set to {discount_percentage} if applicable, otherwise null.
        - 'audience_id': Must be exactly "{audience_id}".
        - 'growth_lever_id': Must be exactly "{growth_lever_id}".
        - 'product_id': Must be exactly "{product_id}".

        {format_instructions_var}
        """
        template_variables.update(
            **{
                "audience_title": audience_title,
                "audience_rationale": selected_audience_data.get('rationale'),
                "audience_size": selected_audience_data.get('audience_size'),
                "audience_id": selected_audience_data.get('id'),

                "growth_lever_type": growth_lever_type,
                "growth_lever_details": selected_growth_lever_data.get('details'),
                "growth_lever_rationale": selected_growth_lever_data.get('rationale'),
                "growth_lever_id": selected_growth_lever_data.get('id'),
                "discount_percentage": discount_percentage,

                "product_name": product_name,
                "product_id": selected_product_data.get('id'),
                "product_description": selected_product_data.get('description'),
            }
        )

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [idea.model_dump() for idea in response.campaign_ideas]
    except ValidationError as e:
        # Removed print statement for production readiness
        return []
    except Exception as e:
        # Removed print statement for production readiness
        return []

async def process_campaign_ideas(
    current_campaigns: List[Dict[str, Any]], # Now passed from UI's session memory
    user_prompt: str = "",
    action_type: str = "generate", # 'generate', 'update_singular', 'delete_singular'
    # For generation: combo of full data, not just IDs
    selected_combinations_data: Optional[List[Dict[str, Any]]] = None, 
    campaign_id_to_affect: Optional[str] = None # For singular update/delete
) -> List[Dict[str, Any]]:
    """
    Generates, updates, or deletes campaign ideas in memory.
    Returns the updated list of campaign records for UI session memory.
    """
    updated_campaigns_list = list(current_campaigns) # Create a mutable copy

    if action_type == "delete_singular":
        if campaign_id_to_affect:
            updated_campaigns_list = [c for c in updated_campaigns_list if c.get('id') != campaign_id_to_affect]
        return updated_campaigns_list

    elif action_type == "update_singular":
        if not campaign_id_to_affect:
            return updated_campaigns_list

        campaign_to_update = next((c for c in updated_campaigns_list if c.get('id') == campaign_id_to_affect), None)
        if not campaign_to_update:
            return updated_campaigns_list # Campaign not found in current memory

        # To update a singular campaign, we need to provide its current context to the LLM.
        # This means extracting the linked audience, growth lever, and product data from the UI's session.
        # This requires the UI to pass these full objects or for these functions to fetch from read-only sources.
        # For this refactor, we assume the UI provides enough context or the `fetch_from_supabase` is sufficient
        # for these *read-only* lookups.

        # Fetch full details of the linked entities from their respective tables (read-only context)
        # Note: These are fetches for context for the LLM, not for saving/updating these tables.
        selected_audience = (await fetch_from_supabase('audience_store', campaign_to_update.get('audience_id')))[0] if campaign_to_update.get('audience_id') else None
        selected_growth_lever = (await fetch_from_supabase('growth_levers_store', campaign_to_update.get('growth_lever_id')))[0] if campaign_to_update.get('growth_lever_id') else None
        selected_product = (await fetch_from_supabase('product_store', campaign_to_update.get('product_id')))[0] if campaign_to_update.get('product_id') else None

        if not all([selected_audience, selected_growth_lever, selected_product]):
            # print(f"ERROR: Linked audience, growth lever, or product data missing for campaign ID {campaign_id_to_affect}. Cannot update.") # Removed print
            return updated_campaigns_list

        if not all([selected_audience, selected_growth_lever, selected_product]):
            return updated_campaigns_list

        llm_response_list = await call_llm_for_campaign_ideas(
            api_key=API_KEY or "",
            user_prompt=user_prompt,
            selected_audience_data=selected_audience if selected_audience is not None else {},
            selected_growth_lever_data=selected_growth_lever if selected_growth_lever is not None else {},
            selected_product_data=selected_product if selected_product is not None else {},
            current_campaign_ideas_for_llm=[campaign_to_update] # Send the specific campaign to modify
        )

        if llm_response_list:
            updated_campaign_data = llm_response_list[0]
            # Find and replace the updated campaign in the list
            for i, camp in enumerate(updated_campaigns_list):
                if camp.get('id') == campaign_id_to_affect:
                    updated_campaign_data['id'] = campaign_id_to_affect # Preserve ID
                    updated_campaigns_list[i] = updated_campaign_data
                    break
        return updated_campaigns_list

    elif action_type == "generate":
        if not selected_combinations_data:
            return updated_campaigns_list

        for combo_data in selected_combinations_data:
            selected_audience = combo_data.get('audience_data')
            selected_growth_lever = combo_data.get('growth_lever_data')
            selected_product = combo_data.get('product_data')

            if not all([selected_audience, selected_growth_lever, selected_product]):
                # print(f"Warning: Skipping combination due to incomplete data.") # Removed print
                continue

            generated_ideas = await call_llm_for_campaign_ideas(
                api_key=API_KEY or "",
                user_prompt=user_prompt,
                selected_audience_data=selected_audience if selected_audience is not None else {},
                selected_growth_lever_data=selected_growth_lever if selected_growth_lever is not None else {},
                selected_product_data=selected_product if selected_product is not None else {},
                current_campaign_ideas_for_llm=None
            )
            
            if generated_ideas:
                for idea in generated_ideas:
                    if 'id' not in idea or not idea['id']:
                        idea['id'] = str(uuid.uuid4())
                    # Ensure IDs from the *selected combo* are correctly linked
                    idea['audience_id'] = selected_audience.get('id') if selected_audience else None
                    idea['growth_lever_id'] = selected_growth_lever.get('id') if selected_growth_lever else None
                    idea['product_id'] = selected_product.get('id') if selected_product else None
                    updated_campaigns_list.append(idea)
        return updated_campaigns_list
    
    return updated_campaigns_list # Default return

# Removed test_campaign_generator and if __name__ == "__main__": block
