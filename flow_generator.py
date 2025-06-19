import json
import os
from typing import List, Dict, Any, Optional
import uuid
import asyncio
from datetime import datetime

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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class Journey(BaseModel):
    """Represents a generated marketing journey, matching the 'journeys' Supabase table schema."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the journey.")
    campaign_name: str = Field(description="The name of the campaign this journey is part of.")
    name: str = Field(description="A descriptive name for the marketing journey (e.g., 'New User Onboarding Email Sequence').")
    description: str = Field(description="Detailed steps, content outlines, and rationale for the journey's progression.")
    ux_type: str = Field(description="The type of user experience or funnel this journey represents (e.g., 'Onboarding Flow', 'Re-engagement Sequence', 'Conversion Funnel', 'Retention Program').")
    channel: str = Field(description="The primary marketing channel used for this journey (e.g., 'Email', 'In-app Notification', 'SMS', 'Social Media', 'Push Notification', 'Website Pop-up').")
    cohort: str = Field(description="The target audience (cohort) for this journey, matching the campaign's target cohort.")
    outcome_metric: str = Field(description="The key metric this journey aims to improve (e.g., 'Conversion Rate', 'Retention Rate', 'Average Order Value', 'Engagement').")
    lever: str = Field(description="The growth lever primarily addressed by this journey (e.g., 'Discount', 'Product Feature Highlight', 'Educational Content').")
    product: str = Field(description="The product name this journey is associated with.")
    last_edited: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp of when the journey was last edited.")

class JourneysList(BaseModel):
    """A list of generated marketing journeys."""
    journeys: List[Journey] = Field(description="A list of marketing journeys.")

# --- Supabase Helper Functions ---
# Note: These are retained for the 'journeys' table as per the requirement,
# and also for fetching *read-only* contextual data from other tables.

async def fetch_from_supabase(table_name: str, id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches data from a specified Supabase table, optionally by ID."""
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

async def insert_into_journeys_store(data: List[Dict]) -> None:
    """Inserts data into the 'journeys' Supabase table."""
    try:
        response = supabase.from_('journeys').insert(data).execute()
        if response and hasattr(response, 'data'):
            pass # Removed print for production
        else:
            pass # Removed print for production
    except Exception as e:
        pass # Removed print for production

async def update_journeys_store(id: str, data: Dict) -> None:
    """Updates a record in the 'journeys' Supabase table."""
    try:
        response = supabase.from_('journeys').update(data).eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            pass # Removed print for production
        else:
            pass # Removed print for production
    except Exception as e:
        pass # Removed print for production

async def delete_from_supabase(table_name: str, id: str) -> None:
    """Deletes a record from a specified Supabase table by ID."""
    try:
        response = supabase.from_(table_name).delete().eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            pass # Removed print for production
        else:
            pass # Removed print for production
    except Exception as e:
        pass # Removed print for production

# --- LLM Call Function ---

async def call_llm_for_journeys(
    api_key: str,
    campaign_data: Dict[str, Any], # Now mandatory, as journeys are derived from campaigns
    user_prompt: str = "",
    current_journeys_for_llm: Optional[List[Dict[str, Any]]] = None # For modification mode
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM to generate or modify marketing journeys."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key) if api_key else None)
    parser = PydanticOutputParser(pydantic_object=JourneysList)
    format_instructions = parser.get_format_instructions()
    
    template_variables = {
        "format_instructions_var": format_instructions,
        "user_prompt": user_prompt,
        "campaign_name": campaign_data.get('name', 'N/A'),
        "target_cohort": campaign_data.get('target_cohort', 'N/A'),
        "lever_config": campaign_data.get('lever_config', 'N/A'),
        "product": campaign_data.get('product', 'N/A'),
        "campaign_description": campaign_data.get('description', 'N/A'),
        "campaign_hypothesis": campaign_data.get('hypothesis', 'N/A'),
        "discount_percentage": campaign_data.get('discount_percentage', 'N/A')
    }

    if current_journeys_for_llm: # Modification mode
        full_prompt_template_str = """
        Based on the following campaign details and the current journey idea(s), please provide an UPDATED list of journeys.
        The user's request for modification is: "{user_prompt}"

        Campaign Details for Context:
        Name: {campaign_name}
        Description: {campaign_description}
        Hypothesis: {campaign_hypothesis}
        Target Cohort: {target_cohort}
        Growth Lever: {lever_config}
        Product: {product}
        Discount: {discount_percentage}

        Current Journeys:
        {current_journeys_json}
        
        Modify the journeys based on the user's request. Maintain the JSON array format as specified by the output instructions.
        If a journey needs to be removed (e.g., if the user prompt implies removal of a specific ID), omit it from the returned list.
        If a new journey is requested, add it (with a new 'id' and 'last_edited' timestamp).
        If an existing journey needs its fields changed, update it.
        The 'id' field for an existing journey MUST be preserved.
        Ensure 'campaign_name', 'cohort', 'lever', and 'product' fields are consistent with the Campaign Details provided above.
        Update 'last_edited' field for any modified or new journeys.
        
        {format_instructions_var}
        """
        template_variables["current_journeys_json"] = json.dumps(current_journeys_for_llm, indent=2)
    else: # Generation mode
        full_prompt_template_str = """
        Based on the following campaign idea, generate 1-2 detailed marketing journeys.
        A journey defines the step-by-step user experience across one or more channels to achieve a specific outcome related to the campaign.

        Campaign Details:
        Name: {campaign_name}
        Description: {campaign_description}
        Hypothesis: {campaign_hypothesis}
        Target Cohort: {target_cohort}
        Growth Lever: {lever_config}
        Product: {product}
        Discount: {discount_percentage}

        User's Specific Request (if any): "{user_prompt}"

        Please generate 1-2 marketing journeys in the following JSON array format.
        For each journey idea, ensure:
        - 'id' field is omitted or set to null (will be generated by the system).
        - 'campaign_name': Must be exactly "{campaign_name}".
        - 'name': A concise, descriptive name for the journey.
        - 'description': Detailed, step-by-step actions and content outlines for the journey.
        - 'ux_type': The type of user experience (e.g., 'Onboarding Flow', 'Re-engagement Sequence', 'Conversion Funnel', 'Retention Program').
        - 'channel': The *primary* marketing channel for this journey (e.g., 'Email', 'In-app Notification', 'SMS', 'Social Media', 'Push Notification', 'Website Pop-up'). Choose one primary channel.
        - 'cohort': Must be exactly "{target_cohort}".
        - 'outcome_metric': The key metric this journey aims to improve (e.g., 'Conversion Rate', 'Retention Rate', 'Average Order Value', 'Engagement').
        - 'lever': Must be exactly "{lever_config}".
        - 'product': Must be exactly "{product}".
        - 'last_edited': Set to the current ISO formatted datetime.

        {format_instructions_var}
        """

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [journey.model_dump() for journey in response.journeys]
    except ValidationError as e:
        # Removed print for production
        return []
    except Exception as e:
        # Removed print for production
        return []

async def process_journeys(
    selected_campaign_data: Dict[str, Any], # Full campaign data passed from UI memory
    user_prompt: str = "",
    action_type: str = "generate", # 'generate', 'update_singular', 'delete_singular'
    journey_id_to_affect: Optional[str] = None # For singular update/delete
) -> List[Dict[str, Any]]:
    """
    Generates, updates, or deletes marketing journeys and updates the 'journeys' Supabase table.
    Returns the updated list of journey records from Supabase (as this is the final save).
    """
    
    current_journeys_from_db = await fetch_from_supabase('journeys') # Still fetch to manage DB state
    
    # current_campaigns_from_db is replaced by `selected_campaign_data` parameter

    if action_type == "delete_singular":
        if journey_id_to_affect:
            await delete_from_supabase('journeys', journey_id_to_affect)
        return await fetch_from_supabase('journeys')

    elif action_type == "update_singular":
        if not journey_id_to_affect:
            return await fetch_from_supabase('journeys')

        journey_to_update = next((j for j in current_journeys_from_db if j.get('id') == journey_id_to_affect), None)
        if not journey_to_update:
            return await fetch_from_supabase('journeys')
        
        llm_response_list = await call_llm_for_journeys(
            api_key=API_KEY or "",
            campaign_data=selected_campaign_data, # Provide the linked campaign for context
            user_prompt=user_prompt, # User's specific modification request
            current_journeys_for_llm=[journey_to_update] # Send the specific journey to modify
        )

        if llm_response_list:
            updated_journey_data = llm_response_list[0] 
            
            # Preserve the original ID and ensure last_edited is updated
            updated_journey_data['id'] = journey_id_to_affect 
            updated_journey_data['last_edited'] = datetime.now().isoformat()
            
            # Ensure consistency with DB schema and chosen IDs
            data_to_update = {
                "id": updated_journey_data.get('id'),
                "campaign_name": updated_journey_data.get('campaign_name'),
                "name": updated_journey_data.get('name'),
                "description": updated_journey_data.get('description'),
                "ux_type": updated_journey_data.get('ux_type'),
                "channel": updated_journey_data.get('channel'),
                "cohort": updated_journey_data.get('cohort'),
                "outcome_metric": updated_journey_data.get('outcome_metric'),
                "lever": updated_journey_data.get('lever'),
                "product": updated_journey_data.get('product'),
                "last_edited": updated_journey_data.get('last_edited'),
            }
            await update_journeys_store(journey_id_to_affect, data_to_update)
        return await fetch_from_supabase('journeys')

    elif action_type == "generate":
        if not selected_campaign_data:
            return await fetch_from_supabase('journeys')
        
        generated_journeys_to_insert = []
        generated_ideas = await call_llm_for_journeys(
            api_key=API_KEY or "",
            campaign_data=selected_campaign_data,
            user_prompt=user_prompt,
            current_journeys_for_llm=None # Not modifying existing, generating new
        )
        
        if generated_ideas:
            for journey_item in generated_ideas:
                # Ensure the journey has an ID and last_edited timestamp
                if 'id' not in journey_item or not journey_item['id']:
                    journey_item['id'] = str(uuid.uuid4())
                journey_item['last_edited'] = datetime.now().isoformat()
                
                # Ensure consistency with the campaign data that drove its generation
                journey_item['campaign_name'] = selected_campaign_data.get('name')
                journey_item['cohort'] = selected_campaign_data.get('target_cohort')
                journey_item['lever'] = selected_campaign_data.get('lever_config')
                journey_item['product'] = selected_campaign_data.get('product')

                generated_journeys_to_insert.append(journey_item)
        
        if generated_journeys_to_insert:
            await insert_into_journeys_store(generated_journeys_to_insert)

        return await fetch_from_supabase('journeys')
    
    return await fetch_from_supabase('journeys')

# Removed test_journey_generator and if __name__ == "__main__": block
