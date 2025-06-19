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
from pydantic import BaseModel, Field, SecretStr
from supabase import create_client, Client # Import Supabase client

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

class GrowthLever(BaseModel):
    """Represents a suggested growth lever with its type, details, and rationale."""
    # Added 'id' field for consistency with UI memory and modification
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the growth lever.")
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
        # Removed print statement for production readiness
        return []

# Removed insert_into_growth_levers_store
# Removed update_growth_levers_store
# Removed delete_from_supabase (as it was only used for growth_levers_store in this file)

# --- LLM Call Function ---

async def call_llm_for_growth_levers(
    prompt_text: str,
    api_key: str,
    product_summary: str, # New: summarized product data
    customer_demographics_summary: str, # New: summarized customer demographics
    campaign_performance_summary: str, # New: summarized campaign performance
    audience_types_summary: str, # New: summarized audience types
    current_growth_levers_for_llm: Optional[List[Dict[str, Any]]] = None # Current state from UI memory
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM for growth lever generation/modification, with reduced context."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key)) 
    parser = PydanticOutputParser(pydantic_object=GrowthLevers)
    format_instructions = parser.get_format_instructions()

    template_variables = {
        "format_instructions_var": format_instructions,
        "product_summary": product_summary,
        "customer_demographics_summary": customer_demographics_summary,
        "campaign_performance_summary": campaign_performance_summary,
        "audience_types_summary": audience_types_summary,
    }

    if current_growth_levers_for_llm:
        full_prompt_template_str = """
        Based on the following summarized data and the current growth lever suggestions, please provide an updated list.
        The user's request for modification is: "{user_modification_prompt}"

        Current Growth Levers:
        {current_growth_levers_json}

        Limited Context Data:
        Product Summary: {product_summary}
        Customer Demographics Summary: {customer_demographics_summary}
        Campaign Performance Summary: {campaign_performance_summary}
        Generated Audience Types Summary: {audience_types_summary}

        Modify the growth levers based on the user's request. Maintain the JSON array format as specified by the output instructions.
        If a lever needs to be removed, omit it. If a new lever is requested, add it.
        If an existing lever needs its type, details, or rationale changed, update it.
        Ensure the output strictly adheres to the JSON schema, including setting 'exact_discount_percentage' to null if no specific discount is applicable.
        When modifying an existing growth lever, preserve its 'id'. When adding a new growth lever, the 'id' field can be omitted or set to null, and the system will assign a new one.

        {format_instructions_var}
        """
        template_variables["user_modification_prompt"] = prompt_text
        template_variables["current_growth_levers_json"] = json.dumps(current_growth_levers_for_llm, indent=2)
    else:
        full_prompt_template_str = """
        Based on the provided summarized product data, customer demographics, campaign performance, and generated audience types,
        suggest 3-5 distinct growth levers. If the user has a specific idea for a growth lever, incorporate it
        and suggest related ones. For each lever, provide:
        - A concise 'type' (e.g., 'Targeted Ad Campaign', 'Product Bundling', 'Customer Loyalty Program').
        - 'Details' outlining specific actions or strategies.
        - 'Rationale' explaining why this lever is suitable for the business and target audiences,
          referencing the provided summarized data.
        - If the growth lever involves a specific discount, include 'exact_discount_percentage' (e.g., 10.5 for 10.5%). Set to null if not applicable.
        - The 'id' field should be omitted or set to null for new growth levers.

        User's specific idea (if any): "{user_initial_prompt}"

        Limited Context Data:
        Product Summary: {product_summary}
        Customer Demographics Summary: {customer_demographics_summary}
        Campaign Performance Summary: {campaign_performance_summary}
        Generated Audience Types Summary: {audience_types_summary}

        Ensure the output strictly adheres to the JSON schema.

        {format_instructions_var}
        """
        template_variables["user_initial_prompt"] = prompt_text

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [gl.model_dump(by_alias=True) for gl in response.growth_levers] # Use by_alias to match field names
    except Exception as e:
        # print(f"Error during LLM growth lever call: {e}") # Removed print for production
        return []

async def process_growth_levers(
    user_prompt: str,
    # current_growth_levers will now be passed from UI's session memory
    current_growth_levers: List[Dict[str, Any]], 
    action_type: str, # 'generate', 'update_singular', 'delete_singular'
    growth_lever_id_to_affect: Optional[str] = None # For singular update/delete
) -> List[Dict[str, Any]]: # Returns list of Dicts (the updated state for UI memory)
    """
    Generates, modifies, or deletes growth levers in memory.
    Returns the updated list of growth lever records for UI session memory.
    """
    # Fetch all necessary data from Supabase for *read-only* context
    product_data_raw = await fetch_from_supabase('product_store')
    customer_data_raw = await fetch_from_supabase('cohort_store') 
    campaign_performance_raw = await fetch_from_supabase('campaign_performance_store') 
    audience_types_raw = await fetch_from_supabase('audience_store') 

    # --- Summarize Contextual Data for LLM ---
    product_summary = "Available products: " + ", ".join([p.get('name', '') for p in product_data_raw[:5]]) + "..." if product_data_raw else "No product data."
    customer_demographics_summary = "Customer cohorts include: " + ", ".join([c.get('cohort_name', '') for c in customer_data_raw[:5]]) + " with varying sizes." if customer_data_raw else "No customer demographics."
    campaign_performance_summary = "Recent campaign performance (top 2): " + json.dumps(campaign_performance_raw[:2], indent=2) + "..." if campaign_performance_raw else "No campaign performance data."
    audience_types_summary = "Generated audience types (top 3): " + ", ".join([a.get('title', '') for a in audience_types_raw[:3]]) + "..." if audience_types_raw else "No audience types generated."
    
    # current_growth_levers_from_db is now `current_growth_levers` passed from the UI
    updated_levers_list = list(current_growth_levers) # Create a mutable copy

    if action_type == "delete_singular":
        if growth_lever_id_to_affect:
            updated_levers_list = [gl for gl in updated_levers_list if gl.get('id') != growth_lever_id_to_affect]
            # print(f"Growth Lever {growth_lever_id_to_affect} deleted from session.") # Removed print
        return updated_levers_list

    elif action_type == "update_singular":
        if not growth_lever_id_to_affect:
            # print("No growth lever ID provided for singular update.") # Removed print
            return updated_levers_list

        growth_lever_to_update = next((gl for gl in updated_levers_list if gl.get('id') == growth_lever_id_to_affect), None)
        if growth_lever_to_update:
            llm_response_list = await call_llm_for_growth_levers(
                prompt_text=user_prompt,
                api_key=API_KEY or "",
                product_summary=product_summary,
                customer_demographics_summary=customer_demographics_summary,
                campaign_performance_summary=campaign_performance_summary, 
                audience_types_summary=audience_types_summary,
                current_growth_levers_for_llm=[growth_lever_to_update] # Pass the single lever for modification
            )
            if llm_response_list:
                updated_gl_data = llm_response_list[0] 
                # Find index and replace
                for i, gl in enumerate(updated_levers_list):
                    if gl.get('id') == growth_lever_id_to_affect:
                        updated_gl_data['id'] = growth_lever_id_to_affect # Preserve ID
                        updated_levers_list[i] = updated_gl_data
                        # print(f"Updated Growth Lever {growth_lever_id_to_affect} in session.") # Removed print
                        break
        else:
            pass # print(f"Growth Lever with ID {growth_lever_id_to_affect} not found for update in session.") # Removed print
        return updated_levers_list

    elif action_type == "generate":
        suggested_growth_levers_raw = await call_llm_for_growth_levers(
            prompt_text=user_prompt,
            api_key=API_KEY or "",
            product_summary=product_summary,
            customer_demographics_summary=customer_demographics_summary,
            campaign_performance_summary=campaign_performance_summary, 
            audience_types_summary=audience_types_summary,
            current_growth_levers_for_llm=None # Generating new, no existing context from LLM's perspective
        )

        if suggested_growth_levers_raw:
            # Add newly generated levers to the list
            for suggested_gl in suggested_growth_levers_raw:
                if 'id' not in suggested_gl or not suggested_gl['id']:
                    suggested_gl['id'] = str(uuid.uuid4()) # Ensure new ID for new items
                updated_levers_list.append(suggested_gl)
            # print("Generated new growth levers and added to session.") # Removed print
        return updated_levers_list
    
    # If action_type is not recognized or falls through, return current list
    return updated_levers_list

# Removed test_growth_levers_analyser and if __name__ == "__main__": block
