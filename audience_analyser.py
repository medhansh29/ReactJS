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

class AudienceType(BaseModel):
    """Represents a suggested audience type with its rationale."""
    # Added 'id' field for consistency with UI memory and modification
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the audience type.")
    type: str = Field(description="A descriptive and concise name for the audience type (e.g., 'Eco-conscious Urban Professionals').")
    rationale: str = Field(description="A detailed explanation of why this audience is suitable, referencing product, customer, and campaign data.")
    audience_size: Optional[int] = Field(default=None, description="Approximate size of the audience.") # Added audience_size for UI


class AudienceTypes(BaseModel):
    """A list of suggested audience types."""
    audiences: List[AudienceType] = Field(description="A list of distinct audience types.")

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

# Removed insert_into_audience_store
# Removed update_audience_store
# Removed delete_from_supabase (as it was only used for audience_store in this file)

def calculate_audience_size(audience_type: Dict[str, str], cohort_data_from_store: List[Dict]) -> int:
    """
    Approximates audience size by finding the closest matching cohort in cohort_data_from_store.
    If no good match, uses the average cohort size.
    """
    if not cohort_data_from_store:
        return 0

    # Calculate average cohort size
    total_cohort_size = sum(c.get('cohort_size', 0) for c in cohort_data_from_store if c.get('cohort_size') is not None)
    num_cohorts = len(cohort_data_from_store)
    average_cohort_size = total_cohort_size / num_cohorts if num_cohorts > 0 else 0

    audience_type_lower = audience_type['type'].lower()
    rationale_lower = audience_type['rationale'].lower()
    
    best_match_score = 0
    matched_cohort_size = 0

    audience_keywords = set(audience_type_lower.split() + rationale_lower.split())

    for cohort in cohort_data_from_store:
        cohort_name = cohort.get('cohort_name', '').lower()
        cohort_size = cohort.get('cohort_size', 0)
        
        cohort_keywords = set(cohort_name.split())
        
        current_score = len(audience_keywords.intersection(cohort_keywords))
        
        # Consider specific phrases or product names from rationale if they appear in query/cohort_name
        if "beard" in rationale_lower and "beard" in cohort_name:
            current_score += 1
        if "skincare" in rationale_lower and "skincare" in cohort_name:
            current_score += 1
        if "sunscreen" in rationale_lower and "sunscreen" in cohort_name:
            current_score += 1
        if "bb cream" in rationale_lower and "bb cream" in cohort_name:
            current_score += 1

        if current_score > best_match_score:
            best_match_score = current_score
            matched_cohort_size = cohort_size

    if best_match_score >= 1: # At least one keyword match
        return matched_cohort_size
    else:
        return int(average_cohort_size)


async def call_llm_for_audiences(
    prompt_text: str,
    api_key: str,
    product_summary: str,
    customer_demographics_summary: str,
    business_context_summary: str,
    current_audiences_for_llm: Optional[List[Dict[str, Any]]] = None # Current state from UI memory
) -> List[Dict[str, Any]]:
    """Calls the OpenAI LLM for audience generation/modification, with reduced context."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key)) 
    parser = PydanticOutputParser(pydantic_object=AudienceTypes)
    format_instructions = parser.get_format_instructions()

    template_variables = {
        "format_instructions": format_instructions,
        "product_summary": product_summary,
        "customer_demographics_summary": customer_demographics_summary,
        "business_context_summary": business_context_summary,
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
        For each audience, provide a descriptive name for the 'type' field and an 'id'.

        Limited Context Data:
        Product Summary: {product_summary}
        Customer Demographics Summary: {customer_demographics_summary}
        Business Context Summary: {business_context_summary}

        {format_instructions}
        """
        # Ensure 'id' is included when passing current audiences to LLM for modification
        audiences_with_id_for_llm = [
            {"id": aud.get('id'), "type": aud.get('type', ''), "rationale": aud.get('rationale', '')}
            for aud in current_audiences_for_llm
        ]
        template_variables["current_audiences_json"] = json.dumps(audiences_with_id_for_llm, indent=2)
        template_variables["user_modification_prompt"] = prompt_text
    else:
        full_prompt_template = """
        Analyze the provided summarized product data, customer demographics, and business context below.
        Based on the user's input "{user_initial_prompt}", suggest 3-5 distinct audience types.
        For each type, provide a descriptive name and a detailed rationale explaining why this audience is suitable,
        referencing the provided summarized data.
        The 'id' field should be omitted or set to null for new audience types.
        The output must strictly adhere to the JSON schema below.

        Limited Context Data:
        Product Summary: {product_summary}
        Customer Demographics Summary: {customer_demographics_summary}
        Business Context Summary: {business_context_summary}

        {format_instructions}
        """
        template_variables["user_initial_prompt"] = prompt_text

    prompt = ChatPromptTemplate.from_template(full_prompt_template)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [aud.model_dump() for aud in response.audiences]
    except Exception as e:
        return []

async def process_audiences(
    user_prompt: str,
    current_audiences: List[Dict[str, Any]], # Now passed from UI's session memory
    action_type: str, # 'generate', 'update_singular', 'delete_singular'
    audience_id_to_affect: Optional[str] = None # For singular update/delete
) -> List[Dict[str, Any]]: 
    """
    Generates, modifies, or deletes audience types in memory.
    Returns the updated list of audience records for UI session memory.
    """
    # Fetch data required for internal logic (like size calculation)
    cohort_data = await fetch_from_supabase('cohort_store')
    
    # --- Fetch and Summarize Contextual Data for LLM ---
    product_data_raw = await fetch_from_supabase('product_store')
    feature_store_data_raw = await fetch_from_supabase('feature_store')
    business_context_data_raw = await fetch_from_supabase('business_context_store')

    # Simple summarization: You can enhance this based on data structure
    product_summary = "Available products: " + ", ".join([p.get('name', '') for p in product_data_raw[:5]]) + "..." if product_data_raw else "No product data."
    feature_summary = "Key features mentioned: " + ", ".join([f.get('name', '') for f in feature_store_data_raw[:5]]) + "..." if feature_store_data_raw else "No feature data."
    business_context_summary = "Business goals/details: " + ", ".join([bc.get('context_detail', '') for bc in business_context_data_raw[:2]]) + "..." if business_context_data_raw else "No business context."
    
    customer_demographics_summary = "Customer cohorts include: " + ", ".join([c.get('cohort_name', '') for c in cohort_data[:5]]) + " with varying sizes." if cohort_data else "No customer demographics."

    llm_context_product = product_summary
    llm_context_customer_demographics = customer_demographics_summary
    llm_context_business_context = business_context_summary + " " + feature_summary

    updated_audiences_list = list(current_audiences) # Create a mutable copy of session memory

    if action_type == "delete_singular":
        if audience_id_to_affect:
            updated_audiences_list = [aud for aud in updated_audiences_list if aud.get('id') != audience_id_to_affect]
        return updated_audiences_list

    elif action_type == "update_singular":
        if not audience_id_to_affect:
            return updated_audiences_list

        audience_to_update = next((aud for aud in updated_audiences_list if aud.get('id') == audience_id_to_affect), None)
        if audience_to_update:
            llm_response_list = await call_llm_for_audiences(
                prompt_text=user_prompt,
                api_key=API_KEY or "",
                product_summary=llm_context_product,
                customer_demographics_summary=llm_context_customer_demographics,
                business_context_summary=llm_context_business_context,
                current_audiences_for_llm=[audience_to_update] # Pass the single audience for modification
            )
            if llm_response_list:
                updated_aud_data = llm_response_list[0]
                size = calculate_audience_size(updated_aud_data, cohort_data)
                
                # Update the audience in the list
                for i, aud in enumerate(updated_audiences_list):
                    if aud.get('id') == audience_id_to_affect:
                        updated_aud_data['id'] = audience_id_to_affect # Preserve ID
                        updated_aud_data['audience_size'] = size # Update size
                        updated_audiences_list[i] = updated_aud_data
                        break
        return updated_audiences_list

    elif action_type == "generate":
        suggested_audiences_raw = await call_llm_for_audiences(
            prompt_text=user_prompt,
            api_key=API_KEY or "",
            product_summary=llm_context_product,
            customer_demographics_summary=llm_context_customer_demographics,
            business_context_summary=llm_context_business_context,
            current_audiences_for_llm=None # Generating new, no existing context from LLM's perspective
        )

        if suggested_audiences_raw:
            for suggested_aud in suggested_audiences_raw:
                if 'id' not in suggested_aud or not suggested_aud['id']:
                    suggested_aud['id'] = str(uuid.uuid4()) # Ensure new ID for new items
                
                # Calculate size for newly generated audiences
                suggested_aud['audience_size'] = calculate_audience_size(suggested_aud, cohort_data)
                updated_audiences_list.append(suggested_aud)
        return updated_audiences_list
    
    return updated_audiences_list # Return current list if action_type is not recognized

# Removed test_audience_analyser and if __name__ == "__main__": block
