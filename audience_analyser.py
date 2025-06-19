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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class AudienceType(BaseModel):
    """Represents a suggested audience type with its rationale."""
    type: str = Field(description="A descriptive and concise name for the audience type (e.g., 'Eco-conscious Urban Professionals').")
    rationale: str = Field(description="A detailed explanation of why this audience is suitable, referencing product, customer, and campaign data.")

class AudienceTypes(BaseModel):
    """A list of suggested audience types."""
    audiences: List[AudienceType] = Field(description="A list of distinct audience types.")

# --- Helper Functions ---

async def fetch_from_supabase(table_name: str, id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches data from a specified Supabase table, optionally by ID.
    Added id parameter to allow fetching specific records, reducing data loaded.
    """
    try:
        query = supabase.from_(table_name).select('*')
        if id:
            query = query.eq('id', id)
        response = query.execute()
        if response and hasattr(response, 'data'):
            # print(f"Successfully fetched data from {table_name}. Number of records: {len(response.data)}")
            return response.data
        # print(f"Warning: No data found in {table_name}.")
        return []
    except Exception as e:
        print(f"Error fetching data from Supabase table {table_name}: {e}")
        return []

async def insert_into_audience_store(data: List[Dict]) -> None:
    """Inserts data into the audience_store Supabase table."""
    try:
        response = supabase.from_('audience_store').insert(data).execute()
        if response and hasattr(response, 'data'):
            print(f"Successfully inserted {len(response.data)} records into audience_store.")
        else:
            print(f"No data returned on insert to audience_store, but request sent.")
    except Exception as e:
        print(f"Error inserting data into audience_store: {e}")

async def update_audience_store(id: str, data: Dict) -> None:
    """Updates a record in the audience_store Supabase table."""
    try:
        response = supabase.from_('audience_store').update(data).eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            print(f"Successfully updated record {id} in audience_store.")
        else:
            print(f"No data returned on update for id {id}, but request sent or no matching id found.")
    except Exception as e:
        print(f"Error updating record {id} in audience_store: {e}")

async def delete_from_supabase(table_name: str, id: str) -> None:
    """Deletes a record from a specified Supabase table by ID."""
    try:
        response = supabase.from_(table_name).delete().eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            print(f"Successfully deleted record {id} from {table_name}.")
        else:
            print(f"No data returned on delete for id {id} from {table_name}, but request sent or no matching id found.")
    except Exception as e:
        print(f"Error deleting record {id} from {table_name}: {e}")

def calculate_audience_size(audience_type: Dict[str, str], cohort_data_from_store: List[Dict]) -> int:
    """
    Approximates audience size by finding the closest matching cohort in cohort_data_from_store.
    If no good match, uses the average cohort size.
    """
    if not cohort_data_from_store:
        print("DEBUG: No cohort data available. Audience size defaulting to 0.")
        return 0

    # Calculate average cohort size
    total_cohort_size = sum(c.get('cohort_size', 0) for c in cohort_data_from_store if c.get('cohort_size') is not None)
    num_cohorts = len(cohort_data_from_store)
    average_cohort_size = total_cohort_size / num_cohorts if num_cohorts > 0 else 0

    audience_type_lower = audience_type['type'].lower()
    rationale_lower = audience_type['rationale'].lower()
    
    best_match_score = 0
    matched_cohort_size = 0
    matched_cohort_name = "N/A (No strong match)"

    # Simple keyword matching for similarity
    audience_keywords = set(audience_type_lower.split() + rationale_lower.split())

    for cohort in cohort_data_from_store:
        cohort_name = cohort.get('cohort_name', '').lower()
        cohort_size = cohort.get('cohort_size', 0)
        
        cohort_keywords = set(cohort_name.split())
        
        # Calculate overlap in keywords
        current_score = len(audience_keywords.intersection(cohort_keywords))
        
        # Consider specific phrases or product names from rationale if they appear in query/cohort_name
        # For example, if "beard" in rationale and "beard" in cohort_name
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
            matched_cohort_name = cohort_name

    # Set a threshold for a "good match"
    # This threshold is heuristic and can be adjusted
    # If the best match score is low, use the average.
    # A score of 1 means at least one common word. A higher score means stronger similarity.
    if best_match_score >= 1: # At least one keyword match
        # print(f"DEBUG: Matched '{audience_type['type']}' to cohort '{matched_cohort_name}' with score {best_match_score}. Size: {matched_cohort_size}")
        return matched_cohort_size
    else:
        # print(f"DEBUG: No strong match found for '{audience_type['type']}'. Using average cohort size: {average_cohort_size}")
        return int(average_cohort_size) # Return as integer


async def call_llm_for_audiences(
    prompt_text: str,
    api_key: str, # Added api_key as a direct parameter
    product_summary: str, # New: summarized product data
    customer_demographics_summary: str, # New: summarized customer demographics
    business_context_summary: str, # New: summarized business context
    current_audiences_for_llm: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
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
        For each audience, provide a descriptive name for the 'type' field.

        Limited Context Data:
        Product Summary: {product_summary}
        Customer Demographics Summary: {customer_demographics_summary}
        Business Context Summary: {business_context_summary}

        {format_instructions}
        """
        template_variables["current_audiences_json"] = json.dumps(current_audiences_for_llm, indent=2)
        template_variables["user_modification_prompt"] = prompt_text
    else:
        full_prompt_template = """
        Analyze the provided summarized product data, customer demographics, and business context below.
        Based on the user's input "{user_initial_prompt}", suggest 3-5 distinct audience types.
        For each type, provide a descriptive name and a detailed rationale explaining why this audience is suitable,
        referencing the provided summarized data.
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
        print(f"Error during LLM audience call: {e}")
        return []

async def process_audiences(
    user_prompt: str,
    is_modification: bool = False,
    audience_id_to_affect: Optional[str] = None # New parameter for single update/delete
) -> List[Dict[str, Any]]: 
    """
    Generates, modifies, updates, or deletes audience types and updates the audience_store Supabase table.
    Returns the updated list of audience records from Supabase.
    """
    # Fetch data required for internal logic (like size calculation)
    cohort_data = await fetch_from_supabase('cohort_store')
    current_audiences_from_db = await fetch_from_supabase('audience_store')
    
    # --- Fetch and Summarize Contextual Data for LLM ---
    # Instead of full dumps, get only relevant parts or summaries
    product_data_raw = await fetch_from_supabase('product_store')
    feature_store_data_raw = await fetch_from_supabase('feature_store')
    business_context_data_raw = await fetch_from_supabase('business_context_store')

    # Simple summarization: You can enhance this based on data structure
    product_summary = "Available products: " + ", ".join([p.get('name', '') for p in product_data_raw[:5]]) + "..." if product_data_raw else "No product data."
    feature_summary = "Key features mentioned: " + ", ".join([f.get('name', '') for f in feature_store_data_raw[:5]]) + "..." if feature_store_data_raw else "No feature data."
    business_context_summary = "Business goals/details: " + ", ".join([bc.get('context_detail', '') for bc in business_context_data_raw[:2]]) + "..." if business_context_data_raw else "No business context."
    
    # Customer demographics can be summarized from cohort_data
    customer_demographics_summary = "Customer cohorts include: " + ", ".join([c.get('cohort_name', '') for c in cohort_data[:5]]) + " with varying sizes." if cohort_data else "No customer demographics."

    # Combine summaries for LLM input
    llm_context_product = product_summary
    llm_context_customer_demographics = customer_demographics_summary
    llm_context_business_context = business_context_summary + " " + feature_summary # Features can be part of business context for LLM

    if audience_id_to_affect: # Handling singular update/delete
        if user_prompt.lower().strip() == "delete":
            await delete_from_supabase('audience_store', audience_id_to_affect)
            print(f"Audience {audience_id_to_affect} deleted.")
            return await fetch_from_supabase('audience_store') # Return updated list after deletion
        else: # Singular update
            audience_to_update = next((aud for aud in current_audiences_from_db if aud.get('id') == audience_id_to_affect), None)
            if audience_to_update:
                llm_response_list = await call_llm_for_audiences(
                    prompt_text=user_prompt,
                    api_key=API_KEY or "",
                    product_summary=llm_context_product,
                    customer_demographics_summary=llm_context_customer_demographics,
                    business_context_summary=llm_context_business_context,
                    current_audiences_for_llm=[{"type": audience_to_update.get('title', ''), "rationale": audience_to_update.get('rationale', '')}]
                )
                if llm_response_list:
                    updated_aud_data = llm_response_list[0]
                    size = calculate_audience_size(updated_aud_data, cohort_data)
                    data_to_update = {
                        "title": updated_aud_data['type'],
                        "audience_size": size,
                        "rationale": updated_aud_data['rationale']
                    }
                    await update_audience_store(audience_id_to_affect, data_to_update)
                    print(f"\n--- Updated Single Audience (ID: {audience_id_to_affect}) ---")
                    print(f"Title: {data_to_update.get('title')}")
                    print(f"Audience Size: {data_to_update.get('audience_size')}")
                    print(f"Rationale: {data_to_update.get('rationale')}")
                    print("-" * 30)
                    return await fetch_from_supabase('audience_store')
                else:
                    print(f"Could not generate update for audience ID {audience_id_to_affect}.")
            else:
                print(f"Audience with ID {audience_id_to_affect} not found for update.")
            return await fetch_from_supabase('audience_store')

    # Handling general generation or modification (multiple audiences via LLM's full generation)
    audiences_for_llm = None
    if is_modification:
        audiences_for_llm = [
            {"type": aud.get('title', ''), "rationale": aud.get('rationale', '')}
            for aud in current_audiences_from_db
        ]

    suggested_audiences_raw = await call_llm_for_audiences(
        prompt_text=user_prompt,
        api_key=API_KEY or "",
        product_summary=llm_context_product,
        customer_demographics_summary=llm_context_customer_demographics,
        business_context_summary=llm_context_business_context,
        current_audiences_for_llm=audiences_for_llm
    )

    if suggested_audiences_raw:
        existing_audiences_map = {aud.get('title', '').lower(): aud for aud in current_audiences_from_db}
        audiences_to_keep_ids = set()
        
        for suggested_aud in suggested_audiences_raw:
            title = suggested_aud['type']
            rationale = suggested_aud['rationale']
            size = calculate_audience_size(suggested_aud, cohort_data)
            
            data_to_process = {
                "title": title,
                "audience_size": size,
                "rationale": rationale
            }

            if title.lower() in existing_audiences_map:
                existing_aud_db = existing_audiences_map[title.lower()]
                existing_id = existing_aud_db.get('id')
                if existing_id:
                    await update_audience_store(existing_id, data_to_process)
                    audiences_to_keep_ids.add(existing_id)
            else:
                new_id = str(uuid.uuid4())
                data_to_process['id'] = new_id
                await insert_into_audience_store([data_to_process])
                audiences_to_keep_ids.add(new_id)
        
        if is_modification:
            for existing_aud in current_audiences_from_db:
                if existing_aud.get('id') not in audiences_to_keep_ids:
                    await delete_from_supabase('audience_store', existing_aud.get('id', ''))

        updated_audiences = await fetch_from_supabase('audience_store')
        
        print("\n--- Generated/Modified Audiences ---")
        if updated_audiences:
            for aud in updated_audiences:
                print(f"ID: {aud.get('id')}")
                print(f"Title: {aud.get('title')}")
                print(f"Audience Size: {aud.get('audience_size')}")
                print(f"Rationale: {aud.get('rationale')}")
                print("-" * 30)
        else:
            print("No audiences generated or found.")
            
        return updated_audiences
    return []

# --- Test Function ---
async def test_audience_analyser():
    print("Welcome to the Audience Analyser Test!")
    while True:
        action = input("\nChoose action: (g)enerate new, (u)pdate singular, (d)elete singular, or (q)uit: ").lower()
        
        if action == 'q':
            break
        elif action == 'g':
            user_prompt = input("Enter your prompt for audience generation: ")
            await process_audiences(user_prompt=user_prompt, is_modification=False)
        elif action == 'u':
            current_audiences = await fetch_from_supabase('audience_store')
            if not current_audiences:
                print("No audiences found to update.")
                continue
            print("\nCurrent Audiences:")
            for aud in current_audiences:
                print(f"  ID: {aud.get('id')}, Title: {aud.get('title')}")
            
            aud_id = input("Enter the ID of the audience to update: ")
            update_prompt = input(f"Enter the new details/prompt for audience ID {aud_id}: ")
            await process_audiences(user_prompt=update_prompt, audience_id_to_affect=aud_id)
        elif action == 'd':
            current_audiences = await fetch_from_supabase('audience_store')
            if not current_audiences:
                print("No audiences found to delete.")
                continue
            print("\nCurrent Audiences:")
            for aud in current_audiences:
                print(f"  ID: {aud.get('id')}, Title: {aud.get('title')}")
            
            aud_id = input("Enter the ID of the audience to delete: ")
            confirm = input(f"Are you sure you want to delete audience with ID {aud_id}? (yes/no): ").lower()
            if confirm == 'yes':
                await process_audiences(user_prompt="delete", audience_id_to_affect=aud_id)
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid action. Please choose from the available options.")

# This block allows you to run the test function directly when the script is executed.
if __name__ == "__main__":
    import asyncio
    
    if not API_KEY:
        print("\nERROR: OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        exit(1)

    asyncio.run(test_audience_analyser())
