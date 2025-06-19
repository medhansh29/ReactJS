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
    # Added 'id' field for consistency with Supabase storage and modification
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

# --- Supabase Helper Functions ---

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

async def insert_into_growth_levers_store(data: List[Dict]) -> None:
    """Inserts data into the growth_levers_store Supabase table."""
    try:
        response = supabase.from_('growth_levers_store').insert(data).execute()
        if response and hasattr(response, 'data'):
            print(f"Successfully inserted {len(response.data)} records into growth_levers_store.")
        else:
            print(f"No data returned on insert to growth_levers_store, but request sent.")
    except Exception as e:
        print(f"Error inserting data into growth_levers_store: {e}")

async def update_growth_levers_store(id: str, data: Dict) -> None:
    """Updates a record in the growth_levers_store Supabase table."""
    try:
        response = supabase.from_('growth_levers_store').update(data).eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            print(f"Successfully updated record {id} in growth_levers_store.")
        else:
            print(f"No data returned on update for id {id}, but request sent or no matching id found.")
    except Exception as e:
        print(f"Error updating record {id} in growth_levers_store: {e}")

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

# --- LLM Call Function ---

async def call_llm_for_growth_levers(
    prompt_text: str,
    api_key: str,
    product_summary: str, # New: summarized product data
    customer_demographics_summary: str, # New: summarized customer demographics
    campaign_performance_summary: str, # New: summarized campaign performance
    audience_types_summary: str, # New: summarized audience types
    current_growth_levers_for_llm: Optional[List[Dict[str, Any]]] = None
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
        print(f"Error during LLM growth lever call: {e}")
        return []

async def process_growth_levers(
    user_prompt: str,
    is_modification: bool = False,
    growth_lever_id_to_affect: Optional[str] = None # For singular update/delete
) -> List[Dict[str, Any]]: # Returns list of Dicts, not Dict
    """
    Generates, modifies, updates, or deletes growth levers and updates the growth_levers_store Supabase table.
    Returns the updated list of growth lever records from Supabase.
    """
    # Fetch all necessary data from Supabase for context
    product_data_raw = await fetch_from_supabase('product_store')
    customer_data_raw = await fetch_from_supabase('cohort_store') # Assuming customer data is in cohort_store or similar
    campaign_performance_raw = await fetch_from_supabase('campaign_performance_store') 
    audience_types_raw = await fetch_from_supabase('audience_store') # Get audiences from audience_store

    # --- Summarize Contextual Data for LLM ---
    product_summary = "Available products: " + ", ".join([p.get('name', '') for p in product_data_raw[:5]]) + "..." if product_data_raw else "No product data."
    customer_demographics_summary = "Customer cohorts include: " + ", ".join([c.get('cohort_name', '') for c in customer_data_raw[:5]]) + " with varying sizes." if customer_data_raw else "No customer demographics."
    campaign_performance_summary = "Recent campaign performance (top 2): " + json.dumps(campaign_performance_raw[:2], indent=2) + "..." if campaign_performance_raw else "No campaign performance data."
    audience_types_summary = "Generated audience types (top 3): " + ", ".join([a.get('title', '') for a in audience_types_raw[:3]]) + "..." if audience_types_raw else "No audience types generated."
    
    current_growth_levers_from_db = await fetch_from_supabase('growth_levers_store')
    
    if growth_lever_id_to_affect: # Handling singular update/delete
        if user_prompt.lower().strip() == "delete":
            await delete_from_supabase('growth_levers_store', growth_lever_id_to_affect)
            print(f"Growth Lever {growth_lever_id_to_affect} deleted.")
            return await fetch_from_supabase('growth_levers_store') # Return updated list after deletion
        else: # Singular update
            # Find the specific growth lever to update
            growth_lever_to_update = next((gl for gl in current_growth_levers_from_db if gl.get('id') == growth_lever_id_to_affect), None)
            if growth_lever_to_update:
                # Use LLM to generate a single updated growth lever based on prompt
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
                    updated_gl_data = llm_response_list[0] # Expecting only one modified lever
                    
                    # Ensure the ID from the database is preserved
                    updated_gl_data['id'] = growth_lever_id_to_affect 

                    # Prepare data to update (remove any fields not matching DB schema if necessary)
                    data_to_update = {
                        "id": updated_gl_data.get('id'),
                        "type": updated_gl_data.get('type'),
                        "details": updated_gl_data.get('details'),
                        "rationale": updated_gl_data.get('rationale'),
                        "exact_discount_percentage": updated_gl_data.get('exact_discount_percentage')
                    }
                    await update_growth_levers_store(growth_lever_id_to_affect, data_to_update)
                    print(f"\n--- Updated Single Growth Lever (ID: {growth_lever_id_to_affect}) ---")
                    print(f"Type: {data_to_update.get('type')}")
                    print(f"Details: {data_to_update.get('details')}")
                    print(f"Rationale: {data_to_update.get('rationale')}")
                    print(f"Discount: {data_to_update.get('exact_discount_percentage')}")
                    print("-" * 30)
                    return await fetch_from_supabase('growth_levers_store') # Return updated list
                else:
                    print(f"Could not generate update for growth lever ID {growth_lever_id_to_affect}.")
            else:
                print(f"Growth Lever with ID {growth_lever_id_to_affect} not found for update.")
            return await fetch_from_supabase('growth_levers_store') # Return current list if update failed

    # Handling general generation or modification (multiple growth levers via LLM's full generation)
    growth_levers_for_llm_context = None
    if is_modification: # If it's a modification request, provide current levers to LLM
        growth_levers_for_llm_context = current_growth_levers_from_db

    suggested_growth_levers_raw = await call_llm_for_growth_levers(
        prompt_text=user_prompt,
        api_key=API_KEY or "",
        product_summary=product_summary,
        customer_demographics_summary=customer_demographics_summary,
        campaign_performance_summary=campaign_performance_summary, 
        audience_types_summary=audience_types_summary,
        current_growth_levers_for_llm=growth_levers_for_llm_context # Pass existing levers for LLM context
    )

    if suggested_growth_levers_raw:
        # Create a map of existing growth levers by their ID for efficient lookup during update/insert
        existing_levers_map = {gl.get('id'): gl for gl in current_growth_levers_from_db if gl.get('id')}
        
        levers_to_keep_ids = set() # Track IDs of growth levers that are still present or newly added
        data_to_insert = []
        
        for suggested_gl in suggested_growth_levers_raw:
            gl_id = suggested_gl.get('id')
            
            data_to_process = {
                "type": suggested_gl.get('type'),
                "details": suggested_gl.get('details'),
                "rationale": suggested_gl.get('rationale'),
                "exact_discount_percentage": suggested_gl.get('exact_discount_percentage')
            }

            if gl_id and gl_id in existing_levers_map:
                # Update existing growth lever
                await update_growth_levers_store(str(gl_id), data_to_process)
                levers_to_keep_ids.add(gl_id)
            else:
                # Insert new growth lever (assign a new UUID)
                new_id = str(uuid.uuid4())
                data_to_process['id'] = new_id
                data_to_insert.append(data_to_process)
                levers_to_keep_ids.add(new_id)
        
        if data_to_insert:
            await insert_into_growth_levers_store(data_to_insert)
        
        # Delete growth levers that were in DB but not returned by LLM (meaning they should be removed)
        if is_modification: # Only delete if it was a modification prompt to the LLM
            for existing_gl in current_growth_levers_from_db:
                if existing_gl.get('id') not in levers_to_keep_ids:
                    await delete_from_supabase('growth_levers_store', existing_gl.get('id', ''))

        updated_growth_levers = await fetch_from_supabase('growth_levers_store')
        
        print("\n--- Generated/Modified Growth Levers ---")
        if updated_growth_levers:
            for gl in updated_growth_levers:
                print(f"ID: {gl.get('id')}")
                print(f"Type: {gl.get('type')}")
                print(f"Details: {gl.get('details')}")
                print(f"Rationale: {gl.get('rationale')}")
                print(f"Discount: {gl.get('exact_discount_percentage')}")
                print("-" * 30)
        else:
            print("No growth levers generated or found.")
            
        return updated_growth_levers
    return []

# --- Test Function ---
async def test_growth_levers_analyser():
    print("Welcome to the Growth Levers Analyser Test!")
    while True:
        action = input("\nChoose action: (g)enerate new, (u)pdate singular, (d)elete singular, or (q)uit: ").lower()
        
        if action == 'q':
            break
        elif action == 'g':
            user_prompt = input("Enter your prompt for growth lever generation: ")
            # is_modification=False for initial generation
            await process_growth_levers(user_prompt=user_prompt, is_modification=False) 
        elif action == 'u':
            current_levers = await fetch_from_supabase('growth_levers_store')
            if not current_levers:
                print("No growth levers found to update.")
                continue
            print("\nCurrent Growth Levers:")
            for gl in current_levers:
                print(f"  ID: {gl.get('id')}, Type: {gl.get('type')}")
            
            gl_id = input("Enter the ID of the growth lever to update: ")
            update_prompt = input(f"Enter the new details/prompt for growth lever ID {gl_id}: ")
            await process_growth_levers(user_prompt=update_prompt, growth_lever_id_to_affect=gl_id)
        elif action == 'd':
            current_levers = await fetch_from_supabase('growth_levers_store')
            if not current_levers:
                print("No growth levers found to delete.")
                continue
            print("\nCurrent Growth Levers:")
            for gl in current_levers:
                print(f"  ID: {gl.get('id')}, Type: {gl.get('type')}")
            
            gl_id = input("Enter the ID of the growth lever to delete: ")
            confirm = input(f"Are you sure you want to delete growth lever with ID {gl_id}? (yes/no): ").lower()
            if confirm == 'yes':
                await process_growth_levers(user_prompt="delete", growth_lever_id_to_affect=gl_id)
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid action. Please choose from the available options.")

# This block allows you to run the test function directly when the script is executed.
if __name__ == "__main__":
    import asyncio
    
    if not API_KEY:
        print("\nERROR: OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        exit(1) # Exit if API key is missing

    asyncio.run(test_growth_levers_analyser())
