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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class CampaignIdea(BaseModel):
    """Represents a generated campaign idea, matching the 'campaigns' Supabase table schema."""
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
    audience_id: str = Field(description="The ID of the target audience used for this campaign (from audience_store).")
    growth_lever_id: str = Field(description="The ID of the growth lever used for this campaign (from growth_levers_store).")
    product_id: str = Field(description="The ID of the product targeted by this campaign (from product_store).")


class CampaignIdeas(BaseModel):
    """A list of generated campaign ideas."""
    campaign_ideas: List[CampaignIdea] = Field(description="A list of 2-3 distinct campaign ideas.")

# --- Supabase Helper Functions ---

async def fetch_from_supabase(table_name: str, id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches data from a specified Supabase table, optionally by ID."""
    try:
        query = supabase.from_(table_name).select('*')
        if id:
            query = query.eq('id', id)
        response = query.execute()
        if response and hasattr(response, 'data'):
            print(f"Successfully fetched data from {table_name}. Number of records: {len(response.data)}")
            return response.data
        print(f"Warning: No data found in {table_name}.")
        return []
    except Exception as e:
        print(f"Error fetching data from Supabase table {table_name}: {e}")
        return []

async def insert_into_campaigns_store(data: List[Dict]) -> None:
    """Inserts data into the 'campaigns' Supabase table."""
    try:
        response = supabase.from_('campaigns').insert(data).execute()
        if response and hasattr(response, 'data'):
            print(f"Successfully inserted {len(response.data)} records into campaigns table.")
        else:
            print(f"No data returned on insert to campaigns, but request sent.")
    except Exception as e:
        print(f"Error inserting data into campaigns table: {e}")

async def update_campaigns_store(id: str, data: Dict) -> None:
    """Updates a record in the 'campaigns' Supabase table."""
    try:
        response = supabase.from_('campaigns').update(data).eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            print(f"Successfully updated record {id} in campaigns table.")
        else:
            print(f"No data returned on update for id {id}, but request sent or no matching id found.")
    except Exception as e:
        print(f"Error updating record {id} in campaigns table: {e}")

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

async def call_llm_for_campaign_ideas(
    api_key: str,
    user_prompt: str = "",
    selected_audience_data: Optional[Dict[str, Any]] = None,
    selected_growth_lever_data: Optional[Dict[str, Any]] = None,
    selected_product_data: Optional[Dict[str, Any]] = None,
    current_campaign_ideas_for_llm: Optional[List[Dict[str, Any]]] = None # For modification mode
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
        template_variables["current_campaign_ideas_json"] = json.dumps(current_campaign_ideas_for_llm, indent=2)
    else: # Generation mode
        # These fields are guaranteed to be present if selected_audience_data etc. are not None
        audience_title = selected_audience_data.get('title', 'N/A') if selected_audience_data else 'N/A'
        growth_lever_type = selected_growth_lever_data.get('type', 'N/A') if selected_growth_lever_data else 'N/A'
        product_name = selected_product_data.get('name', 'N/A') if selected_product_data else 'N/A'
        discount_percentage = selected_growth_lever_data.get('exact_discount_percentage') if selected_growth_lever_data else None

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
                "audience_rationale": selected_audience_data.get('rationale') if selected_audience_data else None,
                "audience_size": selected_audience_data.get('audience_size') if selected_audience_data else None,
                "audience_id": selected_audience_data.get('id') if selected_audience_data else None,

                "growth_lever_type": growth_lever_type,
                "growth_lever_details": selected_growth_lever_data.get('details') if selected_growth_lever_data else None,
                "growth_lever_rationale": selected_growth_lever_data.get('rationale') if selected_growth_lever_data else None,
                "growth_lever_id": selected_growth_lever_data.get('id') if selected_growth_lever_data else None,
                "discount_percentage": discount_percentage, # Already handled for None

                "product_name": product_name,
                "product_id": selected_product_data.get('id') if selected_product_data else None,
                "product_description": selected_product_data.get('description') if selected_product_data else None,
            }
        )

    prompt = ChatPromptTemplate.from_template(full_prompt_template_str)
    chain = prompt | llm | parser
    try:
        response = await chain.ainvoke(template_variables)
        return [idea.model_dump() for idea in response.campaign_ideas]
    except ValidationError as e:
        print(f"Validation Error during LLM campaign idea call: {e.errors()}")
        print(f"LLM probably did not return the expected JSON format. Raw error: {e}")
        return []
    except Exception as e:
        print(f"Error during LLM campaign idea call: {e}")
        return []

async def process_campaign_ideas(
    user_prompt: str = "", # For initial generation prompt or modification prompt
    selected_combinations: Optional[List[Dict[str, str]]] = None, # For generation: [{'audience_id': 'uuid', 'growth_lever_id': 'uuid', 'product_id': 'uuid'}]
    campaign_id_to_affect: Optional[str] = None, # For singular update/delete
    action_type: str = "generate" # 'generate', 'update_singular', 'delete_singular'
) -> List[Dict[str, Any]]:
    """
    Generates, updates, or deletes campaign ideas and updates the 'campaigns' Supabase table.
    Returns the updated list of campaign records from Supabase.
    """
    # Fetch only the campaigns directly relevant to the current operation, if any
    current_campaigns_from_db = await fetch_from_supabase('campaigns')

    if action_type == "delete_singular":
        if campaign_id_to_affect:
            await delete_from_supabase('campaigns', campaign_id_to_affect)
        return await fetch_from_supabase('campaigns')

    elif action_type == "update_singular":
        if not campaign_id_to_affect:
            print("No campaign ID provided for singular update.")
            return await fetch_from_supabase('campaigns')

        campaign_to_update = next((c for c in current_campaigns_from_db if c.get('id') == campaign_id_to_affect), None)
        if not campaign_to_update:
            print(f"Campaign with ID {campaign_id_to_affect} not found for update.")
            return await fetch_from_supabase('campaigns')

        # Fetch only the specific linked audience, lever, and product for the campaign being updated
        selected_audience = (await fetch_from_supabase('audience_store', campaign_to_update.get('audience_id')))[0] if campaign_to_update.get('audience_id') else None
        selected_growth_lever = (await fetch_from_supabase('growth_levers_store', campaign_to_update.get('growth_lever_id')))[0] if campaign_to_update.get('growth_lever_id') else None
        selected_product = (await fetch_from_supabase('product_store', campaign_to_update.get('product_id')))[0] if campaign_to_update.get('product_id') else None

        if not all([selected_audience, selected_growth_lever, selected_product]):
            print(f"ERROR: Linked audience, growth lever, or product data missing for campaign ID {campaign_id_to_affect}.")
            return await fetch_from_supabase('campaigns')

        # Pass the existing campaign idea to the LLM for modification
        llm_response_list = await call_llm_for_campaign_ideas(
            api_key=API_KEY or "",
            user_prompt=user_prompt, # User's specific modification request
            selected_audience_data=selected_audience, # Provide context of linked entities
            selected_growth_lever_data=selected_growth_lever,
            selected_product_data=selected_product,
            current_campaign_ideas_for_llm=[campaign_to_update] # Send the specific campaign to modify
        )

        if llm_response_list:
            updated_campaign_data = llm_response_list[0] # Expecting only one updated campaign
            
            # Preserve the original ID for the update operation
            updated_campaign_data['id'] = campaign_id_to_affect 
            
            # Ensure consistency with DB schema and chosen IDs
            data_to_update = {
                "id": updated_campaign_data.get('id'),
                "name": updated_campaign_data.get('name'),
                "description": updated_campaign_data.get('description'),
                "hypothesis": updated_campaign_data.get('hypothesis'),
                "target_cohort": updated_campaign_data.get('target_cohort'),
                "lever_config": updated_campaign_data.get('lever_config'),
                "product": updated_campaign_data.get('product'),
                "discount_percentage": updated_campaign_data.get('discount_percentage'),
                "audience_id": updated_campaign_data.get('audience_id'), # Ensure these are correct from LLM
                "growth_lever_id": updated_campaign_data.get('growth_lever_id'),
                "product_id": updated_campaign_data.get('product_id'),
            }
            await update_campaigns_store(campaign_id_to_affect, data_to_update)
            print(f"\n--- Updated Single Campaign (ID: {campaign_id_to_affect}) ---")
            print(f"Name: {data_to_update.get('name')}")
            print(f"Description: {data_to_update.get('description')}")
            print(f"Target Audience: {data_to_update.get('target_cohort')} (ID: {data_to_update.get('audience_id')})")
            print(f"Growth Lever: {data_to_update.get('lever_config')} (ID: {data_to_update.get('growth_lever_id')})")
            print(f"Product: {data_to_update.get('product')} (ID: {data_to_update.get('product_id')})")
            print(f"Discount: {data_to_update.get('discount_percentage')}")
            print("-" * 30)
            return await fetch_from_supabase('campaigns')
        else:
            print(f"Could not generate update for campaign ID {campaign_id_to_affect}.")
        return await fetch_from_supabase('campaigns')

    elif action_type == "generate":
        if not selected_combinations:
            print("No combinations selected for campaign generation.")
            return await fetch_from_supabase('campaigns')

        generated_campaigns_to_insert = []
        for combo in selected_combinations:
            aud_id = combo['audience_id']
            gl_id = combo['growth_lever_id']
            prod_id = combo['product_id']

            # Fetch specific data for the current combination only
            selected_audience = (await fetch_from_supabase('audience_store', aud_id))[0] if aud_id else None
            selected_growth_lever = (await fetch_from_supabase('growth_levers_store', gl_id))[0] if gl_id else None
            selected_product = (await fetch_from_supabase('product_store', prod_id))[0] if prod_id else None

            if not all([selected_audience, selected_growth_lever, selected_product]):
                print(f"Warning: Skipping combination due to missing data: Audience ID {aud_id}, GL ID {gl_id}, Product ID {prod_id}.")
                continue

            generated_ideas = await call_llm_for_campaign_ideas(
                api_key=API_KEY or "",
                user_prompt=user_prompt,
                selected_audience_data=selected_audience,
                selected_growth_lever_data=selected_growth_lever,
                selected_product_data=selected_product,
                current_campaign_ideas_for_llm=None # Not modifying existing, generating new
            )
            
            if generated_ideas:
                for idea in generated_ideas:
                    # Populate the IDs from the selected combination
                    idea['audience_id'] = aud_id
                    idea['growth_lever_id'] = gl_id
                    idea['product_id'] = prod_id
                    # The 'id' for a new campaign is generated by the Pydantic model's default_factory
                    generated_campaigns_to_insert.append(idea)
        
        if generated_campaigns_to_insert:
            await insert_into_campaigns_store(generated_campaigns_to_insert)

        updated_campaigns = await fetch_from_supabase('campaigns')
        
        print("\n--- Generated Campaigns ---")
        if updated_campaigns:
            for camp in updated_campaigns:
                print(f"ID: {camp.get('id')}")
                print(f"Name: {camp.get('name')}")
                print(f"Description: {camp.get('description')}")
                print(f"Hypothesis: {camp.get('hypothesis')}")
                print(f"Target Audience: {camp.get('target_cohort')} (ID: {camp.get('audience_id')})")
                print(f"Growth Lever: {camp.get('lever_config')} (ID: {camp.get('growth_lever_id')})")
                print(f"Product: {camp.get('product')} (ID: {camp.get('product_id')})")
                print(f"Discount: {camp.get('discount_percentage')}")
                print("-" * 30)
        else:
            print("No campaigns generated or found.")
            
        return updated_campaigns
    
    return await fetch_from_supabase('campaigns')


# --- Test Function ---
async def test_campaign_generator():
    print("Welcome to the Campaign Generator Test!")
    while True:
        action = input("\nChoose action: (g)enerate new, (u)pdate singular, (d)elete singular, or (q)uit: ").lower()
        
        if action == 'q':
            break
        elif action == 'g':
            audiences = await fetch_from_supabase('audience_store')
            growth_levers = await fetch_from_supabase('growth_levers_store')
            products = await fetch_from_supabase('product_store')

            if not audiences:
                print("No audiences found. Please generate audiences first using audience_analyser.py.")
                continue
            if not growth_levers:
                print("No growth levers found. Please generate growth levers first using growth_levers.py.")
                continue
            if not products:
                print("No products found in product_store.")
                continue

            print("\nAvailable Audiences:")
            for i, aud in enumerate(audiences):
                print(f"   {i+1}. ID: {aud.get('id')}, Title: {aud.get('title')}, Size: {aud.get('audience_size')}")

            print("\nAvailable Growth Levers:")
            for i, gl in enumerate(growth_levers):
                print(f"   {i+1}. ID: {gl.get('id')}, Type: {gl.get('type')}")
            
            print("\nAvailable Products:")
            for i, prod in enumerate(products):
                print(f"   {i+1}. ID: {prod.get('id')}, Name: {prod.get('name')}") # Assuming 'name' column for product

            selected_combinations = []
            while True:
                aud_choice = input("Enter audience number (or 'done' to finish choosing audiences): ")
                if aud_choice.lower() == 'done':
                    break
                try:
                    aud_index = int(aud_choice) - 1
                    if 0 <= aud_index < len(audiences):
                        chosen_aud_id = audiences[aud_index]['id']
                        
                        gl_choice = input("Enter growth lever number for this audience: ")
                        prod_choice = input("Enter product number for this combination: ")

                        try:
                            gl_index = int(gl_choice) - 1
                            prod_index = int(prod_choice) - 1
                            if 0 <= gl_index < len(growth_levers) and 0 <= prod_index < len(products):
                                chosen_gl_id = growth_levers[gl_index]['id']
                                chosen_prod_id = products[prod_index]['id']
                                selected_combinations.append({
                                    'audience_id': chosen_aud_id,
                                    'growth_lever_id': chosen_gl_id,
                                    'product_id': chosen_prod_id
                                })
                                print(f"Selected: Audience '{audiences[aud_index]['title']}', Growth Lever '{growth_levers[gl_index]['type']}', Product '{products[prod_index]['name']}'")
                            else:
                                print("Invalid growth lever or product number. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter numbers for growth lever and product.")
                    else:
                        print("Invalid audience number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number for audience or 'done'.")
            
            if not selected_combinations:
                print("No valid combinations selected. Returning.")
                continue

            user_prompt = input("Enter an optional prompt for campaign generation (e.g., 'Make it humorous'): ")
            await process_campaign_ideas(user_prompt=user_prompt, selected_combinations=selected_combinations, action_type="generate")
        
        elif action == 'u':
            current_campaigns = await fetch_from_supabase('campaigns')
            if not current_campaigns:
                print("No campaigns found to update.")
                continue
            print("\nCurrent Campaigns:")
            for camp in current_campaigns:
                print(f"   ID: {camp.get('id')}, Name: {camp.get('name')}")
            
            camp_id = input("Enter the ID of the campaign to update: ")
            update_prompt = input(f"Enter the new details/prompt for campaign ID {camp_id}: ")
            await process_campaign_ideas(user_prompt=update_prompt, campaign_id_to_affect=camp_id, action_type="update_singular")
        
        elif action == 'd':
            current_campaigns = await fetch_from_supabase('campaigns')
            if not current_campaigns:
                print("No campaigns found to delete.")
                continue
            print("\nCurrent Campaigns:")
            for camp in current_campaigns:
                print(f"   ID: {camp.get('id')}, Name: {camp.get('name')}")
            
            camp_id = input("Enter the ID of the campaign to delete: ")
            confirm = input(f"Are you sure you want to delete campaign with ID {camp_id}? (yes/no): ").lower()
            if confirm == 'yes':
                await process_campaign_ideas(campaign_id_to_affect=camp_id, action_type="delete_singular")
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid action. Please choose from the available options.")

# This block allows you to run the test function directly when the script is executed.
if __name__ == "__main__":
    if not API_KEY:
        print("\nERROR: OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        exit(1) # Exit if API key is missing

    asyncio.run(test_campaign_generator())