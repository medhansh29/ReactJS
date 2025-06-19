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

async def fetch_from_supabase(table_name: str, id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches data from a specified Supabase table, optionally by ID."""
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

async def insert_into_journeys_store(data: List[Dict]) -> None:
    """Inserts data into the 'journeys' Supabase table."""
    try:
        response = supabase.from_('journeys').insert(data).execute()
        if response and hasattr(response, 'data'):
            print(f"Successfully inserted {len(response.data)} records into journeys table.")
        else:
            print(f"No data returned on insert to journeys, but request sent.")
    except Exception as e:
        print(f"Error inserting data into journeys table: {e}")

async def update_journeys_store(id: str, data: Dict) -> None:
    """Updates a record in the 'journeys' Supabase table."""
    try:
        response = supabase.from_('journeys').update(data).eq('id', id).execute()
        if response and hasattr(response, 'data') and response.data:
            print(f"Successfully updated record {id} in journeys table.")
        else:
            print(f"No data returned on update for id {id}, but request sent or no matching id found.")
    except Exception as e:
        print(f"Error updating record {id} in journeys table: {e}")

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
        print(f"Validation Error during LLM journey call: {e.errors()}")
        print(f"LLM probably did not return the expected JSON format. Raw error: {e}")
        return []
    except Exception as e:
        print(f"Error during LLM journey call: {e}")
        return []

async def process_journeys(
    user_prompt: str = "", # For initial generation prompt or modification prompt
    campaign_id_for_generation: Optional[str] = None, # For generation
    journey_id_to_affect: Optional[str] = None, # For singular update/delete
    action_type: str = "generate" # 'generate', 'update_singular', 'delete_singular'
) -> List[Dict[str, Any]]:
    """
    Generates, updates, or deletes marketing journeys and updates the 'journeys' Supabase table.
    Returns the updated list of journey records from Supabase.
    """
    
    current_journeys_from_db = await fetch_from_supabase('journeys')
    current_campaigns_from_db = await fetch_from_supabase('campaigns') # Need campaigns for context

    if action_type == "delete_singular":
        if journey_id_to_affect:
            await delete_from_supabase('journeys', journey_id_to_affect)
        return await fetch_from_supabase('journeys')

    elif action_type == "update_singular":
        if not journey_id_to_affect:
            print("No journey ID provided for singular update.")
            return await fetch_from_supabase('journeys')

        journey_to_update = next((j for j in current_journeys_from_db if j.get('id') == journey_id_to_affect), None)
        if not journey_to_update:
            print(f"Journey with ID {journey_id_to_affect} not found for update.")
            return await fetch_from_supabase('journeys')
        
        # Find the linked campaign for context
        linked_campaign_name = journey_to_update.get('campaign_name')
        linked_campaign = next((c for c in current_campaigns_from_db if c.get('name') == linked_campaign_name), None)

        if not linked_campaign:
            print(f"ERROR: Linked campaign '{linked_campaign_name}' not found for journey ID {journey_id_to_affect}.")
            return await fetch_from_supabase('journeys')

        llm_response_list = await call_llm_for_journeys(
            api_key=API_KEY or "",
            campaign_data=linked_campaign, # Provide the linked campaign for context
            user_prompt=user_prompt, # User's specific modification request
            current_journeys_for_llm=[journey_to_update] # Send the specific journey to modify
        )

        if llm_response_list:
            updated_journey_data = llm_response_list[0] # Expecting only one updated journey
            
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
            print(f"\n--- Updated Single Journey (ID: {journey_id_to_affect}) ---")
            print(f"Name: {data_to_update.get('name')}")
            print(f"Campaign: {data_to_update.get('campaign_name')}")
            print(f"Description: {data_to_update.get('description')}")
            print(f"UX Type: {data_to_update.get('ux_type')}")
            print(f"Channel: {data_to_update.get('channel')}")
            print(f"Cohort: {data_to_update.get('cohort')}")
            print(f"Outcome: {data_to_update.get('outcome_metric')}")
            print(f"Lever: {data_to_update.get('lever')}")
            print(f"Product: {data_to_update.get('product')}")
            print(f"Last Edited: {data_to_update.get('last_edited')}")
            print("-" * 30)
            return await fetch_from_supabase('journeys')
        else:
            print(f"Could not generate update for journey ID {journey_id_to_affect}.")
        return await fetch_from_supabase('journeys')

    elif action_type == "generate":
        if not campaign_id_for_generation:
            print("No campaign ID provided for journey generation.")
            return await fetch_from_supabase('journeys')
        
        selected_campaign = next((c for c in current_campaigns_from_db if c.get('id') == campaign_id_for_generation), None)
        if not selected_campaign:
            print(f"Campaign with ID {campaign_id_for_generation} not found for journey generation.")
            return await fetch_from_supabase('journeys')

        generated_journeys_to_insert = []
        generated_ideas = await call_llm_for_journeys(
            api_key=API_KEY or "",
            campaign_data=selected_campaign,
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
                journey_item['campaign_name'] = selected_campaign.get('name')
                journey_item['cohort'] = selected_campaign.get('target_cohort')
                journey_item['lever'] = selected_campaign.get('lever_config')
                journey_item['product'] = selected_campaign.get('product')

                generated_journeys_to_insert.append(journey_item)
        
        if generated_journeys_to_insert:
            await insert_into_journeys_store(generated_journeys_to_insert)

        updated_journeys = await fetch_from_supabase('journeys')
        
        print("\n--- Generated Journeys ---")
        if updated_journeys:
            for journey in updated_journeys:
                print(f"ID: {journey.get('id')}")
                print(f"Campaign Name: {journey.get('campaign_name')}")
                print(f"Journey Name: {journey.get('name')}")
                print(f"Description: {journey.get('description')}")
                print(f"UX Type: {journey.get('ux_type')}")
                print(f"Channel: {journey.get('channel')}")
                print(f"Cohort: {journey.get('cohort')}")
                print(f"Outcome Metric: {journey.get('outcome_metric')}")
                print(f"Lever: {journey.get('lever')}")
                print(f"Product: {journey.get('product')}")
                print(f"Last Edited: {journey.get('last_edited')}")
                print("-" * 30)
        else:
            print("No journeys generated or found.")
            
        return updated_journeys
    
    return await fetch_from_supabase('journeys')


# --- Test Function ---
async def test_journey_generator():
    print("Welcome to the Journey Generator Test!")
    while True:
        action = input("\nChoose action: (g)enerate new, (u)pdate singular, (d)elete singular, or (q)uit: ").lower()
        
        if action == 'q':
            break
        elif action == 'g':
            campaigns = await fetch_from_supabase('campaigns')
            if not campaigns:
                print("No campaigns found. Please generate campaigns first.")
                continue

            print("\nAvailable Campaigns:")
            for i, camp in enumerate(campaigns):
                print(f"   {i+1}. ID: {camp.get('id')}, Name: {camp.get('name')}, Cohort: {camp.get('target_cohort')}")

            campaign_choice = input("Enter campaign number to generate journeys for: ")
            try:
                camp_index = int(campaign_choice) - 1
                if 0 <= camp_index < len(campaigns):
                    chosen_campaign_id = campaigns[camp_index]['id']
                    user_prompt = input("Enter an optional prompt for journey generation (e.g., 'Make it a 3-step email flow'): ")
                    await process_journeys(user_prompt=user_prompt, campaign_id_for_generation=chosen_campaign_id, action_type="generate")
                else:
                    print("Invalid campaign number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number for the campaign.")
        
        elif action == 'u':
            current_journeys = await fetch_from_supabase('journeys')
            if not current_journeys:
                print("No journeys found to update.")
                continue
            print("\nCurrent Journeys:")
            for jrny in current_journeys:
                print(f"   ID: {jrny.get('id')}, Name: {jrny.get('name')}, Campaign: {jrny.get('campaign_name')}")
            
            jrny_id = input("Enter the ID of the journey to update: ")
            update_prompt = input(f"Enter the new details/prompt for journey ID {jrny_id}: ")
            await process_journeys(user_prompt=update_prompt, journey_id_to_affect=jrny_id, action_type="update_singular")
        
        elif action == 'd':
            current_journeys = await fetch_from_supabase('journeys')
            if not current_journeys:
                print("No journeys found to delete.")
                continue
            print("\nCurrent Journeys:")
            for jrny in current_journeys:
                print(f"   ID: {jrny.get('id')}, Name: {jrny.get('name')}, Campaign: {jrny.get('campaign_name')}")
            
            jrny_id = input("Enter the ID of the journey to delete: ")
            confirm = input(f"Are you sure you want to delete journey with ID {jrny_id}? (yes/no): ").lower()
            if confirm == 'yes':
                await process_journeys(journey_id_to_affect=jrny_id, action_type="delete_singular")
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid action. Please choose from the available options.")

# This block allows you to run the test function directly when the script is executed.
if __name__ == "__main__":
    if not API_KEY:
        print("\nERROR: OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        exit(1) # Exit if API key is missing

    asyncio.run(test_journey_generator())
