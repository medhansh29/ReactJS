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
from pydantic import BaseModel, Field, SecretStr
from supabase import create_client, Client
from fastapi import HTTPException # Import HTTPException for proper error handling

# --- Configuration ---
API_KEY = os.getenv('OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_CLIENT_ANON_KEY')

# Add a check to ensure they are loaded
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    # Use a more user-friendly error or log for production
    raise ValueError("SUPABASE_URL and SUPABASE_CLIENT_ANON_KEY environment variables must be set.")
if not API_KEY:
    # Use a more user-friendly error or log for production
    raise ValueError("OPENAI_API_KEY environment variables must be set.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class AudienceType(BaseModel):
    """Represents a suggested audience type with its rationale and rule definition."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the audience type.")
    type: str = Field(description="A descriptive and concise name for the audience type (e.g., 'Eco-conscious Urban Professionals').")
    rationale: str = Field(description="The strategic reasoning and justification for targeting this audience.")
    rule: Dict[str, Any] = Field(description="A JSON object defining the segmentation rules for this audience (e.g., {'location': 'urban', 'interests': ['eco-friendly', 'sustainability']}).")
    audience_size: Optional[int] = Field(None, description="Estimated size of the audience cohort (e.g., 500000).")
    cohort_score: Optional[float] = Field(None, description="A numerical score indicating the potential value or strategic importance of this cohort (e.g., 0.75).")
    cohort_rationale: Optional[str] = Field(None, description="Explanation for the assigned cohort score.")

class SuggestedAudiencesOutput(BaseModel):
    """Represents the output structure for suggested audience types."""
    suggested_audiences: List[AudienceType] = Field(description="A list of generated audience types.")

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=SecretStr(API_KEY))
parser = PydanticOutputParser(pydantic_object=SuggestedAudiencesOutput)

audience_template = """
You are an expert marketing strategist. Your task is to generate and refine audience types based on user prompts and existing audience data.

Current Date and Time: {current_datetime}

Here is the user's request: "{user_prompt}"

---
Existing Audiences in Session (for context and modification if applicable):
{current_audiences_json}
---

Based on the user's prompt, perform the following action:
{action_instructions}

When generating or updating, ensure each audience type has:
- 'id': A unique identifier (if generating, create a new UUID; if updating, use the existing one).
- 'type': A concise name (e.g., "Tech-Savvy Millennials").
- 'rationale': Strategic justification.
- 'rule': A JSON object for segmentation (e.g., {{"age_range": "25-40", "interests": ["technology", "gaming"]}}).
- 'audience_size': An estimated numerical size (invent if generating).
- 'cohort_score': A numerical score between 0.0 and 1.0 (invent if generating).
- 'cohort_rationale': Explanation for the cohort score.

{format_instructions}
"""

audience_prompt = ChatPromptTemplate.from_template(template=audience_template)

# --- Main Orchestration Function ---
async def orchestrate_audience_actions(
    user_prompt: str,
    current_audiences: List[Dict[str, Any]],
    action_type: str,
    audience_id_to_affect: Optional[str] = None,
    updated_by_user_id: Optional[str] = None,
    action_finalize: Optional[str] = None # Added for finalize action
) -> List[Dict[str, Any]]: # Explicitly declare return type
    """
    Orchestrates actions related to audience types: generate (suggest), update, delete, or finalize (save to DB).

    Args:
        user_prompt (str): The prompt from the user.
        current_audiences (List[Dict[str, Any]]): The current list of audience records from UI session memory.
        action_type (str): The type of action to perform ('generate' -> 'suggest', 'update_singular', 'delete_singular', 'finalize').
        audience_id_to_affect (Optional[str]): The ID of the audience to affect for update/delete.
        updated_by_user_id (Optional[str]): User ID for tracking who finalized the data.
        action_finalize (Optional[str]): Specific action for finalize ('overwrite').

    Returns:
        List[Dict[str, Any]]: The updated list of audience records for UI session memory.
    """
    # Ensure all incoming audience dictionaries conform to AudienceType schema for consistent processing
    # Convert to AudienceType Pydantic objects first, then to dict for mutable list manipulation
    updated_audiences_list: List[Dict[str, Any]] = []
    for aud_data in current_audiences:
        try:
            # Handle potential case where rule is stringified JSON
            if 'rule' in aud_data and isinstance(aud_data['rule'], str):
                aud_data['rule'] = json.loads(aud_data['rule'])
            updated_audiences_list.append(AudienceType(**aud_data).model_dump())
        except Exception as e:
            print(f"Warning: Could not parse existing audience data into AudienceType: {aud_data}. Error: {e}")
            updated_audiences_list.append(aud_data) # Keep original if parsing fails

    current_datetime = datetime.now().isoformat()
    action_instructions = ""

    if action_type == "generate": # This will now act as 'suggest'
        action_instructions = """
        Generate new, distinct audience types based on the user's prompt. Do not regenerate existing ones unless explicitly asked to modify them.
        Provide at least 3-5 new, diverse audience types.
        """
        try:
            chain = audience_prompt | llm | parser
            response: SuggestedAudiencesOutput = await chain.invoke({
                "user_prompt": user_prompt,
                "current_audiences_json": json.dumps(current_audiences, indent=2),
                "action_instructions": action_instructions,
                "format_instructions": parser.get_format_instructions(),
                "current_datetime": current_datetime
            })
            new_audiences = [aud.model_dump() for aud in response.suggested_audiences]
            # Append new audiences, ensuring no duplicates if IDs are generated
            existing_ids = {aud.get("id") for aud in updated_audiences_list if aud.get("id")}
            for new_aud in new_audiences:
                if new_aud.get("id") and new_aud["id"] not in existing_ids:
                    updated_audiences_list.append(new_aud)
        except Exception as e:
            # Log the full exception for debugging
            print(f"Error during LLM generation for audiences: {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed for audiences: {e}")


    elif action_type == "update_singular":
        action_instructions = f"""
        Modify the audience with ID '{audience_id_to_affect}' based on the user's prompt.
        Only return the modified audience object. Ensure its ID remains '{audience_id_to_affect}'.
        """
        # Find the specific audience to modify
        audience_to_modify = next((aud for aud in updated_audiences_list if aud.get("id") == audience_id_to_affect), None)
        if not audience_to_modify:
            raise HTTPException(status_code=404, detail=f"Audience with ID {audience_id_to_affect} not found.")

        try:
            # Temporarily use only the audience to be modified for focused LLM output
            chain = audience_prompt | llm | parser
            response: SuggestedAudiencesOutput = await chain.invoke({
                "user_prompt": user_prompt,
                "current_audiences_json": json.dumps([audience_to_modify], indent=2), # Pass only the relevant audience
                "action_instructions": action_instructions,
                "format_instructions": parser.get_format_instructions(),
                "current_datetime": current_datetime
            })

            if response.suggested_audiences:
                modified_audience = response.suggested_audiences[0].model_dump()
                # Replace the old audience with the modified one
                updated_audiences_list = [
                    modified_audience if aud.get("id") == audience_id_to_affect else aud
                    for aud in updated_audiences_list
                ]
            else:
                print(f"LLM did not return a modified audience for ID {audience_id_to_affect}. Original list returned.")
        except Exception as e:
            print(f"Error during LLM modification for audience {audience_id_to_affect}: {e}")
            raise HTTPException(status_code=500, detail=f"LLM modification failed for audience: {e}")


    elif action_type == "delete_singular":
        if audience_id_to_affect:
            updated_audiences_list = [
                aud for aud in updated_audiences_list if aud.get("id") != audience_id_to_affect
            ]
        else:
            raise ValueError("Audience ID is required for delete_singular action.")

    elif action_type == "finalize":
        if not updated_by_user_id:
            raise ValueError("User ID is required for finalize action.")

        try:
            data_to_save = []
            for aud in updated_audiences_list:
                data_to_save.append({
                    "id": aud.get("id"),
                    "name": aud.get("type"), # Renamed from 'type' to 'name' for Supabase
                    "rationale": aud.get("rationale"),
                    "rule": json.dumps(aud.get("rule", {})), # Store JSON as string in Supabase
                    "estimated_size": aud.get("audience_size"),
                    "cohort_score": aud.get("cohort_score"),
                    "cohort_rationale": aud.get("cohort_rationale"),
                    "user_id": updated_by_user_id,
                    "timestamp": datetime.now().isoformat()
                })

            supabase_response = supabase.table("audience_store").upsert(data_to_save, on_conflict="id").execute()

            if supabase_response.data:
                print(f"Successfully saved {len(supabase_response.data)} audiences to Supabase.")
            else:
                print("No data returned after upsert, check Supabase operation.")

        except Exception as e:
            print(f"An unexpected error occurred during finalize action: {e}")
            # Depending on error handling strategy, you might want to re-raise or return an error state.
            raise HTTPException(status_code=500, detail=f"Failed to finalize audiences to Supabase: {e}")


    # Ensure the returned list is always consistent with the expected AudienceType schema for UI
    # This step reformats the data back to the UI's expected structure if any internal changes occurred.
    final_formatted_audiences = []
    for aud in updated_audiences_list:
        final_formatted_audiences.append({
            "id": aud.get("id"),
            "name": aud.get("type") if aud.get("type") else aud.get("name"), # Prefer 'type' from LLM, fall back to 'name' from DB
            "rule": aud.get("rule", {}),
            "estimated_size": aud.get("audience_size"),
            "estimated_conversion_rate": aud.get("estimated_conversion_rate", round(random.uniform(0.05, 0.25), 2)), # Retain if exists, else invent
            "rationale": aud.get("rationale"),
            "top_features": aud.get("top_features", []),
            "cohort_score": aud.get("cohort_score"),
            "cohort_rationale": aud.get("cohort_rationale"),
        })

    return final_formatted_audiences