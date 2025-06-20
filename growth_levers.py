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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Pydantic Models for Structured Output ---

class GrowthLeverExplanation(BaseModel):
    """Detailed explanation of the growth lever's strategic context."""
    audience_behavior_signals: List[str] = Field(description="Behavioral signals of the audience that justify this lever.")
    key_metrics_impacted: List[str] = Field(description="List of key metrics this lever is expected to impact (e.g., 'Conversion Rate', 'Average Order Value').")
    implementation_channels: List[str] = Field(description="Primary channels through which this lever would be implemented (e.g., 'Email', 'In-App', 'Social Media Ads').")

class GrowthLever(BaseModel):
    """Represents a suggested growth lever."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), alias="growth_lever_id", description="Unique ID for the growth lever.")
    type: str = Field(description="A concise name for the growth lever (e.g., 'Personalized Discount Campaign').")
    details: str = Field(description="Specific, actionable details of the growth lever.")
    rationale: str = Field(description="Strategic reasoning and justification for this growth lever.")
    growth_lever_explanation: GrowthLeverExplanation = Field(description="Detailed explanation of the growth lever's strategic context.")
    estimated_lift: float = Field(description="Estimated percentage lift in a key metric (e.g., 0.15 for 15% lift).")
    estimated_roi: Optional[float] = Field(None, description="Estimated Return on Investment for implementing this lever.")
    cohort_score_impact: Optional[float] = Field(None, description="Numerical impact on the target cohort's score (e.g., 0.1 for a 10% increase).")

class SuggestedGrowthLevers(BaseModel):
    """Represents the output structure for suggested growth levers."""
    suggested_growth_levers: List[GrowthLever] = Field(description="A list of generated growth levers.")

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=SecretStr(API_KEY))
parser = PydanticOutputParser(pydantic_object=SuggestedGrowthLevers)

growth_lever_template = """
You are an expert marketing strategist specializing in identifying and defining actionable growth levers.
Your goal is to propose strategic initiatives that can drive business growth, considering the current context and user's specific requests.

Current Date and Time: {current_datetime}

Here is the user's request: "{user_prompt}"

---
Existing Growth Levers in Session (for context and modification if applicable):
{current_growth_levers_json}
---

Based on the user's prompt, perform the following action:
{action_instructions}

When generating or updating, ensure each growth lever has:
- 'growth_lever_id': A unique identifier (if generating, create a new UUID; if updating, use the existing one).
- 'type': A concise name (e.g., "Retention via Loyalty Program").
- 'details': Specific implementation details.
- 'rationale': Strategic justification.
- 'growth_lever_explanation': An object with:
    - 'audience_behavior_signals': Why this lever fits the audience's behavior.
    - 'key_metrics_impacted': Metrics this lever will influence.
    - 'implementation_channels': Where this lever will be deployed.
- 'estimated_lift': A numerical value (e.g., 0.05 for 5% lift).
- 'estimated_roi': A numerical value (invent if generating).
- 'cohort_score_impact': A numerical value (invent if generating).

{format_instructions}
"""

growth_lever_prompt = ChatPromptTemplate.from_template(template=growth_lever_template)

# --- Main Orchestration Function ---
async def orchestrate_growth_lever_actions(
    user_prompt: str,
    current_growth_levers: List[Dict[str, Any]],
    action_type: str,
    growth_lever_id_to_affect: Optional[str] = None,
    updated_by_user_id: Optional[str] = None,
    action_finalize: Optional[str] = None # Added for finalize action
) -> List[Dict[str, Any]]:
    """
    Orchestrates actions related to growth levers: generate (suggest), update, delete, or finalize (save to DB).

    Args:
        user_prompt (str): The prompt from the user.
        current_growth_levers (List[Dict[str, Any]]): The current list of growth levers from UI session memory.
        action_type (str): The type of action to perform ('generate' -> 'suggest', 'update_singular', 'delete_singular', 'finalize').
        growth_lever_id_to_affect (Optional[str]): The ID of the growth lever to affect for update/delete.
        updated_by_user_id (Optional[str]): User ID for tracking who finalized the data.
        action_finalize (Optional[str]): Specific action for finalize ('overwrite').

    Returns:
        List[Dict[str, Any]]: The updated list of growth lever records for UI session memory.
    """
    updated_levers_list = [GrowthLever(**gl).model_dump(by_alias=True) if not isinstance(gl, GrowthLever) else gl for gl in current_growth_levers]
    current_datetime = datetime.now().isoformat()
    action_instructions = ""

    if action_type == "generate": # This will now act as 'suggest'
        action_instructions = """
        Generate new, distinct growth levers based on the user's prompt. Do not regenerate existing ones unless explicitly asked to modify them.
        Propose at least 3-5 new, diverse growth levers.
        """
        chain = growth_lever_prompt | llm | parser
        response = await chain.ainvoke({
            "user_prompt": user_prompt,
            "current_growth_levers_json": json.dumps(current_growth_levers, indent=2),
            "action_instructions": action_instructions,
            "format_instructions": parser.get_format_instructions(),
            "current_datetime": current_datetime
        })
        new_levers = [gl.model_dump(by_alias=True) for gl in response.suggested_growth_levers]
        existing_ids = {gl.get("growth_lever_id") for gl in updated_levers_list if gl.get("growth_lever_id")}
        for new_gl in new_levers:
            if new_gl.get("growth_lever_id") and new_gl["growth_lever_id"] not in existing_ids:
                updated_levers_list.append(new_gl)

    elif action_type == "update_singular":
        action_instructions = f"""
        Modify the growth lever with ID '{growth_lever_id_to_affect}' based on the user's prompt.
        Only return the modified growth lever object. Ensure its ID remains '{growth_lever_id_to_affect}'.
        """
        lever_to_modify = next((gl for gl in updated_levers_list if gl.get("growth_lever_id") == growth_lever_id_to_affect), None)
        if not lever_to_modify:
            raise HTTPException(status_code=404, detail=f"Growth lever with ID {growth_lever_id_to_affect} not found.")

        chain = growth_lever_prompt | llm | parser
        response = await chain.ainvoke({
            "user_prompt": user_prompt,
            "current_growth_levers_json": json.dumps([lever_to_modify], indent=2),
            "action_instructions": action_instructions,
            "format_instructions": parser.get_format_instructions(),
            "current_datetime": current_datetime
        })

        if response.suggested_growth_levers:
            modified_lever = response.suggested_growth_levers[0].model_dump(by_alias=True)
            updated_levers_list = [
                modified_lever if gl.get("growth_lever_id") == growth_lever_id_to_affect else gl
                for gl in updated_levers_list
            ]
        else:
            print(f"LLM did not return a modified growth lever for ID {growth_lever_id_to_affect}. Original list returned.")


    elif action_type == "delete_singular":
        if growth_lever_id_to_affect:
            updated_levers_list = [
                gl for gl in updated_levers_list if gl.get("growth_lever_id") != growth_lever_id_to_affect
            ]
        else:
            raise ValueError("Growth Lever ID is required for delete_singular action.")

    elif action_type == "finalize":
        if not updated_by_user_id:
            raise ValueError("User ID is required for finalize action.")

        try:
            data_to_save = []
            for gl in updated_levers_list:
                data_to_save.append({
                    "growth_lever_id": gl.get("growth_lever_id"),
                    "type": gl.get("type"),
                    "details": gl.get("details"),
                    "rationale": gl.get("rationale"),
                    "growth_lever_explanation": json.dumps(gl.get("growth_lever_explanation", {})),
                    "estimated_lift": gl.get("estimated_lift"),
                    "estimated_roi": gl.get("estimated_roi"),
                    "cohort_score_impact": gl.get("cohort_score_impact"),
                    "user_id": updated_by_user_id,
                    "timestamp": datetime.now().isoformat(),
                })

            response = supabase.table("growth_levers_store").upsert(data_to_save, on_conflict="growth_lever_id").execute()

            if response.data:
                print(f"Successfully saved {len(response.data)} growth levers to Supabase 'growth_levers_store'.")
            else:
                print("No data returned after upsert, check Supabase operation.")

        except Exception as e:
            print(f"An unexpected error occurred during finalize action: {e}")
            raise # Re-raise to ensure error is propagated if necessary

    # Always return the current state of growth levers for the UI session memory
    # This step ensures data is consistently formatted before returning to the UI
    final_formatted_levers = []
    for gl in updated_levers_list:
        # Ensure 'id' is present for UI consistency, using 'growth_lever_id' as source
        formatted_gl = {
            "id": gl.get("growth_lever_id"),
            "growth_lever_id": gl.get("growth_lever_id"), # Keep both for now to avoid breaking existing UI
            "type": gl.get("type"),
            "details": gl.get("details"),
            "rationale": gl.get("rationale"),
            "growth_lever_explanation": gl.get("growth_lever_explanation"),
            "estimated_lift": gl.get("estimated_lift"),
            "estimated_roi": gl.get("estimated_roi"),
            "cohort_score_impact": gl.get("cohort_score_impact"),
        }
        final_formatted_levers.append(formatted_gl)

    return final_formatted_levers