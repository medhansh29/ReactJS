import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import os
import json

# Import the refactored functions from your modules
# Ensure these modules are in the same directory as api.py
from audience_analyser import orchestrate_audience_actions as process_audiences # Renamed for clarity
from growth_levers import orchestrate_growth_lever_actions as process_growth_levers # Renamed for clarity
from campaign_generator import orchestrate_campaign_actions as process_campaign_ideas # Renamed for clarity
# from flow_generator import process_journeys # This module was not provided, assuming it's external or will be added later if needed for journeys.

app = FastAPI(
    title="Campaign Strategy API",
    description="API for generating and managing campaign strategies, audiences, growth levers, ideas, and journeys.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your frontend (e.g., React app running on localhost:3000) to make requests to this API.
# For production, replace "*" with specific frontend URLs (e.g., ["https://reactjs-a4hv.onrender.com", "http://localhost:3000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Body Models for API Endpoints ---

# Audience models
class AudienceGenerateRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt for generating new audience types.")
    current_audiences: List[Dict[str, Any]] = Field(..., description="Current list of audiences in UI session memory.")

class AudienceModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Change the rationale for Eco-conscious Urban Professionals').")
    audience_id: str = Field(..., description="ID of the audience to modify.")
    current_audiences: List[Dict[str, Any]] = Field(..., description="Current list of audiences in UI session memory.")

class AudienceDeleteRequest(BaseModel):
    audience_id: str = Field(..., description="ID of the audience to delete.")
    current_audiences: List[Dict[str, Any]] = Field(..., description="Current list of audiences in UI session memory.")

class AudienceFinalizeRequest(BaseModel):
    user_id: str = Field(..., description="User ID performing the finalization.")
    current_audiences: List[Dict[str, Any]] = Field(..., description="Final list of audiences to save.")
    action_finalize: str = Field("overwrite", description="Action for finalization: 'overwrite' (upsert based on ID).")


# Growth Lever models
class GrowthLeverGenerateRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt for generating new growth levers.")
    current_growth_levers: List[Dict[str, Any]] = Field(..., description="Current list of growth levers in UI session memory.")

class GrowthLeverModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Update details for Discount Campaign').")
    growth_lever_id: str = Field(..., description="ID of the growth lever to modify.")
    current_growth_levers: List[Dict[str, Any]] = Field(..., description="Current list of growth levers in UI session memory.")

class GrowthLeverDeleteRequest(BaseModel):
    growth_lever_id: str = Field(..., description="ID of the growth lever to delete.")
    current_growth_levers: List[Dict[str, Any]] = Field(..., description="Current list of growth levers in UI session memory.")

class GrowthLeverFinalizeRequest(BaseModel):
    user_id: str = Field(..., description="User ID performing the finalization.")
    current_growth_levers: List[Dict[str, Any]] = Field(..., description="Final list of growth levers to save.")
    action_finalize: str = Field("overwrite", description="Action for finalization: 'overwrite' (upsert based on ID).")


# Campaign Idea models (renamed from Campaign Idea to Campaign Flow as per campaign_generator.py)
class CampaignFlowCombinationData(BaseModel):
    audience_data: Dict[str, Any] = Field(..., description="Full data of the audience to combine with.")
    growth_lever_data: Dict[str, Any] = Field(..., description="Full data of the growth lever to combine with.")
    product_data: Dict[str, Any] = Field(..., description="Full data of the product to combine with.")

class CampaignFlowGenerateRequest(BaseModel):
    user_prompt: str = ""
    current_campaign_flows: List[Dict[str, Any]] = Field(..., description="Current list of campaign flows in UI session memory.")

class CampaignFlowModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Make campaign XYZ more aggressive').")
    flow_id: str = Field(..., description="ID of the campaign flow to modify.")
    current_campaign_flows: List[Dict[str, Any]] = Field(..., description="Current list of campaign flows in UI session memory.")

class CampaignFlowDeleteRequest(BaseModel):
    flow_id: str = Field(..., description="ID of the campaign flow to delete.")
    current_campaign_flows: List[Dict[str, Any]] = Field(..., description="Current list of campaign flows in UI session memory.")

class CampaignFlowFinalizeRequest(BaseModel):
    user_id: str = Field(..., description="User ID performing the finalization.")
    current_campaign_flows: List[Dict[str, Any]] = Field(..., description="Final list of campaign flows to save.")
    action_finalize: str = Field("overwrite", description="Action for finalization: 'overwrite' (upsert based on ID).")


# Journey models (Keeping existing as there's no provided flow_generator.py for full orchestration)
class JourneyGenerateRequest(BaseModel):
    selected_campaign_data: Dict[str, Any] = Field(..., description="Full data of the campaign to generate journeys for.")
    user_prompt: str = ""

class JourneyModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Add a step to journey ABC').")
    journey_id: str = Field(..., description="ID of the journey to modify.")
    selected_campaign_data: Dict[str, Any] = Field(..., description="Full data of the campaign linked to this journey.")

class JourneyDeleteRequest(BaseModel):
    journey_id: str = Field(..., description="ID of the journey to delete.")


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Campaign Strategy API is running!"}

# Audience Endpoints
@app.post("/audiences/generate")
async def generate_audiences(request: AudienceGenerateRequest):
    """
    Generates new audience types based on a user prompt.
    Returns the updated list of audience records for UI session memory.
    """
    try:
        print(f"DEBUG: Calling process_audiences with action_type='generate'")
        updated_audiences = await process_audiences(
            user_prompt=request.user_prompt,
            current_audiences=request.current_audiences,
            action_type="generate"
        )
        print(f"DEBUG: Type returned by process_audiences: {type(updated_audiences)}")

        return {"message": "Audiences generated successfully", "data": updated_audiences}
    except Exception as e:
        print(f"ERROR in /audiences/generate: {e}") # Add this to see the full exception
        raise HTTPException(status_code=500, detail=f"Failed to generate audiences: {e}")

@app.put("/audiences/modify")
async def modify_audience(request: AudienceModifyRequest):
    """
    Modifies a specific audience type by ID in UI session memory.
    Returns the updated list of audience records for UI session memory.
    """
    try:
        updated_audiences = await process_audiences(
            user_prompt=request.user_prompt,
            current_audiences=request.current_audiences,
            action_type="update_singular",
            audience_id_to_affect=request.audience_id
        )
        return {"message": f"Audience {request.audience_id} modified successfully", "data": updated_audiences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify audience: {e}")

@app.delete("/audiences/delete")
async def delete_audience(request: AudienceDeleteRequest):
    """
    Deletes a specific audience type by ID from UI session memory.
    Returns the updated list of audience records for UI session memory.
    """
    try:
        updated_audiences = await process_audiences(
            user_prompt="", # Not used for delete action type
            current_audiences=request.current_audiences,
            action_type="delete_singular",
            audience_id_to_affect=request.audience_id
        )
        return {"message": f"Audience {request.audience_id} deleted successfully", "data": updated_audiences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete audience: {e}")

@app.post("/audiences/finalize")
async def finalize_audiences(request: AudienceFinalizeRequest):
    """
    Finalizes and saves the current list of audiences to the Supabase database.
    """
    try:
        # The process_audiences function will handle the saving to Supabase
        # and return the final state of audiences.
        final_audiences_state = await process_audiences(
            user_prompt="", # Not used for finalize
            current_audiences=request.current_audiences,
            action_type="finalize",
            updated_by_user_id=request.user_id,
            action_finalize=request.action_finalize
        )
        return {"message": "Audiences finalized and saved successfully", "data": final_audiences_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize audiences: {e}")


# Growth Lever Endpoints
@app.post("/growth-levers/generate")
async def generate_growth_levers(request: GrowthLeverGenerateRequest):
    """
    Generates new growth levers based on a user prompt for UI session memory.
    Returns the updated list of growth lever records for UI session memory.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt=request.user_prompt,
            current_growth_levers=request.current_growth_levers,
            action_type="generate"
        )
        return {"message": "Growth levers generated successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate growth levers: {e}")

@app.put("/growth-levers/modify")
async def modify_growth_lever(request: GrowthLeverModifyRequest):
    """
    Modifies a specific growth lever by ID in UI session memory.
    Returns the updated list of growth lever records for UI session memory.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt=request.user_prompt,
            current_growth_levers=request.current_growth_levers,
            action_type="update_singular",
            growth_lever_id_to_affect=request.growth_lever_id
        )
        return {"message": f"Growth lever {request.growth_lever_id} modified successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify growth lever: {e}")

@app.delete("/growth-levers/delete")
async def delete_growth_lever(request: GrowthLeverDeleteRequest):
    """
    Deletes a specific growth lever by ID from UI session memory.
    Returns the updated list of growth lever records for UI session memory.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt="", # Not used for delete action type
            current_growth_levers=request.current_growth_levers,
            action_type="delete_singular",
            growth_lever_id_to_affect=request.growth_lever_id
        )
        return {"message": f"Growth lever {request.growth_lever_id} deleted successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete growth lever: {e}")

@app.post("/growth-levers/finalize")
async def finalize_growth_levers(request: GrowthLeverFinalizeRequest):
    """
    Finalizes and saves the current list of growth levers to the Supabase database.
    """
    try:
        final_levers_state = await process_growth_levers(
            user_prompt="", # Not used for finalize
            current_growth_levers=request.current_growth_levers,
            action_type="finalize",
            updated_by_user_id=request.user_id,
            action_finalize=request.action_finalize
        )
        return {"message": "Growth levers finalized and saved successfully", "data": final_levers_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize growth levers: {e}")


# Campaign Flow Endpoints (Updated from Campaign Idea to Campaign Flow)
@app.post("/campaign-flows/generate")
async def generate_campaign_flows(request: CampaignFlowGenerateRequest):
    """
    Generates new campaign flows based on selected audience, growth lever, and product combinations for UI session memory.
    Returns the updated list of campaign flow records for UI session memory.
    """
    try:
        updated_campaign_flows = await process_campaign_ideas( # Using the renamed function
            user_prompt=request.user_prompt,
            current_campaign_flows=request.current_campaign_flows,
            action_type="generate"
            # selected_combinations_data is handled internally by process_campaign_ideas by fetching from Supabase
            # and combining audiences and growth levers from Supabase directly in 'generate' action.
        )
        return {"message": "Campaign flows generated successfully", "data": updated_campaign_flows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate campaign flows: {e}")

@app.put("/campaign-flows/modify")
async def modify_campaign_flow(request: CampaignFlowModifyRequest):
    """
    Modifies a specific campaign flow by ID in UI session memory.
    Returns the updated list of campaign flow records for UI session memory.
    """
    try:
        updated_campaign_flows = await process_campaign_ideas( # Using the renamed function
            user_prompt=request.user_prompt,
            current_campaign_flows=request.current_campaign_flows,
            action_type="update_singular",
            flow_id_to_affect=request.flow_id # Changed from campaign_id to flow_id
        )
        return {"message": f"Campaign flow {request.flow_id} modified successfully", "data": updated_campaign_flows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify campaign flow: {e}")

@app.delete("/campaign-flows/delete")
async def delete_campaign_flow(request: CampaignFlowDeleteRequest):
    """
    Deletes a specific campaign flow by ID from UI session memory.
    Returns the updated list of campaign flow records for UI session memory.
    """
    try:
        updated_campaign_flows = await process_campaign_ideas( # Using the renamed function
            user_prompt="", # Not used for delete action type
            current_campaign_flows=request.current_campaign_flows,
            action_type="delete_singular",
            flow_id_to_affect=request.flow_id # Changed from campaign_id to flow_id
        )
        return {"message": f"Campaign flow {request.flow_id} deleted successfully", "data": updated_campaign_flows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete campaign flow: {e}")

@app.post("/campaign-flows/finalize")
async def finalize_campaign_flows(request: CampaignFlowFinalizeRequest):
    """
    Finalizes and saves the current list of campaign flows to the Supabase database.
    """
    try:
        final_flows_state = await process_campaign_ideas( # Using the renamed function
            user_prompt="", # Not used for finalize
            current_campaign_flows=request.current_campaign_flows,
            action_type="finalize",
            updated_by_user_id=request.user_id,
            action_finalize=request.action_finalize
        )
        return {"message": "Campaign flows finalized and saved successfully", "data": final_flows_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize campaign flows: {e}")


# Journey Endpoints (assuming process_journeys still handles direct DB interaction)
@app.post("/journeys/generate")
async def generate_journeys(request: JourneyGenerateRequest):
    """
    Generates new marketing journeys for a given campaign and saves to Supabase.
    Returns the updated list of journey records from Supabase.
    """
    try:
        # Assuming process_journeys exists and handles direct DB interaction as before
        # Temporarily commenting out as process_journeys is not defined in this context
        # updated_journeys = await process_journeys(
        #     user_prompt=request.user_prompt,
        #     selected_campaign_data=request.selected_campaign_data,
        #     action_type="generate"
        # )
        # return {"message": "Journeys generated and saved successfully", "data": updated_journeys}
        raise HTTPException(status_code=501, detail="Journey generation is not implemented yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate journeys: {e}")

@app.put("/journeys/modify")
async def modify_journey(request: JourneyModifyRequest):
    """
    Modifies a specific marketing journey by ID and updates in Supabase.
    Returns the updated list of journey records from Supabase.
    """
    try:
        # Assuming process_journeys exists and handles direct DB interaction as before
        # Temporarily commenting out as process_journeys is not defined in this context
        # updated_journeys = await process_journeys(
        #     user_prompt=request.user_prompt,
        #     selected_campaign_data=request.selected_campaign_data,
        #     action_type="update_singular",
        #     journey_id_to_affect=request.journey_id
        # )
        # return {"message": f"Journey {request.journey_id} modified successfully", "data": updated_journeys}
        raise HTTPException(status_code=501, detail="Journey modification is not implemented yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify journey: {e}")

@app.delete("/journeys/delete/{journey_id}")
async def delete_journey(journey_id: str):
    """
    Deletes a specific marketing journey by ID from Supabase.
    Returns the updated list of journey records from Supabase.
    """
    try:
        # Assuming process_journeys exists and handles direct DB interaction as before
        # Temporarily commenting out as process_journeys is not defined in this context
        # updated_journeys = await process_journeys(
        #     user_prompt="", # Not used for delete action type
        #     selected_campaign_data={}, # Placeholder, as it's not strictly needed for delete journey from DB
        #     action_type="delete_singular",
        #     journey_id_to_affect=journey_id
        # )
        # return {"message": f"Journey {journey_id} deleted successfully", "data": updated_journeys}
        raise HTTPException(status_code=501, detail="Journey deletion is not implemented yet.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete journey: {e}")

# The if __name__ == "__main__": block is removed for production deployment with Gunicorn