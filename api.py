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
from audience_analyser import process_audiences
from growth_levers import process_growth_levers
from campaign_generator import process_campaign_ideas
# Renamed from flow_generator to journey_generator as per the latest changes
from flow_generator import process_journeys 

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

class AudienceModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Change the rationale for Eco-conscious Urban Professionals').")
    audience_id: str = Field(..., description="ID of the audience to modify.")

class AudienceDeleteRequest(BaseModel):
    audience_id: str = Field(..., description="ID of the audience to delete.")

# Growth Lever models
class GrowthLeverGenerateRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt for generating new growth levers.")

class GrowthLeverModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Update details for Discount Campaign').")
    growth_lever_id: str = Field(..., description="ID of the growth lever to modify.")

class GrowthLeverDeleteRequest(BaseModel):
    growth_lever_id: str = Field(..., description="ID of the growth lever to delete.")

# Campaign Idea models
class CampaignIdeaCombination(BaseModel):
    audience_id: str = Field(..., description="ID of the audience to combine with.")
    growth_lever_id: str = Field(..., description="ID of the growth lever to combine with.")
    product_id: str = Field(..., description="ID of the product to combine with.")

class CampaignIdeaGenerateRequest(BaseModel):
    selected_combinations: List[CampaignIdeaCombination] = Field(..., description="List of audience-growth lever-product combinations to generate campaigns for.")
    user_prompt: str = ""

class CampaignIdeaModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Make campaign XYZ more aggressive').")
    campaign_id: str = Field(..., description="ID of the campaign to modify.")

class CampaignIdeaDeleteRequest(BaseModel):
    campaign_id: str = Field(..., description="ID of the campaign to delete.")

# Journey models (formerly Flow)
class JourneyGenerateRequest(BaseModel):
    campaign_id: str = Field(..., description="ID of the campaign to generate journeys for.")
    user_prompt: str = ""

class JourneyModifyRequest(BaseModel):
    user_prompt: str = Field(..., description="Prompt describing the modification (e.g., 'Add a step to journey ABC').")
    journey_id: str = Field(..., description="ID of the journey to modify.")

class JourneyDeleteRequest(BaseModel):
    journey_id: str = Field(..., description="ID of the journey to delete.")


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Campaign Strategy API is running!"}

# Removed the /strategy endpoint as campaign_strategy.json is no longer the central storage

@app.post("/audiences/generate")
async def generate_audiences(request: AudienceGenerateRequest):
    """
    Generates new audience types based on a user prompt.
    Returns the updated list of audience records from Supabase.
    """
    try:
        updated_audiences = await process_audiences(user_prompt=request.user_prompt, is_modification=False)
        return {"message": "Audiences generated successfully", "data": updated_audiences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audiences: {e}")

@app.put("/audiences/modify")
async def modify_audience(request: AudienceModifyRequest):
    """
    Modifies a specific audience type by ID.
    Returns the updated list of audience records from Supabase.
    """
    try:
        updated_audiences = await process_audiences(
            user_prompt=request.user_prompt,
            audience_id_to_affect=request.audience_id
        )
        return {"message": f"Audience {request.audience_id} modified successfully", "data": updated_audiences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify audience: {e}")

@app.delete("/audiences/delete/{audience_id}")
async def delete_audience(audience_id: str):
    """
    Deletes a specific audience type by ID.
    Returns the updated list of audience records from Supabase.
    """
    try:
        updated_audiences = await process_audiences(
            user_prompt="delete", # Signal delete action to the function
            audience_id_to_affect=audience_id
        )
        return {"message": f"Audience {audience_id} deleted successfully", "data": updated_audiences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete audience: {e}")

@app.post("/growth-levers/generate")
async def generate_growth_levers(request: GrowthLeverGenerateRequest):
    """
    Generates new growth levers based on a user prompt.
    Returns the updated list of growth lever records from Supabase.
    """
    try:
        updated_levers = await process_growth_levers(user_prompt=request.user_prompt, is_modification=False)
        return {"message": "Growth levers generated successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate growth levers: {e}")

@app.put("/growth-levers/modify")
async def modify_growth_lever(request: GrowthLeverModifyRequest):
    """
    Modifies a specific growth lever by ID.
    Returns the updated list of growth lever records from Supabase.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt=request.user_prompt,
            growth_lever_id_to_affect=request.growth_lever_id
        )
        return {"message": f"Growth lever {request.growth_lever_id} modified successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify growth lever: {e}")

@app.delete("/growth-levers/delete/{growth_lever_id}")
async def delete_growth_lever(growth_lever_id: str):
    """
    Deletes a specific growth lever by ID.
    Returns the updated list of growth lever records from Supabase.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt="delete", # Signal delete action to the function
            growth_lever_id_to_affect=growth_lever_id
        )
        return {"message": f"Growth lever {growth_lever_id} deleted successfully", "data": updated_levers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete growth lever: {e}")

@app.post("/campaign-ideas/generate")
async def generate_campaign_ideas(request: CampaignIdeaGenerateRequest):
    """
    Generates new campaign ideas based on selected audience, growth lever, and product combinations.
    Returns the updated list of campaign records from Supabase.
    """
    try:
        updated_campaigns = await process_campaign_ideas(
            user_prompt=request.user_prompt,
            selected_combinations=[comb.dict() for comb in request.selected_combinations],
            action_type="generate"
        )
        return {"message": "Campaign ideas generated successfully", "data": updated_campaigns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate campaign ideas: {e}")

@app.put("/campaign-ideas/modify")
async def modify_campaign_idea(request: CampaignIdeaModifyRequest):
    """
    Modifies a specific campaign idea by ID.
    Returns the updated list of campaign records from Supabase.
    """
    try:
        updated_campaigns = await process_campaign_ideas(
            user_prompt=request.user_prompt,
            campaign_id_to_affect=request.campaign_id,
            action_type="update_singular"
        )
        return {"message": f"Campaign idea {request.campaign_id} modified successfully", "data": updated_campaigns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify campaign idea: {e}")

@app.delete("/campaign-ideas/delete/{campaign_id}")
async def delete_campaign_idea(campaign_id: str):
    """
    Deletes a specific campaign idea by ID.
    Returns the updated list of campaign records from Supabase.
    """
    try:
        updated_campaigns = await process_campaign_ideas(
            campaign_id_to_affect=campaign_id,
            action_type="delete_singular"
        )
        return {"message": f"Campaign idea {campaign_id} deleted successfully", "data": updated_campaigns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete campaign idea: {e}")


@app.post("/journeys/generate")
async def generate_journeys(request: JourneyGenerateRequest):
    """
    Generates new marketing journeys for a given campaign.
    Returns the updated list of journey records from Supabase.
    """
    try:
        updated_journeys = await process_journeys(
            user_prompt=request.user_prompt,
            campaign_id_for_generation=request.campaign_id,
            action_type="generate"
        )
        return {"message": "Journeys generated successfully", "data": updated_journeys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate journeys: {e}")

@app.put("/journeys/modify")
async def modify_journey(request: JourneyModifyRequest):
    """
    Modifies a specific marketing journey by ID.
    Returns the updated list of journey records from Supabase.
    """
    try:
        updated_journeys = await process_journeys(
            user_prompt=request.user_prompt,
            journey_id_to_affect=request.journey_id,
            action_type="update_singular"
        )
        return {"message": f"Journey {request.journey_id} modified successfully", "data": updated_journeys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to modify journey: {e}")

@app.delete("/journeys/delete/{journey_id}")
async def delete_journey(journey_id: str):
    """
    Deletes a specific marketing journey by ID.
    Returns the updated list of journey records from Supabase.
    """
    try:
        updated_journeys = await process_journeys(
            journey_id_to_affect=journey_id,
            action_type="delete_singular"
        )
        return {"message": f"Journey {journey_id} deleted successfully", "data": updated_journeys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete journey: {e}")

# The if __name__ == "__main__": block is removed for production deployment with Gunicorn
# uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
