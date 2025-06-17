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
from audience_analyser import process_audiences, load_json, CAMPAIGN_STRATEGY_PATH
from growth_levers import process_growth_levers
from campaign_generator import process_campaign_ideas
from flow_generator import process_flows

app = FastAPI(
    title="Campaign Strategy API",
    description="API for generating and managing campaign strategies, audiences, growth levers, ideas, and flows.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your frontend (e.g., React app running on localhost:3000) to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's actual origin(s) in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Request Body Models for API Endpoints ---

class AudienceRequest(BaseModel):
    user_prompt: str
    is_modification: bool = False
    current_audiences_data: Optional[Dict[str, Any]] = None # Dictionary of { "Type 1": {...}, "Type 2": {...} }

class GrowthLeverRequest(BaseModel):
    user_prompt: str
    is_modification: bool = False
    current_growth_levers_data: Optional[Dict[str, Any]] = None # Dictionary of { "gl_id1": {...}, "gl_id2": {...} }

class CampaignIdeaGenerateRequest(BaseModel):
    # For generating new campaign ideas based on specific audience and growth lever
    audience_type_number: int
    growth_lever_id: str
    user_prompt: str = ""

class CampaignIdeaModifyRequest(BaseModel):
    # For modifying existing campaign ideas
    user_prompt: str
    # When modifying, LLM gets the entire list and updates it.
    # The user_prompt should specify which idea (by ID) to modify/remove/add.
    # No need to send current_campaign_ideas_data here as it's loaded by the processing function.

class FlowGenerateRequest(BaseModel):
    campaign_id: str
    preferred_channels: List[str] = []
    user_prompt: str = ""

class FlowModifyRequest(BaseModel):
    user_prompt: str
    # Similar to campaign modification, LLM handles the list.

# --- API Endpoints ---

@app.get("/strategy", response_class=JSONResponse)
async def get_campaign_strategy():
    """
    Retrieves the entire campaign_strategy.json.
    """
    strategy = load_json(CAMPAIGN_STRATEGY_PATH)
    return strategy

@app.post("/audiences", response_model=Dict[str, Any])
async def post_audiences(request: AudienceRequest):
    """
    Generates or modifies audience types.
    - For initial generation: send is_modification=false, user_prompt.
    - For modification: send is_modification=true, user_prompt (describing changes), current_audiences_data.
    Returns the updated 'Audience types' section.
    """
    try:
        updated_audiences = await process_audiences(
            user_prompt=request.user_prompt,
            is_modification=request.is_modification,
            current_audiences_data=request.current_audiences_data
        )
        return updated_audiences
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audiences: {e}")

@app.post("/growth-levers", response_model=Dict[str, Any])
async def post_growth_levers(request: GrowthLeverRequest):
    """
    Generates or modifies growth levers.
    - For initial generation: send is_modification=false, user_prompt.
    - For modification: send is_modification=true, user_prompt (describing changes), current_growth_levers_data.
    Returns the updated 'Growth Levers' section.
    """
    try:
        updated_levers = await process_growth_levers(
            user_prompt=request.user_prompt,
            is_modification=request.is_modification,
            current_growth_levers_data=request.current_growth_levers_data
        )
        return updated_levers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process growth levers: {e}")

@app.post("/campaign-ideas", response_model=Dict[str, Any])
async def post_campaign_ideas(request: Request):
    """
    Generates or modifies campaign ideas.
    - To generate: send {"action": "generate", "combinations": [{"audience_type_number": 1, "growth_lever_id": "xyz"}, ...], "user_prompt": "..."}
    - To modify: send {"action": "modify", "user_prompt": "..."} (prompt should specify campaign ID to modify/remove)
    Returns the updated 'Campaign ideas' section.
    """
    req_body = await request.json()
    action = req_body.get("action")
    user_prompt = req_body.get("user_prompt", "")

    if action == "generate":
        combinations = req_body.get("combinations", [])
        if not combinations:
            raise HTTPException(status_code=400, detail="'combinations' array is required for 'generate' action.")
        try:
            updated_campaigns = await process_campaign_ideas(
                action="generate",
                campaign_ids_for_generation=combinations,
                user_prompt=user_prompt
            )
            return updated_campaigns
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate campaign ideas: {e}")
    elif action == "modify":
        try:
            updated_campaigns = await process_campaign_ideas(
                action="modify",
                user_prompt=user_prompt
            )
            return updated_campaigns
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to modify campaign ideas: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'generate' or 'modify'.")

@app.post("/flows", response_model=Dict[str, Any])
async def post_flows(request: Request):
    """
    Generates or modifies marketing flows.
    - To generate: send {"action": "generate", "campaign_id": "xyz", "preferred_channels": ["Email"], "user_prompt": "..."}
    - To modify: send {"action": "modify", "user_prompt": "..."} (prompt should specify flow ID to modify/remove)
    Returns the updated 'Flows' section.
    """
    req_body = await request.json()
    action = req_body.get("action")
    user_prompt = req_body.get("user_prompt", "")

    if action == "generate":
        campaign_id = req_body.get("campaign_id")
        preferred_channels = req_body.get("preferred_channels", [])
        if not campaign_id:
            raise HTTPException(status_code=400, detail="'campaign_id' is required for 'generate' action.")
        try:
            updated_flows = await process_flows(
                action="generate",
                campaign_id_for_generation=campaign_id,
                preferred_channels_for_generation=preferred_channels,
                user_prompt=user_prompt
            )
            return updated_flows
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate flows: {e}")
    elif action == "modify":
        try:
            updated_flows = await process_flows(
                action="modify",
                user_prompt=user_prompt
            )
            return updated_flows
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to modify flows: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'generate' or 'modify'.")

# --- Run the FastAPI App ---
if __name__ == "__main__":
    # To run this API:
    # 1. Make sure you have uvicorn installed: pip install uvicorn
    # 2. Save this file as api.py
    # 3. Run from your terminal in the same directory: uvicorn api:app --reload --port 8000
    #    --reload is useful for development as it restarts the server on code changes.
    #    --port 8000 specifies the port.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

