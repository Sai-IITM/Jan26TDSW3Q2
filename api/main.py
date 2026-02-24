from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Sentiment Analysis API")

class CommentRequest(BaseModel):
    comment: str

# ✅ FIXED: Added "name" field + "strict": true
response_schema = {
    "name": "sentiment_response",  # REQUIRED by AIpipe!
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "rating": {"type": "integer", "minimum": 1, "maximum": 5}
        },
        "required": ["sentiment", "rating"],
        "additionalProperties": False
    }
}

async def get_ai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing AIpipe token")
    
    return httpx.AsyncClient(
        base_url="https://aipipe.org/openai/v1",
        headers={"Authorization": f"Bearer {api_key}"}
    )

@app.post("/comment")
async def analyze_comment(request: CommentRequest, client: httpx.AsyncClient = Depends(get_ai_client)):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"Analyze sentiment of: {request.comment}"}],
                "response_format": {"type": "json_schema", "json_schema": response_schema}
            },
            timeout=30.0
        )
        
        response.raise_for_status()
        data = response.json()
        result = json.loads(data["choices"][0]["message"]["content"])
        return result
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"AIpipe {e.response.status_code}: {e.response.text[:100]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
