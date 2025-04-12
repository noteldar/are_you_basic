from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from cost_function import evaluate_text
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Are You Basic API",
    description="API for evaluating how basic (AI-generated vs human) a response is",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define request models
class Message(BaseModel):
    role: str
    content: str


class ConversationRequest(BaseModel):
    conversation: List[Message]


# Define response models
class EvaluationResult(BaseModel):
    final_score: float
    ai_detection_score: float
    coherence_score: float
    human_score: float
    length_factor: float
    is_human: bool


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_conversation(request: ConversationRequest):
    """
    Evaluate a conversation to determine if the latest message appears to be human-written.

    The conversation should include at least 2 messages, typically in the format:
    - System message (optional)
    - User message (question)
    - Assistant message (response to evaluate)

    Returns scores indicating how likely the response is to be human-written vs AI-generated.
    """
    # Convert Pydantic models to dictionaries for evaluate_text
    conversation = [msg.dict() for msg in request.conversation]

    # Ensure we have at least a question and response
    if len(conversation) < 2:
        raise HTTPException(
            status_code=400,
            detail="Conversation must contain at least 2 messages (question and response)",
        )

    try:
        # Call the evaluation function
        result = evaluate_text(conversation)

        # Add is_human flag for easier frontend processing
        result["is_human"] = result["final_score"] >= 0.5

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Run the API server when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
