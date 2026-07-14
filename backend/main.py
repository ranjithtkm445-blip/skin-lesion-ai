import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.predict import predict_from_bytes
from src.explain import explain_from_bytes
from src.llm_explain import explain_prediction, answer_question
from config import API_HOST, API_PORT, CLASS_LABELS

app = FastAPI(
    title="Skin Lesion AI API",
    description="AI-powered skin lesion classification with Grad-CAM and clinical LLM explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    predicted_class: str
    class_name: str
    confidence: float
    probabilities: dict
    heatmap_base64: Optional[str] = None
    explanation: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
    predicted_class: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "efficientnet_b3", "classes": list(CLASS_LABELS.keys())}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    explain: bool = True,
    llm_explain: bool = True,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    # Classification
    result = predict_from_bytes(image_bytes)

    # Grad-CAM heatmap
    heatmap_b64 = None
    if explain:
        try:
            pred_idx    = CLASS_LABELS[result["predicted_class"]]
            heatmap_b64 = explain_from_bytes(image_bytes, pred_idx)
        except Exception as e:
            heatmap_b64 = None

    # LLM clinical explanation
    explanation = None
    if llm_explain:
        try:
            explanation = explain_prediction(
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )
        except Exception as e:
            explanation = f"LLM explanation unavailable: {e}"

    return PredictResponse(
        predicted_class=result["predicted_class"],
        class_name=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        heatmap_base64=heatmap_b64,
        explanation=explanation,
    )


@app.post("/explain", response_model=QuestionResponse)
async def ask(req: QuestionRequest):
    try:
        answer = answer_question(req.question, req.predicted_class)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=False)
