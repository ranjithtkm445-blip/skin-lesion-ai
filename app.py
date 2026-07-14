import os
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_assets import download_assets
download_assets()

import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.predict import predict_from_bytes
from src.explain import explain_from_bytes
from src.llm_explain import explain_prediction, answer_question
from config import CLASS_LABELS

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Skin Lesion AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    predicted_class: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str

class PredictResponse(BaseModel):
    predicted_class: str
    class_name: str
    confidence: float
    probabilities: dict
    heatmap_base64: Optional[str] = None
    explanation: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), explain: bool = True, llm_explain: bool = True):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await file.read()
    result = predict_from_bytes(image_bytes)

    heatmap_b64 = None
    if explain:
        try:
            pred_idx = CLASS_LABELS[result["predicted_class"]]
            heatmap_b64 = explain_from_bytes(image_bytes, pred_idx)
        except:
            pass

    explanation = None
    if llm_explain:
        try:
            explanation = explain_prediction(
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )
        except Exception as e:
            explanation = f"LLM unavailable: {e}"

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

# ── Minimal Gradio UI (required for HF Spaces) ────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🔬 DermAI — Skin Lesion API
    This Space runs the **FastAPI backend** for the DermAI skin lesion classifier.

    ### API Endpoints
    - `GET /health` — health check
    - `POST /predict` — classify an image (returns prediction + Grad-CAM + LLM explanation)
    - `POST /explain` — ask a question about a diagnosis

    ### Frontend
    Visit the full UI at: **[ranjithtkm445-blip.github.io/skin-lesion-ai](https://ranjithtkm445-blip.github.io/skin-lesion-ai)**
    """)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
