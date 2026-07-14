import os
import sys
import base64
import io
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Download model + vectorstore from HF Hub if running on Streamlit Cloud
from download_assets import download_assets
download_assets()

import streamlit as st
from PIL import Image

from src.predict import predict_from_bytes
from src.explain import explain_from_bytes
from src.llm_explain import explain_prediction, answer_question
from config import CLASS_LABELS, CLASS_NAMES, IDX_TO_CLASS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e40af; }
    .sub-header  { font-size: 1rem; color: #6b7280; margin-bottom: 1.5rem; }
    .result-card { background: #f0f9ff; border-left: 4px solid #2563eb;
                   padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .warning-box { background: #fffbeb; border-left: 4px solid #f59e0b;
                   padding: 0.75rem; border-radius: 0.5rem; font-size: 0.85rem; }
    .class-bar   { height: 8px; border-radius: 4px; background: #e5e7eb; margin: 2px 0; }
    .class-fill  { height: 8px; border-radius: 4px; background: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Skin Lesion AI")
    st.markdown("**Model:** EfficientNet-B3  \n**Dataset:** HAM10000  \n**Accuracy:** ~87.9%")
    st.divider()
    st.markdown("**7 Lesion Classes:**")
    for k, name in zip(CLASS_LABELS.keys(), CLASS_NAMES):
        st.markdown(f"- `{k}` — {name.split('(')[0].strip()}")
    st.divider()
    st.markdown(
        "<div class='warning-box'>⚠️ For educational purposes only. "
        "Not a substitute for professional medical diagnosis.</div>",
        unsafe_allow_html=True,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🔬 Skin Lesion AI Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-powered classification with Grad-CAM explainability and clinical LLM analysis</div>", unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "🔥 Results & Heatmap", "💬 Ask AI"])

# ── Session state ───────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result      = None
if "heatmap_b64" not in st.session_state:
    st.session_state.heatmap_b64 = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Upload & Analyze
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        uploaded = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])


    with col2:
        st.markdown("### Preview")
        image_bytes = None
        if uploaded:
            image_bytes = uploaded.read()
            st.session_state.image_bytes = image_bytes
        elif st.session_state.image_bytes:
            image_bytes = st.session_state.image_bytes

        if image_bytes:
            st.image(Image.open(io.BytesIO(image_bytes)), width=400)

            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Running AI analysis..."):
                    # Predict
                    result = predict_from_bytes(image_bytes)
                    st.session_state.result = result

                    # Grad-CAM
                    pred_idx = CLASS_LABELS[result["predicted_class"]]
                    st.session_state.heatmap_b64 = explain_from_bytes(image_bytes, pred_idx)

                    # LLM explanation
                    st.session_state.explanation = explain_prediction(
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"],
                        probabilities=result["probabilities"],
                    )
                    st.session_state.chat_history = []

                st.success("Analysis complete! See Results & Heatmap tab.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Results & Heatmap
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if st.session_state.result is None:
        st.info("Upload and analyze an image first.")
    else:
        result = st.session_state.result
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Classification Result")
            st.markdown(f"""
            <div class='result-card'>
                <b>Predicted Class:</b> {result['class_name']}<br>
                <b>Code:</b> <code>{result['predicted_class']}</code><br>
                <b>Confidence:</b> {result['confidence']*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Class Probabilities")
            for cls, prob in result["probabilities"].items():
                label = CLASS_NAMES[CLASS_LABELS[cls]].split("(")[0].strip()
                pct   = prob * 100
                color = "#2563eb" if cls == result["predicted_class"] else "#9ca3af"
                st.markdown(f"""
                <div style='margin:4px 0'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='font-size:0.85rem'><code>{cls}</code> {label}</span>
                        <span style='font-size:0.85rem;font-weight:600'>{pct:.1f}%</span>
                    </div>
                    <div class='class-bar'>
                        <div class='class-fill' style='width:{min(pct,100)}%;background:{color}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Grad-CAM Heatmap")
            if st.session_state.heatmap_b64:
                heatmap_bytes = base64.b64decode(st.session_state.heatmap_b64)
                st.image(Image.open(io.BytesIO(heatmap_bytes)), width=400,
                         caption="Red = high model attention")
            else:
                st.warning("Heatmap not available.")

        st.divider()
        st.markdown("### 🤖 Clinical AI Explanation")
        if st.session_state.explanation:
            st.markdown(st.session_state.explanation)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Ask AI
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.result is None:
        st.info("Analyze an image first to enable Q&A.")
    else:
        pred_cls = st.session_state.result["predicted_class"]
        st.markdown(f"### 💬 Ask about your diagnosis: `{pred_cls}`")

        # ── Quick question buttons ─────────────────────────────────────────────
        QUICK_QUESTIONS = [
            "What is this condition?",
            "Is this dangerous or life-threatening?",
            "What are the common symptoms?",
            "What causes this condition?",
            "What treatment options are available?",
            "Should I see a doctor urgently?",
            "Can this condition spread or worsen?",
            "How is this diagnosed by a dermatologist?",
            "What lifestyle changes can help?",
            "What is the long-term outlook for this condition?",
        ]

        if "quick_q" not in st.session_state:
            st.session_state.quick_q = None

        st.markdown("**Quick Questions — click to ask:**")
        q_cols = st.columns(2)
        for i, q in enumerate(QUICK_QUESTIONS):
            with q_cols[i % 2]:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state.quick_q = q

        st.divider()

        # ── Chat history ───────────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle quick question click OR typed input
        typed_q = st.chat_input("Or type your own question...")
        question = st.session_state.quick_q or typed_q

        if question:
            st.session_state.quick_q = None
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(question=question, predicted_class=pred_cls)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
