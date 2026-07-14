---
title: Skin Lesion AI Classifier
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: 1.41.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# 🔬 Skin Lesion AI Classifier

AI-powered skin lesion classification using HAM10000 dataset.

## Features
- **EfficientNet-B3** trained on 10,015 HAM10000 images (~87.9% val accuracy)
- **7 lesion classes**: akiec, bcc, bkl, df, mel, nv, vasc
- **Grad-CAM** heatmap explainability
- **RAG + Groq LLM** clinical explanations from DermNet NZ
- **Interactive Q&A** with 10 preset medical questions

## Setup
Set `GROQ_API_KEY` in Space secrets.

> ⚠️ For educational purposes only. Not a substitute for professional medical diagnosis.
