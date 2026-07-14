---
title: Skin Lesion API
emoji: 🔬
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
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
