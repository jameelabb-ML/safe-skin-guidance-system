# safe-skin-guidance-system
A safety-first multimodal AI assistant designed to promote early skin awareness using MedGemma (HAI-DEF) and embedding-driven agent orchestration.

---

##  Problem Statement

Many individuals notice unusual skin changes but hesitate to consult a dermatologist due to cost, access limitations, or uncertainty. However, fully automated diagnostic AI systems pose serious risks, including overconfident medical claims and hallucinated diagnoses.

There is a need for a responsible, non-diagnostic AI system that promotes awareness without replacing medical professionals.

The Safe Skin Guidance System addresses this gap.

---

##  Overview

This system is a healthcare-adjacent multimodal AI assistant that:

- Accepts a user-uploaded skin image
- Asks structured follow-up questions
- Provides 4–5 safe, non-medical guidance tips
- Encourages professional consultation

The system does NOT provide diagnoses.

---

## Architecture

The system uses a 4-agent modular pipeline:

### Agent 1 – Image Validation
- HSV-based color filtering
- Semantic embedding comparison
- Ensures image contains human skin

### Agent 2 – Image Similarity
- Generates ViT-B-32 embeddings
- Compares with reference embeddings
- Computes similarity score

### Agent 3 – Structured Follow-Up
- Predefined questions selected based on similarity score
- Controls depth of inquiry safely

### Agent 4 – Guidance Generation
- Uses MedGemma 1.5 (HAI-DEF)
- Combines image + user responses
- Generates non-diagnostic guidance
- Filters medical terminology
- Avoids naming diseases

---

## Why HAI-DEF (MedGemma)?

Generic multimodal models may produce unsafe or overconfident outputs in medical contexts.

MedGemma, as part of the HAI-DEF family, is aligned for healthcare-adjacent reasoning tasks. Its medical-domain grounding allows structured, safe guidance generation while minimizing hallucination risk.

This makes it significantly more appropriate than general-purpose vision-language models for sensitive applications.

---

## Embedding System

Model: ViT-B-32

Used for:
- Image validation
- Similarity scoring
- Follow-up control logic

Reference Images:
- Kaggle Skin Disease Dataset
- Wikimedia Commons

Licenses:
- CC BY 4.0
- Public Domain

---

##  Safety & Ethics

- Non-diagnostic design
- Negative-constraint prompting
- Post-output sanitization
- No image storage
- Stateless inference
- Encourages professional consultation

This project does not provide medical advice.

---

## ⚙ Performance Optimization

- Hybrid inference pipeline
- Embedding filtering reduces unnecessary MedGemma calls
- ~30% efficiency improvement over full multimodal processing
- Optimized for Hugging Face Spaces cold starts

---

## ☁ Deployment

Live Demo:
https://huggingface.co/spaces/jameelabibi/skin_app

The system was developed without access to a local GPU PC and optimized for shared GPU environments on Hugging Face Spaces.

---

## Tech Stack

- Python
- Gradio
- MedGemma 1.5 (HAI-DEF)
- ViT-B-32
- Hugging Face Spaces

---

## License

Project Code: [Apache 2.0]

Reference Images:
- CC BY 4.0
- Public Domain

---

## Author

Jameela Bibi  
AI Developer & System Architect

Independently designed and implemented:
- Embedding comparison logic
- Agent-based orchestration
- Safety-controlled prompting
- Deployment architecture
