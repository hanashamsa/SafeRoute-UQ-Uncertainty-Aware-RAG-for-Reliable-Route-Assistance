# SafeRoute-UQ  
**Uncertainty-Aware Retrieval-Augmented Generation for Reliable Route Assistance**

## Overview
SafeRoute-UQ is an end-to-end RAG system that augments language model answers with calibrated uncertainty estimation and risk-aware decision gating. The system is designed for route and road-notice question answering, where reliability and safe behavior are more important than fluent responses alone.

Instead of only generating answers, SafeRoute-UQ explicitly estimates *how much an answer should be trusted*.

## Key Features
- Dense retrieval using FAISS + SentenceTransformers
- White-box generation uncertainty (token-level entropy)
- Optional LM-Polygraph integration for additional uncertainty signals
- Fusion-based trust score (retrieval + generation uncertainty)
- Calibrated green / amber / red decision gating
- Interactive Gradio-based UI with diagnostics

## Pipeline
1. **Corpus ingestion** (JSONL road notices)
2. **Embedding & indexing** (FAISS cosine similarity)
3. **Query-time retrieval**
4. **RAG-based answer generation**
5. **Uncertainty extraction**
6. **Trust score fusion & calibration**
7. **Decision gating (green / amber / red)**

## Interpretation of Decisions
- **Green**: Reliable behavior (including correct abstention)
- **Amber**: Moderate uncertainty; verification recommended
- **Red**: Unreliable or unsafe; do not act on output

## Evaluation
- Evaluated on 10-item and 50-item labeled datasets
- Metrics include AUROC, AUPRC, ECE, and risk–coverage curves
- Demonstrates improved uncertainty calibration as data scale increases

## Technologies Used
- Python
- Hugging Face Transformers
- SentenceTransformers
- FAISS
- Gradio
- NumPy / SciPy

## Limitations
- Synthetic or small-scale datasets
- No real-time automotive integration
- Limited large-scale ablations

## Use Cases
- Route assistance systems
- Mapping and navigation AI
- Decision-support interfaces
- Research on uncertainty-aware RAG pipelines


