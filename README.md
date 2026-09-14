# Document Summarizer

A document summarizer with a two-stage pipeline: classify the document's topic, then route it to either extractive or abstractive summarization depending on the category. FastAPI backend + Streamlit frontend. [Live demo](https://huggingface.co/spaces/Raannaa/Doc_summarizer)

## Why this project
I wanted to build a summarizer that doesn't use one-size-fits-all summarization — the idea being that dense technical/medical/legal text benefits more from extracting key phrases (so nothing critical gets paraphrased away), while general text (news, entertainment, etc.) works well with full abstractive summarization.

## How it works

1. **Classification** — a DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) with a custom classification head predicts one of 8 categories: legal, medical, news, entertainment, sports, technology, politics, education.
2. **Routing** — based on the predicted category:
   - **medical / legal** → extractive summarization via **KeyBERT** (keyphrase extraction)
   - **everything else** → abstractive summarization via a **Pegasus** sequence-to-sequence model
3. The FastAPI backend exposes a single `/predict` endpoint; the Streamlit frontend accepts pasted text or an uploaded `.txt` / `.docx` / `.pdf` file, sends it to the backend, and displays the returned summary.

## Architecture
```
User → Streamlit UI → HTTP POST /predict → FastAPI backend
                                              → document_classification()
                                                  ├─ medical / legal → KeyBERT (extractive)
                                                  └─ everything else → Pegasus (abstractive)
                                              ← summary
                       ← rendered in UI ←
```

## Project structure
```
backend/
├── fastapi_main.py     # FastAPI app — /predict endpoint
└── functions.py         # classification + summarization logic
frontend/
└── streamlit_app.py     # Streamlit UI, calls the backend over HTTP
requirements.txt
```

## Running it

**Backend:**
```bash
cd backend
pip install -r ../requirements.txt
uvicorn fastapi_main:app --reload
```
Runs at `http://127.0.0.1:8000`.

**Frontend** (in a separate terminal):
```bash
cd frontend
streamlit run streamlit_app.py
```

## Tech Stack
Python · Hugging Face Transformers (DistilBERT, Pegasus) · KeyBERT · FastAPI · Pydantic · Streamlit · python-docx · pypdf

## Current limitations

- **The classification head is not yet fine-tuned.** The base model (`distilbert-base-uncased-finetuned-sst-2-english`) was originally trained for sentiment analysis; its output layer was replaced with a new 8-label head for this project (`ignore_mismatched_sizes=True`), but that head hasn't been trained on labeled examples for these categories yet. **The classification and routing logic reflects the intended design of the pipeline rather than a currently accurate classifier** — the plumbing between classification → routing → summarization works end-to-end, but the category prediction itself isn't reliable yet.
- Next step to fix this: fine-tune the classification head on a small labeled dataset (a few hundred examples per category would likely be enough to get meaningful predictions).
- No automated tests yet.
- Backend URL is hardcoded to localhost in the frontend for local development; a hosted URL is used for the deployed demo.

## What I'd improve
- Fine-tune the classification head on real labeled data per category
- Add a `Dockerfile` / `docker-compose.yml` so backend + frontend can be spun up together with one command
- Add tests for `document_classification()` and `summarize_document()`
- Handle the case where an uploaded file's extension doesn't match its actual content
