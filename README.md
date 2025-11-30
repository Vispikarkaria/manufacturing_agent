# Manufacturing Co-Pilot (RAG)
Turns a CAD image into a vetted manufacturing plan using OpenAI vision + text models, a lightweight RAG layer, and a multi-agent orchestrator in `manufacturing_co_pilot_rag.py`.

## Contents
- `manufacturing_co_pilot_rag.py` — single-file pipeline with all agents, RAG helpers, and example entry point.
- `multi_model_to_vector.py` — multimodal embedder for text/image/audio/video into an in-memory vector store.
- `cad_images/` — place CAD renders (PNG/JPG). `__main__` uses `cad_images/part_1.png`.
- `rags/txt`, `rags/pdf` — drop optional knowledge files for retrieval; created on startup.
- `uploads/images|text|audio|video` — scratch folders to collect files you want to embed with `multi_model_to_vector.py`.
- `agent_communication_log.txt` — chronological trace of each agent’s messages.
- `final_answer.txt` — human-oriented explanation from the Interpreter Agent.

## Requirements
- Python 3.10+ recommended.
- Packages: `openai`, `python-dotenv`, `numpy`, `pypdf` (only needed if you ingest PDFs).
- Environment variable: `OPENAI_API_KEY` (loaded via `.env` or shell env).

Install:
```bash
pip install openai python-dotenv numpy pypdf
```

## Quickstart (CLI)
1) Export your key: `export OPENAI_API_KEY=sk-...` (or create a `.env` with `OPENAI_API_KEY=`).  
2) Add a CAD image at `cad_images/part_1.png` (or adjust the path in `__main__`).  
3) Run: `python manufacturing_co_pilot_rag.py`  
4) Inspect `agent_communication_log.txt` for the trace and `final_answer.txt` for the explanation (also printed).

## How it works (data flow)
1. **CAD Feature Agent** (`cad_feature_agent`): Vision call extracts features/materials/constraints from the CAD image, enriched by retrieved RAG snippets.
2. **Feature List Builder** (`build_manufacturing_feature_list`): Embeds the task description and ranks similar past cases from a small library.
3. **Manufacturing Agent** (`manufacturing_agent`): Proposes process chain, parameters, resources, quality checks, and risks using retrieved process docs.
4. **Process Checker** (`manufacturing_process_checker`): Static QA for missing parameters, infeasible tolerances, unsafe temps/loads, or ordering issues.
5. **Interpreter Agent** (`interpreter_agent`): Converts the plan + checker feedback into a concise engineer-facing narrative.
6. **Logging**: `reset_communication_log`, `log_agent_message`, and `write_final_answer` capture artifacts to disk.

## RAG and retrieval
- Uploads: place `.txt` in `rags/txt` and `.pdf` in `rags/pdf`. PDF ingestion requires `pypdf`; otherwise PDF files are skipped with a log warning.
- Loader: `load_rag_documents()` reads files into `Document` objects with simple metadata (`filename`, `source_type`).
- Store: `SimpleVectorStore` (in-memory, cosine similarity) embeds documents using `text-embedding-3-large`; used by the CAD Feature Agent and Manufacturing Agent for context.
- Default knowledge: `run_manufacturing_copilot` seeds `kb_docs` (general guidelines) and `process_library` (example processes). Extend these lists to steer retrieval.

## Key configuration knobs
- Models: text + vision default to `gpt-5.1`; embeddings to `text-embedding-3-large`. Change in `call_text_model` / `call_vision_model_for_cad`.
- Inputs: `run_manufacturing_copilot(cad_image_url, solver_hint=None, seed=None, temps=None)` — set solver hints, IDs/seeds, and allowable temperatures to guide retrieval and logging.
- Image source: local paths are converted to data URLs; remote `http(s)` or `data:` URLs pass through.
- Similarity depth: adjust `k` in `SimpleVectorStore.search` or top-N selection in `build_manufacturing_feature_list`.

## Multimodal embedding helper (`multi_model_to_vector.py`)
- Purpose: turn text, images (captioned via `gpt-5.1`), audio (transcribed via Whisper), and video (sampled + summarized via `gpt-5.1`) into embeddings using `text-embedding-3-large` for later RAG.
- Inputs: point the script at any file paths; use the provided `uploads/images`, `uploads/text`, `uploads/audio`, `uploads/video` as convenient staging areas.
- Run example:  
  `python multi_model_to_vector.py uploads/images/part.png uploads/text/notes.txt uploads/video/demo.mp4 --query "aluminum bracket with slots"`
- Requirements: `opencv-python` needed for video frame extraction; optional if you only use text/image/audio.
- Behavior: stores documents and embeddings in memory during the run; prints an optional top-3 retrieval for the `--query` string.

## Programmatic use
```python
from manufacturing_co_pilot_rag import run_manufacturing_copilot

run_manufacturing_copilot(
    cad_image_url="cad_images/my_part.png",
    solver_hint="Prefer bending over milling for flanges",
    seed=123,
    temps="Keep bulk temperature below 120 C"
)
```
Inspect `agent_communication_log.txt` and `final_answer.txt` after the run.

## Troubleshooting
- Missing PDFs support: install `pypdf` or remove PDFs from `rags/pdf`.
- Empty retrieval: ensure `.txt`/`.pdf` files exist, or expand `kb_docs` / `process_library`.
- API key errors: confirm `OPENAI_API_KEY` is set and dotenv can load it.
- High token usage: trim RAG files or narrow prompts; embeddings are computed for all loaded docs.

## Extending
- Swap `SimpleVectorStore` for a persistent/vector DB (FAISS/Chroma/cloud) while keeping the same `.search` interface.
- Add richer process libraries and CAD examples to improve retrieval grounding.
- Introduce structured schemas for features/process steps and validate before passing between agents.
