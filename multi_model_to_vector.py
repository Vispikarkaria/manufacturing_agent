"""
Multi-modal to vector store helper.

Converts text, images, audio, and simple videos into text representations,
then embeds them for RAG use. Uses OpenAI vision for image/video captions,
Whisper for audio transcripts, and text embeddings for all vectors.
"""

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Optional dependency for video frame sampling.
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

load_dotenv()
client = OpenAI()


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleVectorStore:
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        self.embedding_model = embedding_model
        self.docs: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        response = client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        self.docs.extend(docs)
        new_emb = self._embed([d.text for d in docs])
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.docs or self.embeddings is None:
            return []
        q_vec = self._embed([query])[0]
        sims = (self.embeddings @ q_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )
        indices = np.argsort(-sims)[:k]
        return [self.docs[i] for i in indices]


def _data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    data_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data_b64}"


def caption_image(image_path: Path, model: str = "gpt-5.1") -> str:
    system_prompt = "You describe images for retrieval; keep it concise but specific."
    content = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image for later search."},
                {"type": "input_image", "image_url": _data_url(image_path)},
            ],
        },
    ]
    resp = client.responses.create(model=model, input=content)
    return resp.output_text.strip()


def transcribe_audio(audio_path: Path, model: str = "whisper-1") -> str:
    with audio_path.open("rb") as f:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="text",
        )
    return transcript.strip()


def _extract_video_frames(video_path: Path, max_frames: int = 3) -> List[str]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video frame extraction.")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    frames: List[str] = []
    idx = 0
    while len(frames) < max_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            b64 = base64.b64encode(buf).decode("utf-8")
            frames.append(f"data:image/jpeg;base64,{b64}")
        idx += step
    cap.release()
    return frames


def summarize_video(video_path: Path, model: str = "gpt-5.1", max_frames: int = 3) -> str:
    frames = _extract_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return "Empty video or could not extract frames."
    system_prompt = "You summarize video frames for retrieval; include objects, actions, and scene context."
    user_content = [{"type": "input_text", "text": "Summarize the key visual content and ordering."}]
    for img in frames:
        user_content.append({"type": "input_image", "image_url": img})
    content = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    resp = client.responses.create(model=model, input=content)
    return resp.output_text.strip()


def embed_file(
    path: Path,
    vector_store: SimpleVectorStore,
    embedding_model: str = "text-embedding-3-large",
    caption_model: str = "gpt-5.1",
    transcription_model: str = "whisper-1",
) -> None:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or ""
    text_payload = ""
    meta = {"source_path": str(path), "mime": mime}

    if mime.startswith("text/") or path.suffix.lower() in {".txt", ".md", ".json"}:
        text_payload = path.read_text(encoding="utf-8", errors="ignore")
        meta["modality"] = "text"
    elif mime.startswith("image/"):
        text_payload = caption_image(path, model=caption_model)
        meta["modality"] = "image"
    elif mime.startswith("audio/"):
        text_payload = transcribe_audio(path, model=transcription_model)
        meta["modality"] = "audio"
    elif mime.startswith("video/"):
        text_payload = summarize_video(path, model=caption_model)
        meta["modality"] = "video"
    else:
        meta["modality"] = "unknown"
        text_payload = ""

    if not text_payload.strip():
        return

    vector_store.add_documents([Document(text=text_payload, metadata=meta)])


def embed_paths(paths: List[str]) -> SimpleVectorStore:
    store = SimpleVectorStore()
    for p in paths:
        path = Path(p).expanduser()
        if path.exists() and path.is_file():
            embed_file(path, store)
    return store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed text/image/audio/video into an in-memory vector store.")
    parser.add_argument("files", nargs="+", help="Paths to files to embed.")
    parser.add_argument("--query", help="Optional query to test retrieval.")
    args = parser.parse_args()

    vs = embed_paths(args.files)
    print(f"Embedded {len(vs.docs)} documents.")

    if args.query:
        results = vs.search(args.query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.metadata.get('modality')} {doc.metadata.get('source_path')}")
            print(doc.text[:400])
            print("-" * 40)
