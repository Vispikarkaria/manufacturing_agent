"""
Manufacturing Co Pilot, multi agent prototype using the OpenAI SDK

This script implements the complete workflow from your Manufacturing Co Pilot diagram:

1, CAD Feature Agent
   Uses a vision transformer style model, via the OpenAI vision endpoint,
   to convert a CAD diagram into a structured set of features and a high level task.

2, Semantic similarity and problem formulation
   Builds a manufacturing feature list by comparing the current task with a library of
   previous manufacturing examples using OpenAI embeddings.

3, Manufacturing Agent
   Uses a text model to synthesize a candidate manufacturing process plan
   given CAD features and similar reference processes.

4, Manufacturing Process Checker
   Uses a model to statically check the plan for runtime problems and feasibility issues.

5, Interpreter Agent
   Produces a human readable explanation of the validated process plan.

The implementation is intentionally modular so that you can later replace the
in memory vector store with a proper database, or plug the agents into
a tool such as LangGraph or Crew AI.
"""

import os
import json
import base64
import mimetypes
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from openai import OpenAI

# The client expects the environment variable OPENAI_API_KEY to be set.
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

COMMUNICATION_LOG_PATH = "agent_communication_log.txt"
FINAL_ANSWER_PATH = "final_answer.txt"

# --------------------------------------------------------------------
# Data containers, vector store, and RAG retriever
# --------------------------------------------------------------------


@dataclass
class Document:
    """
    Simple container for a single knowledge item.

    Parameters
    ----------
    text
        Natural language content of the document.
    metadata
        Optional dictionary with additional information, for example
        process name, material, source, and so on.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleVectorStore:
    """
    Very small in memory vector store.

    This class is only meant for prototyping. It stores all document
    embeddings in memory and uses cosine similarity for retrieval.

    In a real system you would usually replace this with a dedicated
    vector database such as a service backed by FAISS, Chroma, or a
    cloud vector store.

    Parameters
    ----------
    embedding_model
        Name of the OpenAI embedding model to use.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        self.embedding_model = embedding_model
        self.docs: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    # --------------- internal helpers ---------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts using the chosen OpenAI model.
        """
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    # --------------- public API ---------------

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the store and build their embeddings.
        """
        if not docs:
            return

        self.docs.extend(docs)
        new_emb = self._embed([d.text for d in docs])

        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top k most similar documents to the query text.

        Similarity is measured using cosine similarity in the embedding space.
        """
        if not self.docs or self.embeddings is None:
            return []

        q_vec = self._embed([query])[0]

        # Cosine similarity between query and all stored vectors
        sims = (self.embeddings @ q_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )

        indices = np.argsort(-sims)[:k]
        return [self.docs[i] for i in indices]


# --------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------


def reset_communication_log(path: str = COMMUNICATION_LOG_PATH) -> None:
    header = (
        "Manufacturing Co-Pilot agent communication log\n"
        f"Started: {datetime.utcnow().isoformat()}Z\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)


def log_agent_message(agent: str, message: Any, path: str = COMMUNICATION_LOG_PATH) -> None:
    if isinstance(message, (dict, list)):
        body = json.dumps(message, indent=2)
    else:
        body = str(message)

    entry = f"[{datetime.utcnow().isoformat()}Z] {agent}\n{body}\n\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def write_final_answer(text: str, path: str = FINAL_ANSWER_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------
# Shared helper for calling text models
# --------------------------------------------------------------------


def call_text_model(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5.1",
    json_only: bool = False,
) -> Any:
    """
    Helper around the Responses API for text only interactions.

    Parameters
    ----------
    system_prompt
        Instructions for the assistant.
    user_prompt
        Content that describes the task or question.
    model
        OpenAI model name. Uses gpt five point one by default.
    json_only
        When True, the function asks the model to return valid JSON and
        parses the result before returning it.

    Returns
    -------
    Either a plain string (for free form text) or a Python object parsed from JSON.
    """
    if json_only:
        user_prompt = user_prompt + "\n\nReturn only valid JSON, no extra text."

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text

    if json_only:
        return json.loads(text)
    return text


# --------------------------------------------------------------------
# Helper for calling a vision model in a vision transformer style
# --------------------------------------------------------------------


def call_vision_model_for_cad(
    cad_image_url: str,
    system_prompt: str,
    user_text: str,
    model: str = "gpt-5.1",
    json_only: bool = True,
) -> Any:
    """
    Helper for interacting with the OpenAI vision abilities.

    This function conceptually plays the role of a vision transformer.
    It receives a CAD diagram image and a textual prompt and returns
    a structured description of the content.

    Parameters
    ----------
    cad_image_url
        URL of the CAD image that the model can access.
        In a real system this may be a storage URL or a data URI.
    system_prompt
        High level instruction for the agent.
    user_text
        Additional textual context, such as solver hints, temperatures,
        and retrieved knowledge.
    model
        Name of the model used for reasoning on images.
    json_only
        Whether the answer is expected to be JSON.

    Returns
    -------
    Parsed JSON object or text string depending on the json_only flag.
    """
    # Allow local file paths by encoding them as data URLs, or pass through http(s) and data URLs.
    def _prepare_image_url(image_path_or_url: str) -> str:
        if image_path_or_url.startswith(("http://", "https://", "data:")):
            return image_path_or_url

        if os.path.exists(image_path_or_url):
            mime, _ = mimetypes.guess_type(image_path_or_url)
            mime = mime or "image/png"
            with open(image_path_or_url, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{data_b64}"

        raise ValueError(f"Image path or URL is invalid: {image_path_or_url}")

    prepared_image_url = _prepare_image_url(cad_image_url)

    # Build the combined content containing both image and text.
    content = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_text,
                },
                {
                    "type": "input_image",
                    # Responses API expects the URL string directly, not an object
                    "image_url": prepared_image_url,
                },
            ],
        },
    ]

    if json_only:
        # Ask explicitly for JSON to simplify processing later.
        content.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Return only valid JSON, no explanation.",
                    }
                ],
            }
        )

    response = client.responses.create(
        model=model,
        input=content,
    )

    text = response.output_text

    if json_only:
        return json.loads(text)
    return text


# --------------------------------------------------------------------
# CAD Feature Agent using a vision transformer style approach
# --------------------------------------------------------------------


def cad_feature_agent(
    cad_image_url: str,
    solver_hint: Optional[str],
    seed: Optional[int],
    temps: Optional[str],
    vector_store: SimpleVectorStore,
) -> Dict[str, Any]:
    """
    CAD Feature Agent

    This agent behaves like a vision transformer or diffusion style encoder
    that converts a CAD diagram into structured features.

    It receives
    a CAD image,
    small process hints such as solver suggestions and temperature limits,
    and a knowledge base for retrieval.

    The agent performs three conceptual steps:

    1, Vision transformer style encoding
       The OpenAI vision model inspects the CAD drawing and recognizes
       key geometric entities such as pockets, holes, slots, chamfers,
       fillets, and overall bounding dimensions.

    2, Fusion with retrieved knowledge
       The agent queries the vector store for documents related to the
       solver hint and uses these snippets as additional context, similar
       to RAG, to ground its interpretation in manufacturing practice.

    3, JSON feature extraction
       The model outputs a machine readable description of the CAD part
       and the manufacturing problem in JSON format.

    The returned JSON has the structure

    {
        "features": [ ... ],
        "materials": [ ... ],
        "constraints": [ ... ],
        "task_description": "..."
    }
    """

    # Retrieve relevant knowledge based on the solver hint or generic text.
    retrieval_query = solver_hint or "general CAD manufacturing guidelines"
    context_docs = vector_store.search(retrieval_query, k=5)

    context_text = "\n\n".join(
        [f"[Doc {i + 1}] {doc.text}" for i, doc in enumerate(context_docs)]
    )

    system_prompt = """
You are the CAD Feature Agent in a manufacturing co pilot.

You act like a vision transformer for CAD drawings:
you analyze the provided CAD image and convert it into a structured
representation of geometric and manufacturing features.

Tasks
1, Identify geometric features such as pockets, bosses, holes, threads,
   slots, ribs, chamfers, fillets, wall thicknesses, and overall dimensions.
2, Infer reasonable candidate materials and tolerance ranges from context.
3, Extract constraints that affect manufacturing, for example
   surface finish, unsupported overhangs, minimum radii, and tool access.
4, Summarize the manufacturing task in a short natural language description.

Output format
Return a JSON object with keys:
"features", a list of feature objects,
"materials", a list of material or alloy suggestions with notes,
"constraints", a list of constraint strings,
"task_description", a single short description string.

Each feature object should have keys such as
"type", "location", "size", "orientation", and "notes".
"""

    user_prompt = f"""
You are given a CAD diagram image of a mechanical part.

Use the image plus the provided context to infer manufacturing relevant
features and constraints.

Additional information:
Solver hint: {solver_hint}
Random seed or job identifier: {seed}
Permitted or desired temperature range: {temps}

Relevant manufacturing knowledge from the database:
{context_text}
"""

    feature_json = call_vision_model_for_cad(
        cad_image_url=cad_image_url,
        system_prompt=system_prompt,
        user_text=user_prompt,
        model="gpt-5.1",  # a model that supports vision reasoning
        json_only=True,
    )

    return feature_json


# --------------------------------------------------------------------
# Problem formulation,
# semantic similarity,
# and manufacturing feature list construction
# --------------------------------------------------------------------


def build_manufacturing_feature_list(
    feature_json: Dict[str, Any],
    library_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Construct a manufacturing feature list using semantic similarity.

    This function corresponds to
    semantic embeddings,
    lexical similarity,
    pairwise similarity matrix,
    and manufacturing feature list in the diagram.

    Steps
    1, Embed the current task description as a vector using OpenAI embeddings.
    2, Embed all descriptions from the library of previous manufacturing tasks.
    3, Compute cosine similarity between the current task and each library task.
    4, Rank the library tasks and keep the top matches.
    5, Return a compact structure that combines current CAD features with
       the most similar cases from the library.

    Parameters
    ----------
    feature_json
        Output of the CAD Feature Agent.
    library_features
        List of past manufacturing tasks. Each item should at least contain
        a "description" field.

    Returns
    -------
    Dictionary with keys:
    "target_task", "cad_features", and "similar_library_features".
    """

    target_desc = feature_json["task_description"]
    texts = [target_desc] + [item["description"] for item in library_features]

    embed_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    vectors = np.array([e.embedding for e in embed_response.data], dtype="float32")

    target_vec = vectors[0]
    library_vecs = vectors[1:]

    sims = (library_vecs @ target_vec) / (
        np.linalg.norm(library_vecs, axis=1) * np.linalg.norm(target_vec) + 1e-8
    )

    ranked = sorted(
        zip(library_features, sims),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_matches = [item[0] for item in ranked[:5]]

    return {
        "target_task": target_desc,
        "cad_features": feature_json["features"],
        "similar_library_features": top_matches,
    }


# --------------------------------------------------------------------
# Manufacturing Agent
# --------------------------------------------------------------------


def manufacturing_agent(
    feature_list: Dict[str, Any],
    process_library_docs: List[Document],
    vector_store: SimpleVectorStore,
) -> Dict[str, Any]:
    """
    Manufacturing Agent

    This agent converts the manufacturing feature list into a candidate
    end to end manufacturing plan.

    The agent follows this reasoning pattern:

    1, Retrieve relevant manufacturing process examples using the
       target task description as a search query.
    2, Combine retrieved context, CAD features, and similar features
       from the library.
    3, Propose a set of ordered process steps with parameters and resources.
    4, Lay out quality checks and potential risks.

    The returned JSON has keys
    "process_chain", "parameters", "resources", "quality_checks", and "risks".
    """

    # Ensure that the process library documents are present in the vector store.
    vector_store.add_documents(process_library_docs)

    query = feature_list["target_task"]
    context_docs = vector_store.search(query, k=5)
    context_text = "\n\n".join(
        [f"[Process {i + 1}] {doc.text}" for i, doc in enumerate(context_docs)]
    )

    system_prompt = """
You are the Manufacturing Agent in a manufacturing co pilot.

Given CAD features and a manufacturing task description,
synthesize an executable manufacturing plan.

Requirements
1, Propose a process chain as an ordered list of operations.
2, For each step specify key parameters such as feed, speed, power,
   layer thickness, temperature, and coolant usage where relevant.
3, Identify machines, fixtures, tools, and human or robot resources.
4, Define quality checks, inspection points, and sensors.
5, Highlight potential failure modes or risks and how to mitigate them.

Output format
Return a JSON object with keys:
"process_chain", list of step descriptions,
"parameters", dictionary keyed by step index,
"resources", list of machines and tools,
"quality_checks", list of checks and their timing,
"risks", list of identified risks and mitigations.
"""

    user_prompt = f"""
Target manufacturing task:
{feature_list["target_task"]}

CAD features extracted from the vision based CAD agent:
{json.dumps(feature_list["cad_features"], indent=2)}

Similar cases from the manufacturing feature library:
{json.dumps(feature_list["similar_library_features"], indent=2)}

Relevant processes retrieved from the knowledge base:
{context_text}
"""

    plan_json = call_text_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_only=True,
    )

    return plan_json


# --------------------------------------------------------------------
# Manufacturing Process Checker
# --------------------------------------------------------------------


def manufacturing_process_checker(
    plan_json: Dict[str, Any],
    vector_store: SimpleVectorStore,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Manufacturing Process Checker

    This agent performs static analysis of the proposed plan and flags
    potential runtime errors or infeasible configurations.

    The checker looks for
    missing parameters,
    impossible tolerances,
    unsafe loads or temperatures,
    inconsistent ordering,
    and conflicts with known guidelines.

    The returned report has keys
    "has_error", "issues", and "suggested_fixes".
    """

    system_prompt = """
You are the Manufacturing Process Checker in a manufacturing co pilot.

You receive a manufacturing process plan in JSON format and must review it
for possible runtime problems before any machine is started.

Checklist
1, Are any parameters missing or obviously under specified.
2, Are there tolerances or surface finish requirements that the chosen
   processes cannot reliably achieve.
3, Are there temperature, load, or stress levels that exceed safe limits
   for the assumed materials.
4, Are the process steps ordered in a way that is practical and safe.
5, Are there conflicts with common design and manufacturing guidelines.

Output format
Return JSON with keys:
"has_error", boolean,
"issues", list of strings describing each problem,
"suggested_fixes", list of strings with concrete improvements.
"""

    user_prompt = json.dumps(plan_json, indent=2)

    report = call_text_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_only=True,
    )

    has_error = bool(report.get("has_error", False))
    return has_error, report


# --------------------------------------------------------------------
# Interpreter Agent
# --------------------------------------------------------------------


def interpreter_agent(
    plan_json: Dict[str, Any],
    checker_report: Dict[str, Any],
) -> str:
    """
    Interpreter Agent

    Final layer that turns structured JSON outputs into a clear message
    for manufacturing engineers.

    The interpreter summarizes the strategy, explains each step with
    important parameters, notes the changes that were applied due to
    the checker feedback, and surfaces remaining open questions.
    """

    system_prompt = """
You are the Interpreter Agent in a manufacturing co pilot.

You receive a validated manufacturing plan and a checker report.
Your job is to explain the plan in clear technical language that a
manufacturing engineer can implement.

Explain
1, Overall strategy and why this sequence of processes was chosen.
2, Each step of the process chain with the most important parameters.
3, Fixes or changes that were introduced after the checker identified issues.
4, Assumptions, open questions, or items that still need human judgment.

Write your answer as a concise narrative with bullet lists where helpful.
"""

    user_prompt = f"""
Manufacturing plan JSON:
{json.dumps(plan_json, indent=2)}

Checker report:
{json.dumps(checker_report, indent=2)}
"""

    explanation = call_text_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_only=False,
    )

    return explanation


# --------------------------------------------------------------------
# Orchestrator for the whole Manufacturing Co Pilot
# --------------------------------------------------------------------


def run_manufacturing_copilot(
    cad_image_url: str,
    solver_hint: Optional[str] = None,
    seed: Optional[int] = None,
    temps: Optional[str] = None,
) -> None:
    """
    End to end entry point that runs the full multi agent workflow.

    Parameters
    ----------
    cad_image_url
        URL of the CAD diagram image for the part.
    solver_hint
        Optional string hint, for example preferred process family.
    seed
        Optional integer identifier for reproducible runs or logging.
    temps
        Optional information about allowable temperatures.

    The function prints intermediate progress messages and ends with a
    human readable explanation from the Interpreter Agent.
    """
    reset_communication_log()
    log_agent_message(
        "Runner",
        {
            "cad_image_url": cad_image_url,
            "solver_hint": solver_hint,
            "seed": seed,
            "temps": temps,
        },
    )

    # 1, Create and populate the shared vector store with general knowledge.
    vector_store = SimpleVectorStore()

    kb_docs = [
        Document(
            text=(
                "General guidelines for milling, minimum radius on internal corners, "
                "tool access rules, and typical surface finish ranges."
            )
        ),
        Document(
            text=(
                "Rules for laser powder bed fusion, overhang limits, "
                "support structure recommendations, and layer thickness choices."
            )
        ),
        Document(
            text=(
                "Design for turning, features that can be produced on a lathe, "
                "typical diametric tolerances, and surface speed rules of thumb."
            )
        ),
    ]
    vector_store.add_documents(kb_docs)

    # Simple library of previously studied manufacturing tasks.
    process_library = [
        {
            "name": "Five axis milling of aluminum block",
            "description": (
                "High speed milling of a prismatic aluminum 6061 block with pockets "
                "and side slots using multi axis tool paths."
            ),
        },
        {
            "name": "Laser powder bed fusion for stainless steel channels",
            "description": (
                "Additive manufacturing of internal cooling channels in 316L stainless "
                "steel with subsequent support removal and heat treatment."
            ),
        },
    ]
    process_docs = [Document(text=item["description"], metadata=item) for item in process_library]

    # 2, CAD Feature Agent using the vision interface.
    print("Step 1, CAD Feature Agent, vision based feature extraction")
    feature_json = cad_feature_agent(
        cad_image_url=cad_image_url,
        solver_hint=solver_hint,
        seed=seed,
        temps=temps,
        vector_store=vector_store,
    )
    log_agent_message("CAD Feature Agent", feature_json)

    # 3, Problem formulation and similarity search.
    print("Step 2, problem formulation and manufacturing feature list")
    feature_list = build_manufacturing_feature_list(
        feature_json=feature_json,
        library_features=process_library,
    )
    log_agent_message("Feature List Builder", feature_list)

    # 4, Manufacturing Agent synthesizes the plan.
    print("Step 3, Manufacturing Agent, process plan synthesis")
    plan_json = manufacturing_agent(
        feature_list=feature_list,
        process_library_docs=process_docs,
        vector_store=vector_store,
    )
    log_agent_message("Manufacturing Agent", plan_json)

    # 5, Checker validates the plan.
    print("Step 4, Manufacturing Process Checker, static analysis")
    has_error, report = manufacturing_process_checker(
        plan_json=plan_json,
        vector_store=vector_store,
    )
    log_agent_message("Process Checker", report)

    if has_error:
        print("Checker found issues. Suggested fixes will be included in the explanation.")
        print(json.dumps(report, indent=2))

    # 6, Interpreter Agent creates a human oriented explanation.
    print("Step 5, Interpreter Agent, final explanation")
    explanation = interpreter_agent(
        plan_json=plan_json,
        checker_report=report,
    )
    log_agent_message("Interpreter Agent", explanation)
    write_final_answer(explanation)

    print("\nFinal explanation for engineers:\n")
    print(explanation)


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------


if __name__ == "__main__":
    # In a real system this would be a URL pointing to a CAD rendering
    # stored in your object storage. For illustration we set a placeholder.
    example_cad_image_url = "/home/vnk3019/manufacturing_agent/cad_images/part_1.png"

    run_manufacturing_copilot(
        cad_image_url=example_cad_image_url,
        solver_hint="Having bending as a part of the manufacturing process.",
        seed=42,
        temps="Maximum allowable bulk temperature around 120 C during any thermal process step.",
    )
