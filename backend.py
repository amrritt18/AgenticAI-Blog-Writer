from __future__ import annotations

import operator
import os
import re
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver


# ---------------------------------------------------------------------------
# Environment and configuration
# ---------------------------------------------------------------------------
load_dotenv()

# LangChain's Google integration commonly reads GOOGLE_API_KEY. Support the
# project's more explicit GEMINI_API_KEY name as an alias.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Add your PostgreSQL connection URL to .env."
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


# ---------------------------------------------------------------------------
# Multi-provider LLM fallback
# ---------------------------------------------------------------------------
class StructuredFallback:
    """Try structured output with each provider until one succeeds."""

    def __init__(self, models: list[Any], schema: Any, kwargs: dict[str, Any]):
        self.models = models
        self.schema = schema
        self.kwargs = kwargs

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        failures: list[str] = []
        for name, model in self.models:
            try:
                structured_model = model.with_structured_output(
                    self.schema, **self.kwargs
                )
                return structured_model.invoke(messages, **kwargs)
            except Exception as exc:  # fallback also covers unsupported schemas
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[LLM fallback] {name} structured output failed; trying next provider.")
        raise RuntimeError(
            "All configured LLM providers failed for structured output. "
            + " | ".join(failures)
        )


class FallbackLLM:
    """Small invoke-compatible wrapper that preserves the existing node code."""

    def __init__(self, models: list[tuple[str, Any]]):
        self.models = models

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        failures: list[str] = []
        for name, model in self.models:
            try:
                return model.invoke(messages, **kwargs)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print(f"[LLM fallback] {name} failed; trying next provider.")
        raise RuntimeError(
            "All configured LLM providers failed. " + " | ".join(failures)
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredFallback:
        return StructuredFallback(self.models, schema, kwargs)


def build_llm() -> FallbackLLM:
    """Build providers in priority order. Missing keys disable that provider."""
    models: list[tuple[str, Any]] = []

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        models.append(
            (
                "Groq",
                ChatGroq(
                    api_key=groq_key,
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                ),
            )
        )

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        models.append(
            (
                "Gemini",
                ChatGoogleGenerativeAI(
                    google_api_key=gemini_key,
                    model="gemini-2.5-flash",
                    temperature=0,
                ),
            )
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        models.append(
            (
                "OpenRouter",
                ChatOpenRouter(
                    api_key=openrouter_key,
                    model="openrouter/free",
                    temperature=0,
                ),
            )
        )

    if not models:
        raise ValueError(
            "No LLM provider is configured. Set at least one of GROQ_API_KEY, "
            "GEMINI_API_KEY (or GOOGLE_API_KEY), or OPENROUTER_API_KEY in .env."
        )

    configured = ", ".join(name for name, _ in models)
    print(f"[LLM] Configured provider order: {configured}")
    return FallbackLLM(models)


llm = build_llm()


# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ..., description="One sentence describing what the reader should learn."
    )
    bullets: List[str] = Field(
        ..., min_length=3, max_length=6,
        description="3–6 concrete, non-overlapping subpoints for this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict, total=False):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    final: str


# ---------------------------------------------------------------------------
# 2. Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts and fundamentals.
- hybrid (needs_research=true): evergreen topics that benefit from current examples.
- open_book (needs_research=true): volatile topics such as latest news, pricing,
  rankings, policies, or weekly roundups.

If research is needed, output 3–10 specific, high-signal search queries. If the
user asks for latest/this week/last week, include that time constraint in queries.
"""


def router_node(state: State) -> dict:
    decision = llm.with_structured_output(RouterDecision).invoke(
        [SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=f"Topic: {state['topic']}")]
    )
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


# ---------------------------------------------------------------------------
# 3. Research (Tavily)
# ---------------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    normalized: List[dict] = []
    for result in results or []:
        normalized.append(
            {
                "title": result.get("title") or "",
                "url": result.get("url") or "",
                "snippet": result.get("content") or result.get("snippet") or "",
                "published_at": result.get("published_date") or result.get("published_at"),
                "source": result.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Produce a deduplicated list of EvidenceItem objects from the supplied search
results. Include only items with a non-empty URL. Prefer relevant, authoritative
sources. Preserve explicit publication dates as YYYY-MM-DD; never guess dates.
Keep snippets short and deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw_results: List[dict] = []
    for query in queries:
        try:
            raw_results.extend(_tavily_search(query, max_results=6))
        except Exception as exc:
            print(f"[Research] Search failed for one query: {type(exc).__name__}: {exc}")

    if not raw_results:
        return {"evidence": []}

    pack = llm.with_structured_output(EvidencePack).invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )
    dedup: dict[str, EvidenceItem] = {}
    for item in pack.evidence:
        if item.url:
            dedup[item.url] = item
    return {"evidence": list(dedup.values())}


# ---------------------------------------------------------------------------
# 4. Orchestrator (planning)
# ---------------------------------------------------------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Create an actionable technical blog outline with 5–9 sections.
Each task must have a one-sentence goal, 3–6 specific non-overlapping bullets,
and a target word count between 120 and 550.

Quality requirements: include at least two of code example, edge cases/failure
modes, performance/cost, security/privacy (if relevant), or debugging/observability.

Grounding:
- closed_book: evergreen content; do not depend on evidence.
- hybrid: use evidence for current examples; mark relevant sections as requiring
  research and citations.
- open_book: blog_kind must be news_roundup. Focus on events and implications,
  not tutorials, unless explicitly requested. If evidence is insufficient, say so
  transparently and only plan claims that can be supported.
Return data matching the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get("evidence", []) or []
    mode = state.get("mode", "closed_book")
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\nMode: {mode}\n\n"
                    f"Evidence (use only for fresh claims; may be empty):\n"
                    f"{[item.model_dump() for item in evidence][:16]}"
                )
            ),
        ]
    )
    return {"plan": plan}


# ---------------------------------------------------------------------------
# 5. Fanout and worker
# ---------------------------------------------------------------------------
def fanout(state: State):
    plan = state.get("plan")
    if plan is None:
        raise ValueError("The orchestrator did not return a plan.")
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": plan.model_dump(),
                "evidence": [item.model_dump() for item in (state.get("evidence") or [])],
            },
        )
        for task in plan.tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

- Follow the goal and cover every bullet in order.
- Stay within ±15% of the target word count.
- Output only the section content, with a '## <Section Title>' heading.
- For news_roundup, summarize events and implications; do not turn it into a
  tutorial unless the requested bullets explicitly require one.
- For open_book claims, use only the supplied evidence URLs and cite event claims
  with Markdown links. If unsupported, say: 'Not found in provided sources.'
- If requires_citations is true, cite outside-world claims using supplied URLs.
- If requires_code is true, include at least one minimal, correct code snippet.
- Use short paragraphs, helpful bullets, and code fences where appropriate.
"""


def _message_text(content: Any) -> str:
    """Normalize provider output to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, str):
                pieces.append(part)
        return "\n".join(pieces).strip()
    return str(content).strip()


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {item.title} | {item.url} | {item.published_at or 'date:unknown'}"
        for item in evidence[:20]
    )

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\nTone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\nConstraints: {plan.constraints}\n"
                    f"Topic: {topic}\nMode: {mode}\n\n"
                    f"Section title: {task.title}\nGoal: {task.goal}\n"
                    f"Target words: {task.target_words}\nTags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (only use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    )
    section_md = _message_text(response.content)
    return {"sections": [(task.id, section_md)]}


# ---------------------------------------------------------------------------
# 6. Reducer: merge sections and save Markdown (no image generation)
# ---------------------------------------------------------------------------
def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip().rstrip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned or "generated_blog")[:150] + ".md"


def merge_content(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Cannot merge blog sections because plan is missing.")

    ordered_sections = [markdown for _, markdown in sorted(state.get("sections", []), key=lambda pair: pair[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    output_path = Path(_safe_filename(plan.blog_title))
    output_path.write_text(merged_md, encoding="utf-8")
    print(f"[Blog] Saved Markdown blog to: {output_path.resolve()}")
    return {"merged_md": merged_md, "final": merged_md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", END)
reducer_subgraph = reducer_graph.compile()


# ---------------------------------------------------------------------------
# 7. Main graph
# ---------------------------------------------------------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next, {"research": "research", "orchestrator": "orchestrator"}
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)


# ---------------------------------------------------------------------------
# 8. PostgreSQL checkpointer and compiled graph
# ---------------------------------------------------------------------------
DATABASE_URL = get_database_url()
_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()
app = g.compile(checkpointer=checkpointer)


# Example invocation (provide a unique thread_id for each conversation/run):
# result = app.invoke(
#     {
#         "topic": "How vector databases support retrieval-augmented generation",
#         "sections": [],
#     },
#     config={"configurable": {"thread_id": "blog-run-001"}},
# )
# print(result["final"])
