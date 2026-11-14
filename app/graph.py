"""Multi-agent RAG pipeline built with LangGraph.

    START -> planner -> retriever -> answerer -> critic --grounded--> END
                           ^                        |
                           +------ rewriter <-------+  (not grounded, retries left)

Agents:
  planner   splits the question into 1-3 focused search queries
  retriever hybrid of the planner's queries against Qdrant, de-duplicated by chunk id
  answerer  writes a cited answer from the retrieved passages only
  critic    checks the answer is supported by the context; proposes a better query if not
  rewriter  turns the critic's feedback into a new query and loops back to retrieval
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLM, parse_json
from .store import VectorStore

PLANNER_SYSTEM = (
    "You are the Planner agent of a document Q&A system. Break the user's question into "
    "1 to 3 short, self-contained search queries. "
    'Reply ONLY with JSON: {"queries": ["..."]}'
)
ANSWERER_SYSTEM = (
    "You are the Answerer agent. Answer using ONLY the numbered context passages. "
    "Cite the passages you use like [1]. If the context does not contain the answer, "
    "say you could not find it in the documents."
)
CRITIC_SYSTEM = (
    "You are the Critic agent. Decide whether every claim in the answer is supported by "
    "the context. "
    'Reply ONLY with JSON: {"grounded": true|false, "issues": "...", "better_query": "..."} '
    "where better_query is a search query likely to find the missing evidence (or empty)."
)
NO_ANSWER = "I could not find this in the ingested documents."


class AgentState(TypedDict, total=False):
    question: str
    queries: list[str]
    hits: list[dict]
    answer: str
    grounded: bool
    issues: str
    better_query: str
    attempts: int
    trace: list[str]


def _context_block(hits: list[dict]) -> str:
    return "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, start=1))


def build_graph(store: VectorStore, llm: LLM, k: int = 4, max_retries: int = 1):
    def planner(state: AgentState) -> AgentState:
        question = state["question"]
        data = parse_json(llm.complete("planner", PLANNER_SYSTEM, f"Question: {question}"))
        raw = data.get("queries")
        queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:3] if isinstance(raw, list) else []
        queries = queries or [question]
        return {
            "queries": queries,
            "attempts": 0,
            "trace": [f"planner: {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} -> {queries}"],
        }

    def retriever(state: AgentState) -> AgentState:
        best: dict[str, dict] = {}
        for q in state["queries"]:
            for hit in store.search(q, k=k):
                d = hit.as_dict()
                if d["id"] not in best or d["score"] > best[d["id"]]["score"]:
                    best[d["id"]] = d
        cap = min(8, k * len(state["queries"]))
        hits = sorted(best.values(), key=lambda d: d["score"], reverse=True)[:cap]
        return {"hits": hits, "trace": state["trace"] + [f"retriever: {len(hits)} passages"]}

    def answerer(state: AgentState) -> AgentState:
        hits = state["hits"]
        if not hits:
            return {"answer": NO_ANSWER, "trace": state["trace"] + ["answerer: no context"]}
        user = f"Question: {state['question']}\n\nContext:\n{_context_block(hits)}"
        answer = llm.complete("answerer", ANSWERER_SYSTEM, user).strip()
        return {"answer": answer, "trace": state["trace"] + ["answerer: drafted answer"]}

    def critic(state: AgentState) -> AgentState:
        if not state["hits"]:
            return {"grounded": False, "issues": "no context retrieved", "better_query": "",
                    "trace": state["trace"] + ["critic: skipped (no context)"]}
        user = (
            f"Question: {state['question']}\n\nContext:\n{_context_block(state['hits'])}\n\n"
            f"Answer: {state['answer']}"
        )
        data = parse_json(llm.complete("critic", CRITIC_SYSTEM, user))
        # Fail open on unparseable critic output: do not loop on a broken verdict.
        grounded = bool(data.get("grounded", True))
        issues = str(data.get("issues", "") or "")
        better = str(data.get("better_query", "") or "")
        verdict = "grounded" if grounded else f"not grounded ({issues or 'unspecified'})"
        return {"grounded": grounded, "issues": issues, "better_query": better,
                "trace": state["trace"] + [f"critic: {verdict}"]}

    def rewriter(state: AgentState) -> AgentState:
        query = state.get("better_query") or f"{state['question']} {state.get('issues', '')}".strip()
        return {
            "queries": [query],
            "attempts": state["attempts"] + 1,
            "trace": state["trace"] + [f"rewriter: retry {state['attempts'] + 1} with '{query}'"],
        }

    def route(state: AgentState) -> str:
        if state["grounded"] or not state["hits"] or state["attempts"] >= max_retries:
            return "done"
        return "retry"

    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("retriever", retriever)
    g.add_node("answerer", answerer)
    g.add_node("critic", critic)
    g.add_node("rewriter", rewriter)
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "answerer")
    g.add_edge("answerer", "critic")
    g.add_conditional_edges("critic", route, {"done": END, "retry": "rewriter"})
    g.add_edge("rewriter", "retriever")
    return g.compile()


class DocuMindAgent:
    def __init__(self, store: VectorStore, llm: LLM, k: int = 4, max_retries: int = 1) -> None:
        self._graph = build_graph(store, llm, k=k, max_retries=max_retries)

    def ask(self, question: str) -> dict:
        out = self._graph.invoke({"question": question})
        return {
            "answer": out["answer"],
            "grounded": out["grounded"],
            "attempts": out["attempts"],
            "sources": out["hits"],
            "trace": out["trace"],
        }
