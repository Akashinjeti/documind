import json
from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import HashEmbedder
from app.graph import DocuMindAgent
from app.llm import DemoLLM, parse_json
from app.store import VectorStore

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def make_store() -> VectorStore:
    store = VectorStore(HashEmbedder(), collection="g")
    for p in sorted(DOCS.glob("*.md")):
        store.add(chunk_text(p.read_text(), p.name, max_chars=300, overlap=50))
    return store


class ScriptedLLM:
    """Returns canned replies per role, in order, and records every call."""

    def __init__(self, **replies):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[str] = []

    def complete(self, role, system, user):
        self.calls.append(role)
        queue = self.replies[role]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def test_happy_path_runs_each_agent_once():
    agent = DocuMindAgent(make_store(), DemoLLM())
    out = agent.ask("How long do refunds take to process?")
    assert out["grounded"] is True and out["attempts"] == 0
    assert out["sources"][0]["source"] == "refund-policy.md"
    assert "[1]" in out["answer"]
    assert [t.split(":")[0] for t in out["trace"]] == ["planner", "retriever", "answerer", "critic"]


def test_critic_failure_triggers_rewrite_and_second_retrieval():
    llm = ScriptedLLM(
        planner=[json.dumps({"queries": ["refund timing"]})],
        answerer=["Draft one [1]", "Better answer [1]"],
        critic=[
            json.dumps({"grounded": False, "issues": "missing timeline", "better_query": "refunds processed within business days"}),
            json.dumps({"grounded": True, "issues": "", "better_query": ""}),
        ],
    )
    out = DocuMindAgent(make_store(), llm, max_retries=2).ask("How fast are refunds?")
    assert out["attempts"] == 1
    assert out["grounded"] is True
    assert out["answer"] == "Better answer [1]"
    assert llm.calls == ["planner", "answerer", "critic", "answerer", "critic"]
    assert any(t.startswith("rewriter:") for t in out["trace"])


def test_retries_are_capped():
    llm = ScriptedLLM(
        planner=[json.dumps({"queries": ["x"]})],
        answerer=["a [1]"],
        critic=[json.dumps({"grounded": False, "issues": "bad", "better_query": "refund"})],
    )
    out = DocuMindAgent(make_store(), llm, max_retries=2).ask("Tell me about refunds")
    assert out["attempts"] == 2
    assert out["grounded"] is False
    assert llm.calls.count("critic") == 3


def test_planner_garbage_falls_back_to_original_question():
    llm = ScriptedLLM(planner=["not json at all"], answerer=["ok [1]"],
                      critic=[json.dumps({"grounded": True})])
    out = DocuMindAgent(make_store(), llm).ask("What is the refund window?")
    assert "planner: 1 query" in out["trace"][0]
    assert "What is the refund window?" in out["trace"][0]


def test_empty_store_returns_no_answer_without_calling_answerer_or_critic():
    llm = ScriptedLLM(planner=[json.dumps({"queries": ["q"]})], answerer=["x"], critic=["{}"])
    empty = VectorStore(HashEmbedder(), collection="empty")
    out = DocuMindAgent(empty, llm).ask("Anything at all?")
    assert out["grounded"] is False
    assert "could not find" in out["answer"]
    assert llm.calls == ["planner"]


def test_parse_json_handles_fences_and_noise():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert parse_json("nope") == {}
    assert parse_json("[1,2]") == {}
