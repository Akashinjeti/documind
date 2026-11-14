from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

client = TestClient(create_app(Settings(groq_api_key=None)))

REFUND = "Customers can request a full refund within 30 days of purchase. Refunds are processed within 5 business days."


def test_health_reports_demo_mode():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["mode"] == "demo"


def test_ingest_then_ask():
    r = client.post("/ingest", json={"source": "refunds.md", "text": REFUND})
    assert r.status_code == 200 and r.json()["chunks_added"] >= 1
    a = client.post("/ask", json={"question": "How long do refunds take to process?"})
    body = a.json()
    assert a.status_code == 200
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "refunds.md"
    assert body["trace"][0].startswith("planner")


def test_ingest_file_upload_and_validation():
    ok = client.post("/ingest/file", files={"file": ("notes.txt", b"Backups run nightly at 02:00 UTC.", "text/plain")})
    assert ok.status_code == 200
    bad = client.post("/ingest/file", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 415
    big = client.post("/ingest/file", files={"file": ("big.txt", b"a" * 1_000_001, "text/plain")})
    assert big.status_code == 413
    blank = client.post("/ingest", json={"source": "x", "text": "   "})
    assert blank.status_code == 422


def test_ask_validation():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
