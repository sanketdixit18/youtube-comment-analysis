"""
Automated tests. Run from the project root with:

    pip install pytest httpx
    pytest -q

They use FastAPI's TestClient, so no server has to be running, and the YouTube
API is replaced by fake responses, so no internet or API key is needed.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as app_module                      # noqa: E402
from ml.preprocess import clean_text          # noqa: E402
from utils import youtube                     # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:      # "with" runs the start-up (model loading)
        yield c


@pytest.fixture(autouse=True)
def reset_rate_limit():
    app_module.rate_limiter._hits.clear()
    yield


# ----------------------------- preprocessing ----------------------------- #
def test_clean_text_basics():
    assert clean_text("I DIDN'T like it!!! https://x.co") == "not like exclamationmark"
    assert "http" not in clean_text("see https://youtu.be/abc now")


def test_clean_text_keeps_negation_and_converts_emoji():
    assert "not" in clean_text("not good").split()
    assert "emojipositive" in clean_text("great 😍").split()


# ----------------------------- video id parsing -------------------------- #
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=10",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=abc",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "youtube.com/watch?v=dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
])
def test_extract_video_id_valid(url):
    assert youtube.extract_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", ["", "   ", "hello", "https://example.com/watch?v=dQw4w9WgXcQ",
                                 "https://www.youtube.com/watch?v=short", "https://www.youtube.com/"])
def test_extract_video_id_invalid(url):
    with pytest.raises(youtube.InvalidURLError):
        youtube.extract_video_id(url)


# ----------------------------- API: basic -------------------------------- #
def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "YouTube Sentiment Analyzer" in r.text
    assert r.headers["x-content-type-options"] == "nosniff"


def test_status_never_leaks_key(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "SECRET-KEY-123")
    body = client.get("/api/status").json()
    assert body["youtube_api_configured"] is True
    assert "SECRET-KEY-123" not in str(body)


def test_model_info_has_real_metrics(client):
    info = client.get("/api/model-info").json()
    assert info["train_samples"] + info["test_samples"] == info["total_samples"]
    assert 0 < info["accuracy"] <= 1
    matrix = info["confusion_matrix"]["matrix"]
    assert sum(map(sum, matrix)) == info["test_samples"]


def test_demo(client):
    body = client.post("/api/demo").json()
    s = body["summary"]
    assert body["mode"] == "demo" and body["fallback"] is False
    assert s["total"] == sum(s["counts"].values()) == len(body["comments"])
    assert abs(sum(s["percentages"].values()) - 100) < 0.5
    assert all(0 < c["confidence"] <= 1 for c in body["comments"])
    assert body["insights"]["overall_words"]


# ----------------------------- API: single comment ----------------------- #
def test_single_comment_positive_and_negative(client):
    pos = client.post("/api/analyze-comment", json={"text": "I really enjoyed this video. The explanation was excellent!"}).json()
    neg = client.post("/api/analyze-comment", json={"text": "Waste of time, terrible and boring."}).json()
    assert pos["sentiment"] == "Positive"
    assert neg["sentiment"] == "Negative"
    assert abs(sum(pos["probabilities"].values()) - 1) < 0.01
    assert pos["explanation"]


def test_single_comment_errors(client):
    assert client.post("/api/analyze-comment", json={"text": "   "}).status_code == 400
    assert client.post("/api/analyze-comment", json={}).status_code == 400
    r = client.post("/api/analyze-comment", json={"text": "x" * 5000})
    assert r.status_code == 422 and "error" in r.json()


def test_html_in_comment_is_returned_as_plain_data(client):
    r = client.post("/api/analyze-comment", json={"text": "<script>alert(1)</script> great video"})
    assert r.status_code == 200          # the front-end renders it with textContent


# ----------------------------- API: analyze ------------------------------ #
def test_analyze_validation(client):
    assert client.post("/api/analyze", json={"url": ""}).status_code == 400
    r = client.post("/api/analyze", json={"url": "https://example.com/x"})
    assert r.status_code == 400 and r.json()["code"] == "invalid_url"
    assert client.post("/api/analyze", json={"url": "dQw4w9WgXcQ", "max_comments": 99999}).status_code == 422


def test_analyze_without_key_falls_back_to_demo(client, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    body = client.post("/api/analyze", json={"url": "https://youtu.be/dQw4w9WgXcQ"}).json()
    assert body["mode"] == "demo" and body["fallback"] is True and "API key" in body["notice"]


def test_analyze_without_key_and_no_fallback_is_error(client, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    r = client.post("/api/analyze", json={"url": "dQw4w9WgXcQ", "allow_fallback": False})
    assert r.status_code == 503


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def comment_item(text):
    return {"snippet": {"topLevelComment": {"snippet": {
        "textDisplay": text, "authorDisplayName": "a", "likeCount": 1, "publishedAt": "2024-01-01T00:00:00Z"}}}}


def test_live_mode_with_pagination(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        if url.endswith("/videos"):
            return FakeResponse(200, {"items": [{"snippet": {"title": "My Video", "channelTitle": "Chan",
                                                             "thumbnails": {"medium": {"url": "https://i.ytimg.com/x.jpg"}}},
                                                 "statistics": {"commentCount": "42"}}]})
        if "pageToken" not in params:
            return FakeResponse(200, {"items": [comment_item("Great video, loved it") for _ in range(100)],
                                      "nextPageToken": "NEXT"})
        return FakeResponse(200, {"items": [comment_item("This is terrible and boring") for _ in range(50)]})

    monkeypatch.setattr(youtube.requests, "get", fake_get)
    body = client.post("/api/analyze", json={"url": "dQw4w9WgXcQ", "max_comments": 200}).json()
    assert body["mode"] == "live" and body["fallback"] is False
    assert body["summary"]["total"] == 150
    assert body["video"]["title"] == "My Video"
    assert any(p.get("pageToken") == "NEXT" for p in calls)       # pagination was followed


@pytest.mark.parametrize("status, reason, expected_status, expected_code", [
    (403, "commentsDisabled", 422, "comments_disabled"),
    (404, "videoNotFound", 404, "video_not_found"),
])
def test_live_mode_video_errors(client, monkeypatch, status, reason, expected_status, expected_code):
    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
    err = {"error": {"message": "x", "errors": [{"reason": reason}]}}
    monkeypatch.setattr(youtube.requests, "get", lambda *a, **k: FakeResponse(status, err))
    r = client.post("/api/analyze", json={"url": "dQw4w9WgXcQ"})
    assert r.status_code == expected_status and r.json()["code"] == expected_code


def test_live_mode_quota_falls_back_to_demo(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
    err = {"error": {"message": "quota", "errors": [{"reason": "quotaExceeded"}]}}
    monkeypatch.setattr(youtube.requests, "get", lambda *a, **k: FakeResponse(403, err))
    body = client.post("/api/analyze", json={"url": "dQw4w9WgXcQ"}).json()
    assert body["mode"] == "demo" and body["fallback"] is True and "quota" in body["notice"].lower()


def test_network_error_does_not_leak_key(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "TOPSECRET")

    def boom(*a, **k):
        raise youtube.requests.exceptions.ConnectionError("failed for url ...key=TOPSECRET")

    monkeypatch.setattr(youtube.requests, "get", boom)
    body = client.post("/api/analyze", json={"url": "dQw4w9WgXcQ"}).json()
    assert "TOPSECRET" not in str(body)
    assert body["fallback"] is True


def test_unknown_route_is_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.json()


def test_rate_limit(client):
    codes = [client.post("/api/analyze-comment", json={"text": "hello"}).status_code
             for _ in range(app_module.RATE_LIMIT_REQUESTS + 3)]
    assert codes[-1] == 429
