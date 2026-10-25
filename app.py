"""
YouTube Sentiment Analyzer - FastAPI backend.

Run with:
    python app.py
or:
    uvicorn app:app --reload

Endpoints
---------
GET  /                      dashboard page
GET  /api/status            is the model loaded? is a YouTube API key configured?
GET  /api/model-info        training/evaluation statistics of the model
POST /api/analyze           analyse comments of a YouTube video URL
POST /api/demo              analyse the built-in sample comments (no API key needed)
POST /api/analyze-comment   analyse one piece of text
GET  /docs                  automatic interactive API documentation (Swagger UI)
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml.predictor import ModelNotReadyError, SentimentPredictor, get_predictor
from utils import youtube
from utils.insights import build_insights

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")            # reads YOUTUBE_API_KEY from .env if the file exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("yt-sentiment")

SAMPLE_CSV = BASE_DIR / "data" / "sample_comments.csv"
DEFAULT_MAX_COMMENTS = 200
MAX_COMMENTS_LIMIT = 500
MAX_COMMENT_CHARS_DISPLAY = 1000
MAX_SINGLE_TEXT_CHARS = 2000
RATE_LIMIT_REQUESTS = 40          # per client ...
RATE_LIMIT_WINDOW = 60            # ... per minute


def get_api_key() -> str:
    """The key lives only in the server environment. It is never returned to the browser."""
    return os.getenv("YOUTUBE_API_KEY", "").strip()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class APIError(Exception):
    """An error with a friendly message that is safe to show in the UI."""

    def __init__(self, message: str, status_code: int = 400, code: str = "error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def error_response(message: str, status_code: int, code: str, headers: dict = None) -> JSONResponse:
    return JSONResponse({"error": message, "code": code}, status_code=status_code, headers=headers)


# --------------------------------------------------------------------------- #
# Very small in-memory rate limiter (protects the server and the YouTube quota)
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit, self.window = limit, window
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int:
        """Return 0 if allowed, otherwise the number of seconds to wait."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, int(self.window - (now - hits[0])))
            hits.append(now)
            if len(self._hits) > 5000:                 # keep memory bounded
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return 0


rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    wait = rate_limiter.check(client)
    if wait:
        raise APIError(f"Too many requests. Please wait {wait} seconds and try again.",
                       429, "rate_limited")


# --------------------------------------------------------------------------- #
# Request models (validation)
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    url: str = Field(default="", max_length=500)
    max_comments: int = Field(default=DEFAULT_MAX_COMMENTS, ge=10, le=MAX_COMMENTS_LIMIT)
    allow_fallback: bool = True

    @field_validator("url")
    @classmethod
    def strip_url(cls, value: str) -> str:
        return value.strip()


class CommentRequest(BaseModel):
    text: str = Field(default="", max_length=MAX_SINGLE_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


# --------------------------------------------------------------------------- #
# Application set-up
# --------------------------------------------------------------------------- #
state = {"predictor": None, "model_error": None, "demo_comments": []}


def load_demo_comments() -> list:
    """Load data/sample_comments.csv into a list of dicts."""
    try:
        df = pd.read_csv(SAMPLE_CSV)
        df = df.dropna(subset=["comment"])
        df["comment"] = df["comment"].astype(str).str.strip()
        df = df[df["comment"] != ""]
        return [
            {"text": row.comment, "author": getattr(row, "author", None),
             "likes": int(getattr(row, "likes", 0) or 0)}
            for row in df.itertuples()
        ]
    except Exception as exc:
        logger.error("Could not load sample comments: %s", exc)
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = get_predictor()
        logger.info("Model ready (accuracy %.3f)", state["predictor"].metrics.get("accuracy", 0))
    except ModelNotReadyError as exc:
        state["model_error"] = str(exc)
        logger.error("Model not ready: %s", exc)
    state["demo_comments"] = load_demo_comments()
    logger.info("Loaded %d demo comments", len(state["demo_comments"]))
    if get_api_key():
        logger.info("YouTube API key found - live mode available")
    else:
        logger.info("No YOUTUBE_API_KEY set - Demo Mode will be used")
    yield


app = FastAPI(title="YouTube Sentiment Analyzer", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: https://i.ytimg.com https://*.ytimg.com; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'",
        )
    return response


# ---- exception handlers: always JSON, never a Python traceback ------------- #
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    headers = {"Retry-After": "30"} if exc.status_code == 429 else None
    return error_response(exc.message, exc.status_code, exc.code, headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = {str(err["loc"][-1]) for err in exc.errors() if err.get("loc")}
    if "text" in fields:
        message = f"Please enter text of at most {MAX_SINGLE_TEXT_CHARS} characters."
    elif "max_comments" in fields:
        message = f"max_comments must be between 10 and {MAX_COMMENTS_LIMIT}."
    elif "url" in fields:
        message = "That URL is too long to be a valid YouTube link."
    else:
        message = "The request was not valid. Please check your input and try again."
    return error_response(message, 422, "validation_error")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    messages = {404: "That page or endpoint does not exist.", 405: "That method is not allowed here."}
    return error_response(messages.get(exc.status_code, "Request failed."), exc.status_code, "http_error")


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)       # full detail only in the server log
    return error_response("Something went wrong on the server. Please try again.", 500, "server_error")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def require_model() -> SentimentPredictor:
    predictor = state["predictor"]
    if predictor is None:
        raise APIError(state["model_error"] or "The sentiment model is not loaded. "
                       "Run 'python ml/train.py' and restart the server.", 503, "model_unavailable")
    return predictor


def overall_sentiment(counts: dict, total: int) -> tuple:
    """
    Decide the overall sentiment of a video.

    net score = (positive - negative) / total
      >= +0.15            -> Positive
      <= -0.15            -> Negative
      otherwise           -> Neutral if neutral comments are the largest group, else Mixed
    """
    net = (counts["Positive"] - counts["Negative"]) / total
    if net >= 0.15:
        return "Positive", net
    if net <= -0.15:
        return "Negative", net
    if counts["Neutral"] >= max(counts["Positive"], counts["Negative"]):
        return "Neutral", net
    return "Mixed", net


def analyze_comments(raw_comments: list, mode: str, video: dict, notice: str = None, fallback: bool = False) -> dict:
    """Run the model on a list of comment dicts and build the dashboard payload."""
    predictor = require_model()
    texts = [c["text"] for c in raw_comments]
    predictions = predictor.predict(texts)

    comments, counts, confidence_sum = [], {"Positive": 0, "Negative": 0, "Neutral": 0}, 0.0
    for index, (comment, pred) in enumerate(zip(raw_comments, predictions), start=1):
        counts[pred["sentiment"]] += 1
        confidence_sum += pred["confidence"]
        text = comment["text"]
        comments.append({
            "id": index,
            "text": text[:MAX_COMMENT_CHARS_DISPLAY],
            "sentiment": pred["sentiment"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
        })

    total = len(comments)
    overall, net = overall_sentiment(counts, total)
    summary = {
        "total": total,
        "counts": counts,
        "percentages": {k: round(v * 100 / total, 1) for k, v in counts.items()},
        "overall": overall,
        "net_score": round(net, 4),
        "average_confidence": round(confidence_sum / total, 4),
    }
    insights = build_insights(texts, [c["sentiment"] for c in comments])
    return {
        "mode": mode,                 # "live" (YouTube API) or "demo" (sample data)
        "fallback": fallback,         # True when demo data replaced a failed live request
        "notice": notice,
        "video": video,
        "summary": summary,
        "comments": comments,
        "insights": insights,
    }


DEMO_VIDEO = {
    "video_id": None,
    "title": "Demo video: Machine Learning Explained for Beginners",
    "channel": "Sample dataset (data/sample_comments.csv)",
    "thumbnail": None,
    "comment_count": None,
}


def run_demo(notice: str = None, fallback: bool = False) -> dict:
    comments = state["demo_comments"]
    if not comments:
        raise APIError("The demo dataset could not be loaded. Check data/sample_comments.csv.",
                       500, "demo_unavailable")
    return analyze_comments(comments, "demo", DEMO_VIDEO, notice, fallback)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"example_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})


@app.get("/api/status")
def status():
    predictor = state["predictor"]
    return {
        "model_loaded": predictor is not None,
        "model_error": state["model_error"],
        "youtube_api_configured": bool(get_api_key()),   # only a yes/no - never the key itself
        "demo_comments": len(state["demo_comments"]),
        "limits": {"max_comments": MAX_COMMENTS_LIMIT, "default_comments": DEFAULT_MAX_COMMENTS,
                   "max_text_chars": MAX_SINGLE_TEXT_CHARS},
    }


@app.get("/api/model-info")
def model_info():
    info = require_model().info()
    info["youtube_api_configured"] = bool(get_api_key())
    return info


@app.post("/api/analyze", dependencies=[Depends(rate_limit)])
def analyze(body: AnalyzeRequest):
    # 1. validate the URL first - an invalid link is an error even in demo mode
    try:
        video_id = youtube.extract_video_id(body.url)
    except youtube.InvalidURLError as exc:
        raise APIError(exc.message, 400, exc.code)

    require_model()
    api_key = get_api_key()

    def fall_back(reason: str):
        if not body.allow_fallback:
            raise APIError(reason, 503, "api_unavailable")
        return run_demo(notice=f"{reason} Showing the demo dataset instead.", fallback=True)

    # 2. no API key -> automatic demo mode
    if not api_key:
        return fall_back("No YouTube API key is configured on the server.")

    # 3. real YouTube API mode
    try:
        video = youtube.fetch_video_info(video_id, api_key)
        raw_comments = youtube.fetch_comments(video_id, api_key, body.max_comments)
    except (youtube.QuotaExceededError, youtube.InvalidAPIKeyError, youtube.YouTubeAPIError,
            youtube.MissingAPIKeyError) as exc:
        logger.warning("YouTube API unavailable (%s)", exc.code)
        return fall_back(exc.message)
    except youtube.VideoNotFoundError as exc:
        raise APIError(exc.message, 404, exc.code)
    except youtube.CommentsDisabledError as exc:
        raise APIError(exc.message, 422, exc.code)
    except youtube.NoCommentsError as exc:
        raise APIError(exc.message, 404, exc.code)
    except youtube.YouTubeError as exc:
        raise APIError(exc.message, 502, exc.code)

    return analyze_comments(raw_comments, "live", video)


@app.post("/api/demo", dependencies=[Depends(rate_limit)])
def demo():
    require_model()
    return run_demo()


@app.post("/api/analyze-comment", dependencies=[Depends(rate_limit)])
def analyze_comment(body: CommentRequest):
    if not body.text:
        raise APIError("Please type a comment to analyze.", 400, "empty_comment")
    result = require_model().analyze_single(body.text)
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"\n  YouTube Sentiment Analyzer  ->  http://127.0.0.1:{port}\n")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
