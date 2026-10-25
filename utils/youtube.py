"""
YouTube Data API v3 helper.

* extract_video_id()   - understands every common YouTube URL format
* fetch_video_info()   - title / channel / thumbnail (optional, never fatal)
* fetch_comments()     - top-level comments with pagination

The API key is read from the environment (YOUTUBE_API_KEY) on the SERVER only.
It is never sent to the browser, and it is never included in error messages.
"""

import re
from urllib.parse import parse_qs, urlparse

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
REQUEST_TIMEOUT = 15          # seconds
PAGE_SIZE = 100               # maximum allowed by the API for commentThreads
HARD_MAX_COMMENTS = 500       # protect the API quota

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_PATH_PREFIXES = ("embed", "shorts", "live", "v", "e")


# --------------------------------------------------------------------------- #
# Exceptions (each one has a friendly message that is safe to show to users)
# --------------------------------------------------------------------------- #
class YouTubeError(Exception):
    """Base class. `message` is always safe to display in the UI."""
    code = "youtube_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidURLError(YouTubeError):
    code = "invalid_url"


class MissingAPIKeyError(YouTubeError):
    code = "missing_api_key"


class InvalidAPIKeyError(YouTubeError):
    code = "invalid_api_key"


class QuotaExceededError(YouTubeError):
    code = "quota_exceeded"


class VideoNotFoundError(YouTubeError):
    code = "video_not_found"


class CommentsDisabledError(YouTubeError):
    code = "comments_disabled"


class NoCommentsError(YouTubeError):
    code = "no_comments"


class YouTubeAPIError(YouTubeError):
    """Network problems or unexpected API failures."""
    code = "api_unavailable"


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #
def extract_video_id(url: str) -> str:
    """
    Return the 11-character video ID from a YouTube URL (or a bare video ID).

    Supported: watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID, /live/ID, m.youtube.com, bare ID.
    Raises InvalidURLError for anything else.
    """
    if not url or not str(url).strip():
        raise InvalidURLError("Please enter a YouTube video URL.")

    raw = str(url).strip()
    if len(raw) > 500:
        raise InvalidURLError("That URL is too long to be a valid YouTube link.")

    if _VIDEO_ID_RE.match(raw):                    # user pasted only the ID
        return raw

    if "://" not in raw:
        raw = "https://" + raw                      # allow "youtube.com/watch?v=..."
    try:
        parsed = urlparse(raw)
    except ValueError:
        raise InvalidURLError("That doesn't look like a valid URL.") from None

    host = (parsed.hostname or "").lower()
    candidate = None

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path.rstrip("/") == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
                candidate = parts[1]
    else:
        raise InvalidURLError("That is not a YouTube link. Use a URL like "
                              "https://www.youtube.com/watch?v=VIDEO_ID")

    if not candidate or not _VIDEO_ID_RE.match(candidate):
        raise InvalidURLError("Could not find a valid video ID in that URL. "
                              "Use a link like https://www.youtube.com/watch?v=VIDEO_ID")
    return candidate


# --------------------------------------------------------------------------- #
# API calls
# --------------------------------------------------------------------------- #
def _get(endpoint: str, params: dict, api_key: str) -> dict:
    """GET request with consistent, key-safe error handling."""
    if not api_key:
        raise MissingAPIKeyError("No YouTube API key is configured on the server.")

    try:
        response = requests.get(
            f"{API_BASE}/{endpoint}",
            params={**params, "key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise YouTubeAPIError("The YouTube API took too long to respond.") from None
    except requests.exceptions.RequestException:
        # NOTE: do not include the exception text - it may contain the request URL (and the key).
        raise YouTubeAPIError("Could not reach the YouTube API. Check the internet connection.") from None

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise YouTubeAPIError("The YouTube API returned an unreadable response.") from None

    _raise_for_error(response)


def _raise_for_error(response) -> None:
    """Translate a YouTube API error response into one of our exceptions."""
    reason, api_message = "", ""
    try:
        error = response.json().get("error", {})
        api_message = str(error.get("message", ""))
        errors = error.get("errors") or []
        reason = errors[0].get("reason", "") if errors else ""
    except (ValueError, AttributeError, IndexError):
        pass

    status = response.status_code
    if reason in ("commentsDisabled",):
        raise CommentsDisabledError("Comments are turned off for this video. Try another video or use Demo Mode.")
    if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"):
        raise QuotaExceededError("The YouTube API quota has been used up for today.")
    if reason in ("videoNotFound", "notFound") or status == 404:
        raise VideoNotFoundError("That video was not found. It may be private, deleted or the link is wrong.")
    if reason in ("keyInvalid", "badRequest") and "API key" in api_message:
        raise InvalidAPIKeyError("The configured YouTube API key was rejected by Google.")
    if status in (400, 401) and "API key" in api_message:
        raise InvalidAPIKeyError("The configured YouTube API key was rejected by Google.")
    if reason == "forbidden":
        raise CommentsDisabledError("The comments of this video cannot be accessed. They may be "
                                    "disabled or restricted. Try another video or use Demo Mode.")
    if reason in ("accessNotConfigured", "ipRefererBlocked", "referrerNotAllowed") or status == 403:
        raise InvalidAPIKeyError("The YouTube API key is not allowed to use the YouTube Data API v3. "
                                 "Enable that API for the key in Google Cloud Console.")
    raise YouTubeAPIError("The YouTube API returned an unexpected error. Please try again later.")


def fetch_video_info(video_id: str, api_key: str) -> dict:
    """Return basic video details. Never raises: returns a minimal dict if anything fails."""
    fallback = {"video_id": video_id, "title": f"YouTube video {video_id}", "channel": None,
                "thumbnail": None, "comment_count": None}
    try:
        data = _get("videos", {"part": "snippet,statistics", "id": video_id}, api_key)
    except YouTubeError:
        return fallback
    items = data.get("items") or []
    if not items:
        return fallback
    snippet = items[0].get("snippet", {})
    stats = items[0].get("statistics", {})
    thumbs = snippet.get("thumbnails", {})
    thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
    return {
        "video_id": video_id,
        "title": snippet.get("title") or fallback["title"],
        "channel": snippet.get("channelTitle"),
        "thumbnail": thumb,
        "comment_count": int(stats["commentCount"]) if str(stats.get("commentCount", "")).isdigit() else None,
    }


def fetch_comments(video_id: str, api_key: str, max_comments: int = 200) -> list:
    """
    Fetch up to `max_comments` top-level comments (newest pages are followed with pageToken).

    Returns a list of dicts: {"text", "author", "likes", "published"}.
    Raises a YouTubeError subclass on failure.
    """
    max_comments = max(1, min(int(max_comments), HARD_MAX_COMMENTS))
    comments, page_token = [], None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(PAGE_SIZE, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        data = _get("commentThreads", params, api_key)

        for item in data.get("items", []):
            top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = (top.get("textDisplay") or "").strip()
            if text:
                comments.append({
                    "text": text,
                    "author": top.get("authorDisplayName"),
                    "likes": top.get("likeCount", 0),
                    "published": top.get("publishedAt"),
                })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    if not comments:
        raise NoCommentsError("No comments were found for this video.")
    return comments[:max_comments]
