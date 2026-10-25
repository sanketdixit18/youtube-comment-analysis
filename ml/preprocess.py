"""
Text preprocessing for the YouTube Sentiment Analyzer.

The same functions are used for training AND for prediction, so the model always
sees text that has been cleaned in exactly the same way.

Cleaning steps
--------------
1. Convert common emojis into sentiment marker words (YouTube comments use many)
2. Lowercase
3. Remove URLs, @mentions and HTML tags
4. Expand contractions ("don't" -> "do not") so negation survives
5. Replace timestamps (3:45) and years (2025) with marker words and remember
   whether the comment contained "?" or "!" (strong style clues)
6. Remove punctuation / special characters
7. Collapse extra whitespace
8. Remove stopwords (but KEEP negations such as "not" and "never")
"""

import re

# --------------------------------------------------------------------------- #
# Stopwords
# --------------------------------------------------------------------------- #
# Words that carry sentiment or negation must NOT be removed, otherwise
# "not good" would become just "good".
NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither",
    "cannot", "without",
}

_STOPWORDS_RAW = """
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself of
off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to under until up was we were what when where which while who
whom why will with would you your yours yourself yourselves also get got
really
"""

# Question words are useful clues (questions are usually Neutral comments).
QUESTION_WORDS = {"what", "how", "why", "which", "who", "when", "where"}

# Negations, intensifiers and contrast words are kept because they change the
# meaning of a comment ("not good", "too boring", "good but slow").
INTENSIFIERS = {"but", "more", "most", "too", "very"}
_KEEP = NEGATIONS | QUESTION_WORDS | INTENSIFIERS
STOPWORDS = set(_STOPWORDS_RAW.split()) - _KEEP

# --------------------------------------------------------------------------- #
# Emoji handling
# --------------------------------------------------------------------------- #
POSITIVE_EMOJIS = set("😍🥰😊😁😀😄🤩👍👏🙌🔥💯👌😂🤣💖💕❤♥🥳😎🤯💙💚💜🧡")
NEGATIVE_EMOJIS = set("😡😠🤬👎🤮💩😤😒🙄😞😴🤦😩😫")
NEUTRAL_EMOJIS = set("🤔🤷😐😑")

_VARIATION_SELECTORS = re.compile(r"[︎️‍]")

# --------------------------------------------------------------------------- #
# Regular expressions (compiled once)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HTML_RE = re.compile(r"<[^>]+>")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SPACES_RE = re.compile(r"\s+")

_CONTRACTIONS = [
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bcan['’]t\b"), "cannot"),
    (re.compile(r"\bain['’]t\b"), "is not"),
    (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"['’]re\b"), " are"),
    (re.compile(r"['’]ve\b"), " have"),
    (re.compile(r"['’]ll\b"), " will"),
    (re.compile(r"['’]d\b"), " would"),
    (re.compile(r"\bi['’]m\b"), "i am"),
    (re.compile(r"['’]s\b"), ""),
]

# Words that are produced by the preprocessing itself (not real words).
MARKER_TOKENS = {
    "emojipositive", "emojinegative", "emojineutral",
    "timestamp", "yearnum", "questionmark", "exclamationmark",
}


def _convert_emojis(text: str) -> str:
    """Replace known emojis with marker words; other symbols are left to be stripped."""
    text = _VARIATION_SELECTORS.sub("", text)
    out = []
    for ch in text:
        if ch in POSITIVE_EMOJIS:
            out.append(" emojipositive ")
        elif ch in NEGATIVE_EMOJIS:
            out.append(" emojinegative ")
        elif ch in NEUTRAL_EMOJIS:
            out.append(" emojineutral ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text, remove_stopwords: bool = True) -> str:
    """
    Clean one comment and return a string of space-separated tokens.

    >>> clean_text("I DIDN'T like it!!! https://x.co 😡")
    'not like emojinegative exclamationmark'
    """
    if text is None:
        return ""
    text = str(text)

    text = _convert_emojis(text)
    has_question = "?" in text
    has_exclamation = "!" in text
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _TIMESTAMP_RE.sub(" timestamp ", text)
    text = _YEAR_RE.sub(" yearnum ", text)
    text = _NON_ALNUM_RE.sub(" ", text)          # punctuation, symbols, other emojis
    text = _SPACES_RE.sub(" ", text).strip()

    tokens = text.split()
    if has_question:
        tokens.append("questionmark")
    if has_exclamation:
        tokens.append("exclamationmark")
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    # Drop single characters (e.g. stray "x", "k") except digits-free noise
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Helpers for the "Comment Insights" section
# --------------------------------------------------------------------------- #
# Extra words that are frequent in YouTube comments but not interesting to display.
_INSIGHT_EXTRA_STOPWORDS = {
    "video", "videos", "channel", "youtube", "comment", "comments", "watch",
    "watching", "watched", "one", "would", "like", "still", "even", "much",
    "many", "make", "made", "thing", "things", "know", "think", "going", "want",
    "see", "way", "lot", "lol", "bro", "guys", "dont", "didnt", "something",
    "anyone", "everyone", "actually", "honestly", "yeah", "okay", "ok",
}
INSIGHT_STOPWORDS = (
    NEGATIONS | QUESTION_WORDS | INTENSIFIERS | _INSIGHT_EXTRA_STOPWORDS | MARKER_TOKENS
)


def tokenize_for_insights(text) -> list:
    """Return meaningful, unique-per-comment words used for word-frequency charts."""
    cleaned = clean_text(text, remove_stopwords=True)
    words = []
    seen = set()
    for token in cleaned.split():
        if len(token) < 3 or token.isdigit() or token in INSIGHT_STOPWORDS:
            continue
        if token not in seen:          # count a word once per comment
            seen.add(token)
            words.append(token)
    return words


if __name__ == "__main__":
    demo = [
        "This video was AMAZING!!! 😍 https://youtu.be/abc",
        "I didn't like it. Waste of time 😡",
        "What song is at 3:45? Who's here in 2025?",
    ]
    for sentence in demo:
        print(f"{sentence!r}\n  -> {clean_text(sentence)!r}")
