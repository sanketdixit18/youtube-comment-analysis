"""
Word-frequency insights ("Comment Insights" section of the dashboard).

Everything here is computed from the comments that were actually analysed.
A word is counted at most once per comment, so one spammy comment that repeats
"love love love" cannot dominate the charts.
"""

from collections import Counter, defaultdict

from ml.preprocess import tokenize_for_insights

SENTIMENTS = ("Positive", "Negative", "Neutral")


def _top(counter: Counter, n: int) -> list:
    # sort by count (desc) then alphabetically so the output is deterministic
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"word": word, "count": count} for word, count in items[:n]]


def build_insights(texts: list, sentiments: list, top_n: int = 10, cloud_n: int = 40) -> dict:
    """
    Parameters
    ----------
    texts       raw comment strings
    sentiments  predicted sentiment for each comment (same order)
    """
    per_sentiment = {s: Counter() for s in SENTIMENTS}
    overall = Counter()
    by_word = defaultdict(Counter)       # word -> {sentiment: count}

    for text, sentiment in zip(texts, sentiments):
        for word in tokenize_for_insights(text):
            overall[word] += 1
            if sentiment in per_sentiment:
                per_sentiment[sentiment][word] += 1
                by_word[word][sentiment] += 1

    cloud = []
    for item in _top(overall, cloud_n):
        word_counts = by_word[item["word"]]
        # dominant sentiment = the class this word appears in most often
        dominant = max(SENTIMENTS, key=lambda s: (word_counts.get(s, 0), s == "Neutral"))
        cloud.append({**item, "sentiment": dominant})

    return {
        "positive_words": _top(per_sentiment["Positive"], top_n),
        "negative_words": _top(per_sentiment["Negative"], top_n),
        "neutral_words": _top(per_sentiment["Neutral"], top_n),
        "overall_words": _top(overall, top_n),
        "word_cloud": cloud,
    }
