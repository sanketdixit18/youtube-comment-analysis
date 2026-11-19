
import os
import requests

from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FILES = {
    "sentiment_model.pkl": "https://github.com/sanketdixit18/youtube-comment-analysis/releases/download/v1.0.0/sentiment_model.pkl",
    "vectorizer.pkl": "https://github.com/sanketdixit18/youtube-comment-analysis/releases/download/v1.0.0/vectorizer.pkl",
}

for filename, url in FILES.items():
    file_path = MODEL_DIR / filename

    print(f"Downloading {filename}...")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    file_path.write_bytes(response.content)

    print(f"Downloaded: {file_path}")

print("All model files downloaded successfully!")
