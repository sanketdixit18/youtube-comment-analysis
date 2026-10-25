# YouTube Sentiment Analyzer

A complete NLP web application that classifies YouTube comments as **Positive**, **Negative** or **Neutral** and shows the results in a modern analytics dashboard.

> Backend: **FastAPI** · NLP/ML: **scikit-learn** (TF-IDF + Logistic Regression) · Frontend: **HTML, CSS, vanilla JavaScript, Chart.js**

---

## 1. Project overview

You paste a YouTube link, the app downloads the video's comments (YouTube Data API), cleans the text, converts it to TF-IDF vectors, predicts a sentiment with a trained Logistic Regression model and displays counts, charts, word insights and a searchable table.

It also has a **Demo Mode** with 154 realistic sample comments, so the project works for a college demonstration with **no internet, no API key and no paid service**.

## 2. Features

- Analyze a YouTube video URL (real **YouTube API mode**) or use **Demo Mode** (sample dataset)
- Automatic, clearly labelled fallback to demo data if there is no API key, the quota is used up or the API is unreachable
- Summary cards: total, positive, negative, neutral and overall sentiment
- Doughnut chart, bar chart and percentage bars
- Searchable, filterable, sortable, paginated comment table with confidence scores (+ CSV download)
- **Comment Insights**: top positive / negative / overall words, word-frequency chart, word cloud (all computed from the analysed comments)
- **Analyze a Comment**: type any text and get sentiment, confidence, class probabilities and the words that influenced the model
- Model information page: training/testing samples, accuracy, precision, recall, F1, per-class table, confusion matrix, 5-fold cross-validation (all calculated, nothing hard-coded)
- "How It Works" pipeline and "About the Project" explanations for presentations
- Dark / light mode (saved in `localStorage`), responsive layout, loading animation, toasts, empty and error states
- Security basics: API key only on the server, input validation, rate limiting, safe rendering of comments, security headers, no stack traces shown to users
- Interactive API documentation at `/docs` (FastAPI feature)

## 3. Technologies used

| Layer | Tools |
|---|---|
| Backend | Python 3.9+, FastAPI, Uvicorn, Pydantic, Jinja2 |
| NLP / ML | pandas, numpy, scikit-learn (TfidfVectorizer, LogisticRegression), joblib |
| YouTube | YouTube Data API v3 through `requests` |
| Frontend | HTML5, CSS3 (custom properties, grid), vanilla JavaScript, Chart.js 4 (bundled locally, works offline) |

## 4. NLP workflow

1. **Load** labelled comments (`data/training_data.csv`, 557 hand-written comments, 3 balanced classes)
2. **Clean** each comment (`ml/preprocess.py`)
   - emojis become marker words (😍 → `emojipositive`, 😡 → `emojinegative`)
   - lowercase, remove URLs / @mentions / HTML
   - expand contractions (`don't` → `do not`) so negation survives
   - timestamps (`3:45`) and years (`2025`) become marker words; `?` and `!` are remembered as markers
   - remove punctuation and extra whitespace
   - remove stopwords, but **keep** negations (`not`, `never`, `no`) and question words
3. **Split** 80 % train / 20 % test (stratified, `random_state=42`)
4. **TF-IDF** vectorization with unigrams + bigrams (fitted on the training set only)
5. **Train** Logistic Regression
6. **Evaluate** on the unseen test set + 5-fold cross-validation
7. **Save** `models/vectorizer.pkl`, `models/sentiment_model.pkl`, `models/metrics.json`

## 5. Machine learning algorithm

**Logistic Regression** (multinomial) learns one weight per word (and word pair) for each class. For a new comment it multiplies the TF-IDF values by the weights, adds them up and uses the softmax function to turn the totals into **probabilities** that add up to 1. The highest probability is the prediction and **is the confidence shown in the UI**. These are real model probabilities, not invented numbers.

Why this model: fast, works well with small text datasets, no GPU needed, gives probabilities, and its weights are easy to explain in a viva.

## 6. Project structure

```
youtube-sentiment-analysis/
├── app.py                  FastAPI application and API routes
├── requirements.txt
├── README.md
├── VIVA_AND_PRESENTATION.md   presentation script + 20 viva questions
├── .env.example            template for your YouTube API key
├── data/
│   ├── training_data.csv   labelled data used to train the model
│   └── sample_comments.csv demo comments (never used for training)
├── models/                 created by training (vectorizer.pkl, sentiment_model.pkl, metrics.json)
├── ml/
│   ├── preprocess.py       text cleaning
│   ├── train.py            training + evaluation pipeline
│   └── predictor.py        loads the model and predicts
├── utils/
│   ├── youtube.py          YouTube API helper (URL parsing, pagination, errors)
│   └── insights.py         word-frequency insights
├── templates/index.html    dashboard page
├── static/
│   ├── css/style.css
│   ├── js/script.js        dashboard logic
│   ├── js/theme-init.js    applies the saved theme before first paint
│   └── vendor/chart.umd.min.js   Chart.js (bundled, MIT licence)
└── tests/test_app.py       automated tests (optional)
```

## 7. Installation

You need **Python 3.9 or newer** (`python --version`).

### 8. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows (Command Prompt):** `venv\Scripts\activate`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source venv/bin/activate`

### 9. Install dependencies

```bash
pip install -r requirements.txt
```

### 10. Train the model

```bash
python ml/train.py
```

This prints the accuracy, precision, recall, F1 score and confusion matrix and saves the model into `models/`.
(If you forget this step, the app trains the model automatically on first start, and also re-trains if the saved model was created with a different scikit-learn version.)

### 11. Run the application

```bash
python app.py
```

Open **http://127.0.0.1:8000** in your browser.
Alternative: `uvicorn app:app --reload` (auto-restarts when you edit the code).
API docs: http://127.0.0.1:8000/docs

## 12. Using Demo Mode

Press the **Demo Mode** button. The app analyses the 154 comments in `data/sample_comments.csv` with the real trained model, so every chart and number is genuine, only the comments come from a file instead of YouTube. No API key and no internet are required.

If you press **Analyze Video** without an API key, the app shows the same demo data and a notice explains why.

## 13. Configuring the YouTube API

```bash
# Windows:        copy .env.example .env
# macOS / Linux:  cp .env.example .env
```

Open `.env` and set `YOUTUBE_API_KEY=your_key_here`, then restart the server.
Alternatively set an environment variable: `set YOUTUBE_API_KEY=...` (Windows) or `export YOUTUBE_API_KEY=...` (macOS/Linux).

The key stays on the server. It is never sent to the browser and never shown in error messages. The status pill under the input box shows **YouTube API connected** when a key is found.

## 14. How to get a YouTube API key

1. Go to <https://console.cloud.google.com/> and sign in.
2. Create a new project (top bar → project list → *New project*).
3. Open **APIs & Services → Library**, search for **YouTube Data API v3** and click **Enable**.
4. Open **APIs & Services → Credentials → Create credentials → API key**.
5. Copy the key into your `.env` file. (Optional: restrict the key to the YouTube Data API v3.)

Notes: the API has a free daily quota (10,000 units by default). Each page of 100 comments costs about 1 unit, so even 500 comments is cheap. If the quota is exceeded the app falls back to Demo Mode automatically.

## 15. Example usage

1. Start the server and open the page.
2. Press **Demo Mode** → the dashboard fills with 154 comments.
3. Use the table: search `audio`, filter *Negative*, sort by confidence, download the CSV.
4. Scroll to **Analyze a Comment**, press the *Positive example* chip and **Analyze Sentiment**.
5. With an API key: paste `https://www.youtube.com/watch?v=dQw4w9WgXcQ` and press **Analyze Video**.

API examples (also try them in `/docs`):

```bash
curl -X POST http://127.0.0.1:8000/api/analyze-comment \
     -H "Content-Type: application/json" \
     -d '{"text": "I really enjoyed this video. The explanation was excellent!"}'

curl -X POST http://127.0.0.1:8000/api/demo
```

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Dashboard page |
| GET | `/api/status` | Model loaded? API key configured? (yes/no only) |
| GET | `/api/model-info` | Training statistics and metrics |
| POST | `/api/analyze` | Analyse a video URL `{url, max_comments, allow_fallback}` |
| POST | `/api/demo` | Analyse the sample dataset |
| POST | `/api/analyze-comment` | Analyse one text `{text}` |

How the overall sentiment is decided: `net score = (positive − negative) / total`. At least +0.15 → *Positive*, at most −0.15 → *Negative*, otherwise *Neutral* (if neutral is the biggest group) or *Mixed*.

## Running the tests (optional)

```bash
pip install pytest httpx
pytest -q        # a Starlette deprecation warning about httpx may appear; it is harmless
```

## Limitations

- The training set is small (557 comments) and hand-labelled, so accuracy on real YouTube comments will be lower than the test-set figure.
- Sarcasm, slang, mixed feelings and languages other than English are hard for a bag-of-words model.
- Only top-level comments are analysed (not replies), at most 500 per request.
- The rate limiter and the model are in memory, which is fine for a demo but not for a multi-server deployment.

## 16. Future improvements

- Train on a large public dataset (for example Sentiment140 or a labelled YouTube-comments corpus)
- Use transformer models (BERT / DistilBERT / RoBERTa) for better accuracy
- Detect language and support Hinglish / multilingual comments
- Analyse replies and track sentiment over time
- Add user accounts and a database to save analysis history
- Export a PDF report; deploy with Docker

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Port 8000 already in use | `set PORT=8010` (Windows) / `export PORT=8010`, then `python app.py` |
| "Model not loaded" message | Run `python ml/train.py`, then restart |
| Always shows demo data | `YOUTUBE_API_KEY` is not set or the quota is used up (read the notice in the dashboard) |
| `python` not found on macOS/Linux | use `python3` |
