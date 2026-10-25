# Presentation script and viva preparation

## Part 1 - 5-minute demo script

1. **Problem (30 s).** "Popular videos get thousands of comments. Nobody can read them all, so we use NLP to measure audience opinion automatically."
2. **Pipeline (45 s).** Scroll to *How It Works*: comments → preprocessing → TF-IDF → Logistic Regression → classification → dashboard.
3. **Demo Mode (90 s).** Press *Demo Mode*. Point to the summary cards, the doughnut and bar charts, and the percentages. Explain that the demo comments come from a CSV but the predictions come from the real trained model.
4. **Table and insights (60 s).** Search "audio", filter *Negative*, sort by confidence. Show the word cloud and top positive / negative words.
5. **Analyze a Comment (45 s).** Click *Positive example*, then *Analyze Sentiment*. Show the probabilities and key terms. Then type a tricky sentence such as "not bad at all" and discuss.
6. **Model information (30 s).** Show accuracy, F1, confusion matrix. Say honestly: "small dataset, so this is a demonstration; real accuracy would need more data."
7. **Live API (optional).** If you have a key, paste a URL. Mention that without a key the app falls back to demo data automatically.

Tips: switch to dark mode on the projector if the room is dark; run `python ml/train.py` once before the demo so the metrics are fresh; keep the terminal visible to show the training output.

## Part 2 - Short explanations for the About section

- **NLP**: teaching computers to understand human language.
- **Sentiment analysis**: deciding whether text is positive, negative or neutral.
- **TF-IDF**: word importance = how often it appears in this comment × how rare it is across all comments.
- **Logistic Regression**: learns word weights, turns the total into class probabilities.

## Part 3 - Viva questions and answers

**1. What is NLP?**
Natural Language Processing is a branch of AI that lets computers read, understand and generate human language. Examples: translation, chatbots, spam filters, sentiment analysis.

**2. What is sentiment analysis?**
An NLP task that finds the opinion or emotion in text, here classified as Positive, Negative or Neutral.

**3. Why did you use TF-IDF?**
Computers need numbers, not words. TF-IDF turns each comment into a vector and gives higher weight to words that are frequent in the comment but rare in other comments, and lower weight to common words. It is simple, fast and works well on small datasets. I also used bigrams so phrases like "not good" are captured.

**4. Why Logistic Regression?**
It is a strong baseline for text, trains in under a second, needs little data, outputs real probabilities (used as confidence) and is interpretable because every word has a weight. Deep models need much more data and hardware.

**5. What is tokenization?**
Splitting text into smaller units (tokens), usually words. "Great video!" becomes ["great", "video"]. TfidfVectorizer does this internally after my cleaning step.

**6. What are stopwords?**
Very common words such as "the", "is", "and" that carry little meaning. I remove them but deliberately keep negations ("not", "never") because removing them would flip the meaning of a sentence.

**7. What is stemming / lemmatization?**
Both reduce words to a base form. Stemming chops endings ("running" → "run", sometimes crude: "studies" → "studi"). Lemmatization uses a dictionary to return a real word ("better" → "good"). I did not use them because TF-IDF with bigrams already worked well and the original words are easier to show in the word charts. They could be added as an improvement.

**8. What is a train-test split?**
Dividing data into a training part (the model learns from it) and a test part (kept hidden to measure performance). I use 80 % / 20 %, stratified so every class keeps the same proportion. The vectorizer is fitted only on the training set to avoid data leakage.

**9. What is overfitting?**
When a model memorises the training data and performs badly on new data. Signs: very high training accuracy, much lower test accuracy. I reduce the risk with a held-out test set, cross-validation and L2 regularisation (the default in Logistic Regression).

**10. What is precision?**
Of all comments the model labelled as class X, the fraction that really are X. `precision = TP / (TP + FP)`.

**11. What is recall?**
Of all comments that really are class X, the fraction the model found. `recall = TP / (TP + FN)`.

**12. What is F1 score?**
The harmonic mean of precision and recall: `2PR / (P + R)`. It is high only when both are high. I report the macro average (the mean over the three classes).

**13. What is a confusion matrix?**
A table where rows are the actual classes and columns the predicted classes. The diagonal shows correct predictions; off-diagonal cells show which classes get confused (for example Neutral predicted as Negative).

**14. How does the YouTube API work?**
The YouTube Data API v3 is a web service. My server sends an HTTPS GET request to the `commentThreads` endpoint with the video ID and an API key; YouTube returns JSON with up to 100 comments and a `nextPageToken`. I keep requesting pages with that token until I have enough comments. The key is read from an environment variable and stays on the server.

**15. What are the limitations of this project?**
Small hand-labelled training set; difficulty with sarcasm, slang and mixed-language text; only English; bag-of-words ignores long-range context; only top-level comments; daily API quota; the neutral class is the hardest to define.

**16. How is confidence calculated?**
`predict_proba` of Logistic Regression: softmax over the class scores gives probabilities that sum to 1. The confidence is the probability of the predicted class. It is a model probability, not a guarantee of correctness.

**17. Why is Demo Mode needed and is it fake?**
Demo Mode removes dependence on internet and API quota during a presentation. The comments are loaded from a CSV, but sentiment, confidence, charts and insights are produced by the real model. The dataset is never used for training.

**18. How did you handle emojis and negation?**
Emojis are converted to marker words (`emojipositive`, `emojinegative`) which the model learns from the training data. Contractions are expanded ("don't" → "do not") and negation words are not removed, so bigrams such as "not good" can be learned.

**19. What does the "Analyze a Comment" explanation show?**
Logistic Regression is linear, so the contribution of each word is `weight × TF-IDF value`. I show the words with the largest positive contribution to the predicted class. It is a simple hint about the model, not a proof of causality.

**20. How did you secure the application?**
API key in an environment variable (never in JavaScript or responses), input validation with Pydantic, length limits, a per-client rate limit, comments rendered with `textContent` (prevents XSS), CSV export protected against formula injection, security headers including a Content-Security-Policy, and friendly error messages instead of stack traces.

**21. Why FastAPI instead of Flask?**
FastAPI gives automatic request validation, automatic interactive API documentation at `/docs`, type hints and better performance. The structure is similar to Flask (routes are decorated functions), so it is easy to understand.

**22. How could accuracy be improved?**
More and better labelled data, lemmatization, tuning hyper-parameters with cross-validation, other models (Linear SVM, Naive Bayes, ensembles) and transformer models such as BERT.
