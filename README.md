# 🔍 Multilingual Fake News Detector

An AI-powered fake news detection system that combines a trained stacking ensemble classifier with live web search verification via OpenAI. Supports English, Igbo, Hausa, and Yoruba.

---

## How It Works

Every news statement goes through two independent verification pipelines:

**1. ML Model Pipeline**
- Text is cleaned and transformed using TF-IDF (word + character n-grams) combined with sentence embeddings (`all-MiniLM-L6-v2`)
- A stacking ensemble (Logistic Regression + LinearSVC + XGBoost → Logistic Regression meta-classifier) predicts whether the statement is REAL or FAKE
- Returns a probability score and confidence level

**2. AI Web Search Pipeline**
- OpenAI `gpt-4o-mini` with `web_search_preview` searches the web for the claim
- Returns a numbered fact-check analysis with cited, clickable sources

Non-English statements (Igbo, Hausa, Yoruba) are automatically detected and translated to English before analysis, then the result is translated back.

---

## Project Structure

```
fake-news-detector/
│
├── app.py                          # Streamlit app (single entry point)
│
├── models/
│   ├── preprocessor.joblib         # Fitted TextPreprocessor
│   └── classifier.joblib           # Fitted StackingClassifier
│
├── src/
│   ├── scripts/
│   │   ├── preprocessor.py         # TextPreprocessor class
│   │   └── train.py                # Model training script
│   └── utils/
│       └── utils.py                # load_data, save_objects helpers
│
├── data/
│   └── datasets/
│       └── final_data/
│           └── data.csv            # Training dataset (statement, label)
│
├── requirements.txt
├── .env                            # Local secrets (never commit)
├── .gitignore
└── README.md
```

---

## Model Architecture

### TextPreprocessor
Combines three feature representations:

| Feature | Method | Config |
|---|---|---|
| Word-level TF-IDF | `TfidfVectorizer` | 5000 features, unigrams + bigrams |
| Character-level TF-IDF | `TfidfVectorizer` | 5000 features, 3–5 char n-grams |
| Sentence Embeddings | `all-MiniLM-L6-v2` | 384-dim dense vectors |

All three are horizontally stacked into a single sparse matrix per sample.

### Stacking Classifier

| Layer | Models |
|---|---|
| Base learners | Logistic Regression, CalibratedClassifierCV(LinearSVC), XGBoost |
| Meta-learner | Logistic Regression |
| CV strategy | 3-fold, `predict_proba` stack method |

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 5. Train the model (if models/ are not present)
```bash
python src/scripts/train.py
```
This saves `preprocessor.joblib` and `classifier.joblib` into `models/`.

### 6. Run the app
```bash
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

1. Push your repo to GitHub — **do not include `.env` or `models/`** if they are large
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file to `app.py`
4. Click **Advanced Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-..."
```

5. Deploy — no backend server required

> **Note:** If your `.joblib` model files are large (>100MB), use [Git LFS](https://git-lfs.com/) or host them on cloud storage and load them at runtime.

---

## Requirements

```
streamlit
openai
joblib
scikit-learn
xgboost
sentence-transformers
scipy
langdetect
python-dotenv
nltk
```

---

## Dataset Format

The training script expects a CSV at `data/datasets/final_data/data.csv` with two columns:

| Column | Type | Description |
|---|---|---|
| `statement` | string | The raw news statement |
| `label` | int | `1` = Real, `0` = Fake |

---

## Supported Languages

| Code | Language |
|---|---|
| `en` | English |
| `ig` | Igbo |
| `ha` | Hausa |
| `yo` | Yoruba |

Language is auto-detected. Unsupported languages default to English processing.

---

## Output

| Field | Description |
|---|---|
| **Verdict** | REAL or FAKE |
| **Fake Probability** | Model confidence (0–100%) |
| **Reliability** | High / Medium / Low based on probability distance from 0.5 |
| **Analysis** | Numbered fact-check from GPT-4o with web sources |
| **Sources** | Clickable links to referenced articles |

Results can be exported as a JSON file.

---

## Limitations

- The ML model's accuracy depends on the training data distribution
- Web search verification requires a valid OpenAI API key with `web_search_preview` access
- Translation quality for Igbo, Hausa, and Yoruba depends on GPT-4o-mini coverage of those languages
- The model was not trained on real-time events; always cross-check critical claims

---

## License

MIT
