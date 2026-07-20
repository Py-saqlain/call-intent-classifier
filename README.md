# Call Intent Classifier

A system that reads real call transcripts and classifies them into one of four intents: **booking**, **transfer**, **cancellation**, or **modification** — built as part of an AI internship task at Senarios.

🔗 **Live demo:** [call-intent-classifier-9bzatdar4zw6fmkwvuscvp.streamlit.app](https://call-intent-classifier-9bzatdar4zw6fmkwvuscvp.streamlit.app)

---

## Overview

The task started as a classical ML problem — train a model to classify call intent — but the real dataset (2000+ raw, unlabeled, multi-turn call transcripts) made that approach impractical without a large manual labeling effort. This repo documents both phases:

1. **Phase 1 — Classical ML experiment:** trained and compared 5 models (Logistic Regression, SVM, Random Forest, Decision Tree, KNN) on a small synthetic dataset using TF-IDF features, validated with 5-fold cross-validation. SVM performed best (~89% CV accuracy).
2. **Phase 2 — Production approach (LLM-based):** since real data was unlabeled and conversational (not short clean phrases), the final system uses prompt-based zero-shot classification with Groq's LLaMA 3.3 70B model, reasoning over the full transcript rather than isolated keywords.

## Features

- Upload a `.txt` call transcript **or** paste text directly
- Returns one of 4 intents with a confidence score and a one-line explanation
- Handles messy, real conversational transcripts (interruptions, hesitations, ASR noise)
- Correctly distinguishes caller intent from incidental events (e.g. an agent-side technical transfer isn't misread as the caller's request)

## Screenshots

**Upload interface:**

![Upload UI](screenshots/upload_ui.jpg)

**Classification result:**

![Classification result](screenshots/classify_result.jpg)

## Tech Stack

- **Python**, **Streamlit** (UI)
- **Groq API** (LLaMA 3.3 70B) for zero-shot intent classification
- **scikit-learn**, **pandas** (Phase 1 classical ML experiment)
- **python-dotenv** for local secret management

## Architecture

```
User uploads/pastes transcript
        ↓
Streamlit app (app.py)
        ↓
Groq LLaMA 3.3 70B (system prompt defines 4 categories + edge-case rules)
        ↓
Structured JSON response: {intent, confidence, reason}
        ↓
Displayed in UI
```

## Setup (local)

```bash
git clone https://github.com/Py-saqlain/call-intent-classifier.git
cd call-intent-classifier
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Evaluation Notes

Accuracy was spot-checked against manually labeled real transcript samples. Results were strong on clear-intent calls; the main source of error was genuinely ambiguous calls (e.g. a caller "checking on a delayed pickup" doesn't cleanly map to any single category). A full statistically robust accuracy report would require a larger labeled validation set than was feasible within the task timeline.

## Project Structure

```
call-intent-classifier/
├── app.py                      # Streamlit UI (final deployed app)
├── requirements.txt
├── src/
│   ├── generate_data.py        # Phase 1: synthetic dataset creation
│   ├── eda.py                  # Phase 1: exploratory data analysis
│   ├── preprocess.py           # Phase 1: TF-IDF vectorization
│   ├── train.py                # Phase 1: trains 5 classical models
│   ├── evaluate.py             # Phase 1: single-split evaluation
│   ├── cross_validate.py       # Phase 1: 5-fold CV comparison
│   ├── llm_classify.py         # Phase 2: core LLM classification logic
│   ├── batch_process.py        # Phase 2: batch classify a folder of transcripts
│   ├── create_validation_sample.py  # Phase 2: sample files for manual labeling
│   ├── quick_label.py          # Phase 2: CLI tool for fast manual labeling
│   └── compute_accuracy.py     # Phase 2: compares manual labels vs predictions
└── data/                       # synthetic training data (Phase 1)
```

## Author

Saqlain — BS Software Engineering, PUCIT | AI Internship, Senarios
[GitHub](https://github.com/Py-saqlain) · [HuggingFace](https://huggingface.co/py-saqlain)
