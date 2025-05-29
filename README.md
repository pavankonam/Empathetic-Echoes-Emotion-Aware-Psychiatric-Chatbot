# 🧠 Empathetic Echoes – Emotion-Aware Psychiatric Chatbot

> “The most important thing in communication is hearing what isn't said.” – Peter Drucker  
>  
> _Empathetic Echoes_ builds on this idea — understanding emotional context to generate meaningful, supportive responses.

---

## 🌟 Project Overview

**Empathetic Echoes** is an advanced, emotion-aware psychiatric chatbot designed to simulate empathetic conversations using a fine-tuned GPT-2 model. It generates responses that reflect **emotional intelligence**, **contextual memory**, and **compassionate tone**, aiming to emulate therapist-like interactions.

This project leverages the [Hugging Face Empathetic Dialogues](https://huggingface.co/datasets/empathetic_dialogues) dataset (~25,000 real-world conversations) to train a language model that:
- Detects emotional context
- Maintains multi-turn dialogue history
- Generates supportive and human-like replies

---

## 🧩 Key Features

| Feature | Description |
|--------|-------------|
| 🤖 Emotion-Aware Responses | Responses conditioned on 30+ emotional labels |
| 🎯 Contextual Memory | Retains dialogue history for relevant responses |
| 🧠 GPT-2 Backbone | Lightweight yet expressive Transformer model |
| 📊 MLFlow Integration | Track experiments, metrics, and checkpoints |
| 🖥️ Streamlit Interface | Chat via a user-friendly web UI |
| 📈 BLEU & ROUGE Evaluation | Measures model performance quantitatively |

---

## 📁 Project Structure
``` text
EmpatheticEchoes/
├── data/
│ ├── raw/ # Original Hugging Face dataset
│ └── processed/ # Preprocessed JSON files for training
├── models/
│ └── gpt2_finetuned/ # Trained model and tokenizer
├── src/
│ ├── data_preprocessing.py # Prepares and tokenizes dataset
│ ├── model_training.py # Fine-tunes GPT-2 on dialogue data
│ ├── inference.py # Emotion-aware response generation
│ ├── evaluate_model.py # BLEU, ROUGE metrics evaluation
│ ├── gui_app.py # Streamlit web app
│ └── inspect_dataset.py # Dataset exploration/debugging
├── mlflow_runs/ # MLFlow logs and tracking artifacts
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

## 🔧 How It Works

### ✅ Data Preprocessing

```bash
python src/data_preprocessing.py
```
- Downloads and prepares the Empathetic Dialogues dataset.
- Formats conversations as input-output pairs.
- Tags each dialogue with the corresponding emotion ([proud], [anxious], etc.).

### ✅ Training the Model

```bash
cd /gpfs/home/bol2142/GenAI_Text_Chatbot
PYTHONPATH=$(pwd) python src/model_training.py
```
- Fine-tunes GPT-2 on the processed dialogues.
- Special tokens represent emotions to guide the model.
- Tracked with MLFlow to log metrics, configs, and checkpoints.

