# Empathetic Echoes – Emotion-Aware Psychiatric Chatbot

> “The most important thing in communication is hearing what isn't said.” – Peter Drucker  
>  
> _Empathetic Echoes_ builds on this idea — understanding emotional context to generate meaningful, supportive responses.

---

## Project Overview

**Empathetic Echoes** is an advanced, emotion-aware psychiatric chatbot designed to simulate empathetic conversations using a fine-tuned GPT-2 model. It generates responses that reflect **emotional intelligence**, **contextual memory**, and **compassionate tone**, aiming to emulate therapist-like interactions.

This project leverages the [Hugging Face Empathetic Dialogues](https://huggingface.co/datasets/empathetic_dialogues) dataset (~25,000 real-world conversations) to train a language model that:
- Detects emotional context
- Maintains multi-turn dialogue history
- Generates supportive and human-like replies

---

## Key Features

| Feature | Description |
|--------|-------------|
|  Emotion-Aware Responses | Responses conditioned on 30+ emotional labels |
|  Contextual Memory | Retains dialogue history for relevant responses |
|  GPT-2 Backbone | Lightweight yet expressive Transformer model |
|  MLFlow Integration | Track experiments, metrics, and checkpoints |
|  Streamlit Interface | Chat via a user-friendly web UI |
|  BLEU & ROUGE Evaluation | Measures model performance quantitatively |

---

##  Project Structure
``` text
EmpatheticEchoes/
├── models/
│ └── gpt2_finetuned/ # Trained model and tokenizer #Look into the link for model data
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

##  How It Works

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

### ✅ Inference and Generation

```bash
from src.inference import generate_response
generate_response("I'm feeling overwhelmed and anxious.", emotion="anxious")
```
- Loads trained model and tokenizer.
- Accepts emotion label and user input.
- Returns emotion-aligned, contextually coherent response.

### ✅ Evaluation

```bash
cd /gpfs/home/bol2142/GenAI_Text_Chatbot
PYTHONPATH=$(pwd) python src/evaluate_model.py
```
- Calculates BLEU and ROUGE scores.
- Compares generated responses to ground truth.

### ✅ Run the Chatbot UI

```bash
streamlit run src/gui_app.py
```
- Provides an intuitive GUI for interaction.
- Users can select an emotion or let the bot infer one.
- Real-time, multi-turn conversation support.

_Note: For the trained model use the below link
https://drive.google.com/drive/folders/18yTUZVUxPhXAWSTzuhZYr6ZWdVqRVc7L?usp=drive_link_
## Evaluation Results
| Metric      | Score  |
| ----------- | ------ |
| **BLEU**    | 0.0080 |
| **ROUGE-1** | 0.1311 |
| **ROUGE-2** | 0.0056 |
| **ROUGE-L** | 0.1180 |


## Sample Dialogue
<img width="1512" alt="Screenshot 2025-06-03 at 09 37 11" src="https://github.com/user-attachments/assets/4fcac436-f468-4d18-bc05-c6564a6a1e53" />

## Installation Guide
### 1. Clone the Repository
```bash
git clone https://github.com/pavankonam/Empathetic-Echoes-Emotion-Aware-Psychiatric-Chatbot.git
cd empathetic-echoes
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Final Note
Empathetic Echoes showcases how machine learning and NLP can foster emotional awareness in AI systems. It's not a substitute for professional mental health support but serves as a tool for empathy-driven interaction.
