# src/evaluate_model.py

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define emotions used (match what was used during training)
EMOTIONS = [
    "sentimental", "worried", "excited", "nervous", "lonely",
    "sad", "terrified", "hopeful", "ashamed", "grateful"
]

def load_trained_model(model_path=None):
    if model_path is None:
        model_path = "/Users/pavankonam/Desktop/GenAI_Text/models/gpt2_finetuned_improved"
        print(f"Using default model path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Make sure you've trained and saved it first.")

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def generate_response(model, tokenizer, emotion, user_input, chat_history="", max_length=150):
    emotion_tag = f"[{emotion}]"
    prompt = f"{emotion_tag} User: {chat_history}\nUser: {user_input}\nBot:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    bot_response = response.split("Bot:")[-1].strip()
    for emo in EMOTIONS:
        bot_response = bot_response.replace(f"[{emo}]", "").strip()
    bot_response = bot_response.split("\n")[0]
    return bot_response.strip()

def compute_metrics(generated_texts, reference_texts):
    print("Computing BLEU and ROUGE scores...")

    # Initialize scorers
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge1, rouge2, rougeL = [], [], []

    for gen, ref in tqdm(zip(generated_texts, reference_texts), total=len(generated_texts)):
        # Compute BLEU
        bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu)

        # Compute ROUGE
        scores = scorer.score(ref, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(rouge1) / len(rouge1)
    avg_rouge2 = sum(rouge2) / len(rouge2)
    avg_rougeL = sum(rougeL) / len(rougeL)

    return {
        "BLEU": avg_bleu,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL
    }

def main():
    # Load dataset
    print("Loading empathetic_dialogues dataset...")
    dataset = load_dataset("empathetic_dialogues")
    test_data = dataset["test"]

    # Group by conversation ID
    conversations = {}
    for example in tqdm(test_data, desc="Grouping conversations"):
        conv_id = example['conv_id']
        if conv_id not in conversations:
            conversations[conv_id] = {
                'context': example['context'],
                'utterances': []
            }
        speaker = "User" if example['speaker_idx'] == 0 else "Bot"
        conversations[conv_id]['utterances'].append({
            'speaker': speaker,
            'text': example['utterance']
        })

    # Build input-response pairs
    inputs, references = [], []

    for conv_id, conv_data in tqdm(conversations.items()):
        utterances = conv_data['utterances']
        emotion = conv_data['context']

        for i in range(len(utterances)):
            if utterances[i]['speaker'] != "User":
                continue
            context = ""
            for j in range(i):
                ctx_speaker = utterances[j]['speaker']
                ctx_text = utterances[j]['text']
                context += f"{ctx_speaker}: {ctx_text}\n"

            if i + 1 < len(utterances) and utterances[i + 1]['speaker'] == "Bot":
                user_input = utterances[i]['text']
                bot_response = utterances[i + 1]['text']
                inputs.append((emotion, user_input, context))

                # Save ground truth
                references.append(bot_response)

    print(f"\nEvaluating on {len(inputs)} examples...")

    # Load model
    model, tokenizer = load_trained_model()

    # Generate responses
    generated_responses = []
    for emotion, user_input, context in tqdm(inputs, desc="Generating responses"):
        response = generate_response(model, tokenizer, emotion, user_input, context)
        generated_responses.append(response)

    # Compute metrics
    metrics = compute_metrics(generated_responses, references)

    print("\n📊 Evaluation Results:")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

if __name__ == "__main__":
    main()