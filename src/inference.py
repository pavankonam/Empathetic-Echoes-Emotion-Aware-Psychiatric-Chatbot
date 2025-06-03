# src/inference.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Emotion list – must match what was used in training
EMOTIONS = [
    "sentimental", "worried", "excited", "nervous", "lonely",
    "sad", "terrified", "hopeful", "ashamed", "grateful"
]
class EmpatheticChatbot:
    def __init__(self, model_path="/Users/pavankonam/Desktop/GenAI_Text/models/gpt2_finetuned_improved"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, emotion, user_input, chat_history="", max_length=150):
        emotion_tag = f"[{emotion}]"
        prompt = f"{emotion_tag} User: {chat_history}\nUser: {user_input}\nBot:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        bot_response = response.split("Bot:")[-1].strip()

        # Clean up unwanted tokens
        for emo in EMOTIONS:
            bot_response = bot_response.replace(f"[{emo}]", "").strip()
        bot_response = bot_response.split("\n")[0]  # Only take first line

        return bot_response