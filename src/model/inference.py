from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class FastChatbot:
    def __init__(self, model_path="models/fine-tuned-chatbot"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def chat(self, message):
        prompt = f"Human: {message}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=80)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()

# Test
if __name__ == "__main__":
    bot = FastChatbot()
    print(bot.chat("Hello!"))