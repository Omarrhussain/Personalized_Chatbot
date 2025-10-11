from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

class FastChatbot:
    def __init__(self, base_model_name="distilbert/distilgpt2", adapter_path="models/fine-tuned-chatbot"):
        print("🔄 Loading base model + LoRA adapter...")
        
        # Force offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # Load base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                local_files_only=False,  # Need to download base model once
                trust_remote_code=True
            )
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=False
            )
            
            # Load LoRA adapter
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    local_files_only=True
                )
                print("✅ Base model + LoRA adapter loaded")
            else:
                self.model = self.base_model
                print("⚠️  Using base model only (adapter not found)")
                
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def chat(self, message):
        try:
            prompt = f"Human: {message}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Assistant:")[-1].strip()
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

def test_chatbot():
    try:
        bot = FastChatbot()
        print("🤖 Chatbot test:")
        response = bot.chat("Hello!")
        print(f"Bot: {response}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_chatbot()