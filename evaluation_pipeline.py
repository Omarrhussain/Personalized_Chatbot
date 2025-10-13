import os
import sys
sys.path.append('src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

class FixedChatbotEvaluator:
    def __init__(self, base_model_name="distilbert/distilgpt2", adapter_path=None):
        print("🔄 Loading model with LoRA adapter...")
        
        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        if adapter_path and os.path.exists(adapter_path):
            print(f"✅ Loading adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            print("⚠️  Using base model only")
            self.model = self.base_model
        
        # Load metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        
        print("✅ Model loaded successfully!")

    def generate_response(self, input_text, max_length=100):
        """Generate response for evaluation"""
        prompt = f"Human: {input_text}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()

    def calculate_perplexity(self, texts):
        """Calculate perplexity on given texts - FIXED VERSION"""
        perplexities = []
        
        for text in tqdm(texts, desc="Calculating Perplexity"):
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                perplexities.append(1000)  # High perplexity for empty responses
                continue
                
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # Check if input is not empty
                if inputs["input_ids"].numel() == 0:
                    perplexities.append(1000)
                    continue
                    
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                    
            except Exception as e:
                print(f"⚠️  Perplexity calculation failed for text: {e}")
                perplexities.append(1000)  # Default high perplexity
        
        return np.mean(perplexities), perplexities

    def evaluate_on_dataset(self, test_data_path=r"D:\Personalized_Chatbot\data\processed\validation.csv", sample_size=30):
        """Comprehensive evaluation"""
        print("📊 Starting evaluation...")
        
        # Load test data
        df = pd.read_csv(test_data_path)
        test_samples = df.sample(min(sample_size, len(df)))
        print(f"Evaluating on {len(test_samples)} samples...")
        
        # Generate predictions
        predictions = []
        references = []
        
        for idx, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Generating responses"):
            input_text = row['input']
            reference_response = row['response']
            
            generated_response = self.generate_response(input_text)
            
            predictions.append(generated_response)
            references.append([reference_response])
        
        # Calculate metrics
        print("🔄 Calculating metrics...")
        
        # BLEU and ROUGE
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        
        # Perplexity
        perplexity_mean, _ = self.calculate_perplexity(predictions)
        
        # Compile results
        results = {
            'bleu_score': bleu_results['bleu'],
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'perplexity': perplexity_mean
        }
        
        return results, predictions, references

def find_latest_checkpoint():
    """Find the latest checkpoint in the model directory"""
    model_dir = r"D:\Personalized_Chatbot\models\fine-tuned-chatbot"
    
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(model_dir, item)))
            except:
                continue
    
    if checkpoints:
        latest_step, latest_path = max(checkpoints, key=lambda x: x[0])
        print(f"🎯 Using latest checkpoint: {latest_path} (step {latest_step})")
        return latest_path
    
    # If no checkpoints, check for adapter files in root
    if os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")):
        print("🎯 Using adapter files from model root")
        return model_dir
    
    return None

def run_complete_evaluation():
    """Run complete evaluation with both models"""
    print("🤖 CHATBOT EVALUATION PIPELINE")
    print("=" * 60)
    
    # Find and evaluate fine-tuned model
    adapter_path = find_latest_checkpoint()
    
    results_ft = None
    results_base = None
    
    if adapter_path:
        print("\n🎯 EVALUATING FINE-TUNED MODEL...")
        evaluator_ft = FixedChatbotEvaluator(adapter_path=adapter_path)
        results_ft, preds_ft, refs_ft = evaluator_ft.evaluate_on_dataset(sample_size=25)
    else:
        print("❌ No fine-tuned model found")
    
    # Evaluate base model for comparison
    print("\n🔵 EVALUATING BASE MODEL...")
    evaluator_base = FixedChatbotEvaluator(adapter_path=None)
    results_base, preds_base, refs_base = evaluator_base.evaluate_on_dataset(sample_size=25)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    if results_ft:
        print("\n🎯 FINE-TUNED MODEL RESULTS:")
        for metric, value in results_ft.items():
            print(f"  {metric:15}: {value:.4f}")
    
    print("\n🔵 BASE MODEL RESULTS:")
    for metric, value in results_base.items():
        print(f"  {metric:15}: {value:.4f}")
    
    # Calculate improvement
    if results_ft:
        print("\n📊 IMPROVEMENT (Fine-tuned vs Base):")
        print("  " + "-" * 40)
        for metric in results_base.keys():
            if metric in results_ft:
                improvement = results_ft[metric] - results_base[metric]
                pct_improvement = (improvement / results_base[metric]) * 100 if results_base[metric] != 0 else 0
                arrow = "↑" if improvement > 0 else "↓"
                print(f"  {metric:15}: {improvement:+.4f} ({pct_improvement:+.1f}%) {arrow}")
    
    # Show sample responses
    if results_ft:
        print("\n" + "=" * 60)
        print("🧪 SAMPLE RESPONSES")
        print("=" * 60)
        
        for i in range(min(3, len(preds_ft))):
            print(f"\nInput: {refs_ft[i][0]}")
            print(f"Reference: {refs_ft[i][0]}")
            print(f"Fine-tuned: {preds_ft[i]}")
            print(f"Base: {preds_base[i]}")
            print("-" * 50)
    
    return results_ft, results_base

if __name__ == "__main__":
    run_complete_evaluation()