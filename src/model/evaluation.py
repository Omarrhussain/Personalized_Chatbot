# working_evaluation.py
import os
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def evaluate_chatbot():
    print("🚀 Starting Chatbot Evaluation")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv(r"D:\Personalized_Chatbot\data\processed\cleaned_conversations.csv")
        test_samples = df.sample(min(25, len(df)))
        print(f"📊 Loaded {len(test_samples)} test samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Find latest checkpoint
    model_dir = "D:\Personalized_Chatbot\models\fine-tuned-chatbot"
    checkpoints = []
    
    for item in os.listdir(model_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(model_dir, item)))
            except:
                continue
    
    if checkpoints:
        latest_step, adapter_path = max(checkpoints, key=lambda x: x[0])
        print(f"🎯 Using checkpoint: {adapter_path} (step {latest_step})")
    else:
        print("❌ No checkpoints found")
        return
    
    # Load models
    print("🔄 Loading models...")
    
    # Base model
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    
    # Fine-tuned model with LoRA
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    
    # Evaluation function
    def generate_response(model, input_text):
        prompt = f"Human: {input_text}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()
    
    # Generate responses
    print("🔄 Generating responses...")
    
    base_predictions = []
    ft_predictions = []
    references = []
    
    for idx, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
        input_text = row['input']
        reference = row['response']
        
        base_response = generate_response(base_model, input_text)
        ft_response = generate_response(ft_model, input_text)
        
        base_predictions.append(base_response)
        ft_predictions.append(ft_response)
        references.append([reference])
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    
    # Base model metrics
    base_bleu = bleu.compute(predictions=base_predictions, references=references)
    base_rouge = rouge.compute(predictions=base_predictions, references=references)
    
    # Fine-tuned model metrics
    ft_bleu = bleu.compute(predictions=ft_predictions, references=references)
    ft_rouge = rouge.compute(predictions=ft_predictions, references=references)
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\n🔵 BASE MODEL:")
    print(f"  BLEU:  {base_bleu['bleu']:.4f}")
    print(f"  ROUGE-1: {base_rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {base_rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {base_rouge['rougeL']:.4f}")
    
    print(f"\n🎯 FINE-TUNED MODEL:")
    print(f"  BLEU:  {ft_bleu['bleu']:.4f}")
    print(f"  ROUGE-1: {ft_rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {ft_rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {ft_rouge['rougeL']:.4f}")
    
    # Show samples
    print(f"\n🧪 SAMPLE RESPONSES")
    print("=" * 30)
    for i in range(min(3, len(base_predictions))):
        print(f"\nInput: {test_samples['input'].iloc[i]}")
        print(f"Reference: {references[i][0]}")
        print(f"Base: {base_predictions[i]}")
        print(f"Fine-tuned: {ft_predictions[i]}")

if __name__ == "__main__":
    evaluate_chatbot()