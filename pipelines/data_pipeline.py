# data_pipeline.py
from src.data_preprocessing.data_cleaning import PersonaChatProcessor
from src.data_preprocessing.data_cleaning import PersonaChatProcessor

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False

# Download NLTK data first (run this once)
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords') 
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data downloaded successfully!")
except:
    print("NLTK data already downloaded or download failed")

# Run the pipeline
processor = PersonaChatProcessor(r'D:\Personalized_Chatbot\data\raw\personality.csv')
processed_data = processor.process_dataset()

# Ensure output folder exists and write CSV either with pandas or csv
import os
os.makedirs('data/processed', exist_ok=True)

if _HAS_PANDAS:
    df = pd.DataFrame(processed_data)
    df.to_csv('data/processed/cleaned_conversations.csv', index=False)
    print(f"✅ Processing complete! Saved {len(df)} conversation pairs.")
    print(f"Final dataset columns: {df.columns.tolist()}")
else:
    # Fallback: write with csv module
    import csv
    if processed_data:
        keys = list(processed_data[0].keys())
    else:
        keys = ['input', 'response']
    out_path = 'data/processed/cleaned_conversations.csv'
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in processed_data:
            writer.writerow(row)
    print(f"✅ Processing complete! Saved {len(processed_data)} conversation pairs (csv fallback).")