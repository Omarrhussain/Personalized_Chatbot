import json
import pandas as pd
from typing import List, Dict
from src.data_preprocessing.preprocessing import DataPreprocessor

class PersonaChatProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
    
    def load_dataset(self):
        """Load dataset from JSON or CSV.

        - If the file is JSON, parse and return the JSON structure.
        - If the file is CSV, read into a list of dicts with columns as keys.
        """
        # Try JSON first
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception:
            # Not JSON — try CSV via pandas
            try:
                df = pd.read_csv(self.data_path)
                # Convert DataFrame rows to dicts for downstream processing
                return df.to_dict(orient='records')
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from {self.data_path}: {e}")
    
    def extract_conversations(self, data):
        """Extract conversations from the dataset structure"""
        conversations = []
        for item in data:
            # Persona-Chat specific structure
            if isinstance(item, dict) and 'dialog' in item:
                for utterance in item['dialog']:
                    if isinstance(utterance, dict) and 'text' in utterance:
                        conversations.append(utterance['text'])
            else:
                # Handle CSV-style rows where one column contains the multi-line chat
                if isinstance(item, dict):
                    chat_key = next((k for k in item.keys() if 'chat' in k.lower()), None)
                    if chat_key:
                        raw_chat = item.get(chat_key)
                        # Skip missing values (pandas may produce NaN floats)
                        if raw_chat is None:
                            continue
                        try:
                            import math
                            if isinstance(raw_chat, float) and math.isnan(raw_chat):
                                continue
                        except Exception:
                            pass

                        # Split multiline chats into utterances
                        lines = [ln.strip() for ln in str(raw_chat).splitlines() if ln.strip()]
                        conversations.extend(lines)
        return conversations
    
    def create_training_pairs(self, data):
        """Create input-output pairs for training"""
        training_pairs = []
        for item in data:
            dialog = None
            # Persona-Chat JSON-like structure
            if isinstance(item, dict) and 'dialog' in item and isinstance(item['dialog'], list):
                dialog = [utterance['text'] for utterance in item['dialog'] if isinstance(utterance, dict) and 'text' in utterance]
            else:
                # CSV-style row with a chat column
                if isinstance(item, dict):
                    chat_key = next((k for k in item.keys() if 'chat' in k.lower()), None)
                    if chat_key:
                        raw_chat = item.get(chat_key)
                        if raw_chat is None:
                            continue
                        try:
                            import math
                            if isinstance(raw_chat, float) and math.isnan(raw_chat):
                                continue
                        except Exception:
                            pass
                        dialog = [ln.strip() for ln in str(raw_chat).splitlines() if ln.strip()]

            if not dialog:
                continue

            # Create pairs: (input, response)
            for i in range(len(dialog) - 1):
                input_text = dialog[i]
                response_text = dialog[i + 1]
                training_pairs.append({
                    'input': input_text,
                    'response': response_text
                })
        return training_pairs
    
    def process_dataset(self):
        """Main processing pipeline"""
        raw_data = self.load_dataset()
        conversations = self.extract_conversations(raw_data)
        training_pairs = self.create_training_pairs(raw_data)
        
        # Preprocess all conversations
        processed_pairs = []
        for pair in training_pairs:
            processed_input = self.preprocessor.preprocess_conversation(pair['input'])
            processed_response = self.preprocessor.preprocess_conversation(pair['response'])
            
            if processed_input and processed_response:  # Skip empty
                processed_pairs.append({
                    'input': processed_input,
                    'response': processed_response
                })
        
        return processed_pairs