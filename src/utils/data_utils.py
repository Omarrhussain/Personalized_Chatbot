import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_chat_data(df, test_size=0.1):
    """Split data into train/validation sets"""
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )
    
    # Save splits
    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/validation.csv', index=False)
    
    return train_df, val_df

def format_conversation_template(input_text, response_text, template_type="llama"):
    """Format conversations based on model requirements"""
    templates = {
        "llama": f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response_text}<|eot_id|>",
        "deepseek": f"Human: {input_text}\nAssistant: {response_text}",
        "mixtral": f"<s>[INST] {input_text} [/INST] {response_text}</s>"
    }
    return templates.get(template_type, f"Human: {input_text}\nAssistant: {response_text}")