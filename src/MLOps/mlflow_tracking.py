import mlflow
import json
from datetime import datetime
import os

class MLflowTracker:
    def __init__(self, experiment_name="gemini_rag_chatbot"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_chat_interaction(self, question: str, response: dict, response_time: float):
        """Log a single chat interaction"""
        with mlflow.start_run(run_name=f"chat_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("question", question)
            mlflow.log_metric("response_time", response_time)
            mlflow.log_metric("sources_used", response.get('sources_count', 0))
            mlflow.log_metric("success", int(response['success']))
            
            # Log response length
            if response['success']:
                answer_length = len(response['answer'].split())
                mlflow.log_metric("answer_length", answer_length)
    
    def log_evaluation_results(self, evaluation_results: dict):
        """Log comprehensive evaluation results"""
        with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d')}"):
            # Log summary metrics
            summary = evaluation_results.get('summary', {})
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log evaluation details
            mlflow.log_dict(evaluation_results, "evaluation_results.json")

# Quick test
def test_mlflow():
    tracker = MLflowTracker()
    sample_response = {
        'success': True,
        'answer': 'This is a test response',
        'sources_count': 2
    }
    tracker.log_chat_interaction("Test question", sample_response, 1.5)
    print("✅ MLflow test completed!")

if __name__ == "__main__":
    test_mlflow()