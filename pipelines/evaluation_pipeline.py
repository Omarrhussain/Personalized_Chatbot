import pandas as pd
from src.model.gemini_rag_system import GeminiRAGSystem
import json
from datetime import datetime

class EvaluationPipeline:
    def __init__(self):
        self.chatbot = GeminiRAGSystem()
        self.test_questions = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What is deep learning?",
            "How do neural networks work?"
        ]
    
    def run_evaluation(self):
        print("🧪 Starting Evaluation Pipeline...")
        print(f"Testing {len(self.test_questions)} questions\n")
        results = []
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"[{i}/{len(self.test_questions)}] Testing: {question}")
            try:
                result = self.chatbot.ask_question(question)
                results.append({
                    'question': question,
                    'answer': result['answer'],
                    'success': result['success'],
                    'sources_used': result.get('sources_count', 0)
                })
                print(f"✅ {'Success' if result['success'] else 'Failed'}\n")
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'success': False,
                    'sources_used': 0
                })
                print(f"❌ Error: {str(e)}\n")
        
        # Save results
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_questions': len(results),
                'success_rate': sum(1 for r in results if r['success']) / len(results) * 100
            }
        }
        
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Evaluation complete! Success rate: {report['summary']['success_rate']:.1f}%")

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.run_evaluation()