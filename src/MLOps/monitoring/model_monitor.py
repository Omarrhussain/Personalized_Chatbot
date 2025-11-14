import json
import pandas as pd
from datetime import datetime, timedelta
import os

class ModelMonitor:
    def __init__(self, log_file="monitoring/logs/chat_logs.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_interaction(self, question: str, response: dict, response_time: float):
        """Log chat interaction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response_length': len(response.get('answer', '')),
            'response_time': response_time,
            'sources_used': response.get('sources_count', 0),
            'success': response['success'],
            'error': '' if response['success'] else response.get('answer', '')
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        try:
            # Read logs from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    log = json.loads(line)
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if log_time >= cutoff_time:
                        logs.append(log)
            
            if not logs:
                return {"message": "No logs in the last 24 hours"}
            
            df = pd.DataFrame(logs)
            
            report = {
                'report_date': datetime.now().isoformat(),
                'total_interactions': len(df),
                'success_rate': (df['success'].sum() / len(df)) * 100,
                'avg_response_time': df['response_time'].mean(),
                'avg_sources_used': df['sources_used'].mean(),
                'avg_response_length': df['response_length'].mean(),
                'error_count': len(df[df['success'] == False])
            }
            
            # Save report
            report_file = f"monitoring/reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except FileNotFoundError:
            return {"error": "No log file found"}
        except Exception as e:
            return {"error": str(e)}

# Quick test
def test_monitoring():
    monitor = ModelMonitor()
    sample_response = {
        'success': True,
        'answer': 'Test response',
        'sources_count': 2
    }
    monitor.log_interaction("Test question", sample_response, 1.2)
    
    report = monitor.generate_daily_report()
    print("📊 Monitoring Report:", report)

if __name__ == "__main__":
    test_monitoring()