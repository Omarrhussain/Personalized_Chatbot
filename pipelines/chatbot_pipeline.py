"""
Chatbot Training Pipeline for Gemini RAG System
"""
from typing import List
import pandas as pd
import os
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_core.documents import Document

class ChatbotTrainingPipeline:
    def __init__(self, data_path: str, output_path: str = "models/gemini-rag"):
        self.data_path = data_path
        self.output_path = output_path
        
    def run_pipeline(self) -> bool:
        """Run complete training pipeline"""
        try:
            print("🚀 Starting Chatbot Training Pipeline...")
            
            # 1. Load and prepare data
            print("📊 Loading data...")
            data = self._load_data()
            
            # 2. Create documents
            print("📝 Creating documents...")
            documents = self._create_documents(data)
            
            # 3. Split into chunks
            print("✂️ Splitting documents...")
            chunks = self._split_documents(documents)
            
            # 4. Create vector database
            print("🔧 Creating vector database...")
            vector_db = self._create_vector_db(chunks)
            
            # 5. Save vector database
            print("💾 Saving vector database...")
            self._save_vector_db(vector_db)
            
            print("✅ Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            return False
    
    def _load_data(self) -> pd.DataFrame:
        """Load conversation data"""
        return pd.read_csv(self.data_path)
    
    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create LangChain documents from DataFrame"""
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=f"Question: {row['input']}\nAnswer: {row['response']}",
                metadata={'source': 'conversation_data'}
            )
            documents.append(doc)
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        return splitter.split_documents(documents)
    
    def _create_vector_db(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector database"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.from_documents(chunks, embeddings)
    
    def _save_vector_db(self, vector_db: FAISS):
        """Save vector database"""
        os.makedirs(self.output_path, exist_ok=True)
        vector_db.save_local(self.output_path)

if __name__ == "__main__":
    pipeline = ChatbotTrainingPipeline("data/processed/cleaned_conversations.csv")
    success = pipeline.run_pipeline()