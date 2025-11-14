from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

# Create compact knowledge base for cloud deployment
documents = [
    Document(page_content="Artificial intelligence (AI) creates machines that simulate human intelligence and learning.", metadata={"category": "AI"}),
    Document(page_content="Machine learning (ML) enables computers to learn patterns from data without explicit programming.", metadata={"category": "ML"}),
    Document(page_content="Deep learning uses multi-layer neural networks to analyze complex data patterns and features.", metadata={"category": "DL"}),
    Document(page_content="Natural language processing (NLP) allows computers to understand, interpret and generate human language.", metadata={"category": "NLP"}),
    Document(page_content="Computer vision enables machines to identify, process and analyze visual information from the world.", metadata={"category": "CV"}),
    Document(page_content="Neural networks are computing systems inspired by biological neural networks in human brains.", metadata={"category": "NN"}),
    Document(page_content="Transformers are deep learning models that process sequential data using self-attention mechanisms.", metadata={"category": "Transformers"}),
    Document(page_content="Reinforcement learning trains AI agents through reward-based learning and decision making.", metadata={"category": "RL"}),
    Document(page_content="Supervised learning uses labeled datasets to train algorithms for classification and prediction.", metadata={"category": "SL"}),
    Document(page_content="Unsupervised learning finds hidden patterns in unlabeled data without human supervision.", metadata={"category": "UL"}),
]

print("🔄 Creating cloud-optimized vector database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embeddings)

# Create directory if it doesn't exist
os.makedirs("model/gemini-rag-small", exist_ok=True)

# Save to different folder for cloud deployment
vector_db.save_local("model/gemini-rag-small")
print("✅ Created cloud vector database at: model/gemini-rag-small/")
print("📊 Database size: Small & optimized for deployment")