from setuptools import setup, find_packages

setup(
    name="personalized_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'langchain-huggingface',
        'langchain-community',
        'google-generativeai',
        'faiss-cpu',
        'pandas',
        'numpy',
    ],
)