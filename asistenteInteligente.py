#!/usr/bin/env python
# coding: utf-8

# # RAG Quanam Challenge
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# ## Initialize LLM

# In[58]:

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
GENERATION_LLM_MODEL = os.getenv("GENERATION_LLM_MODEL")
EVALUATION_LLM_MODEL = os.getenv("EVALUATION_LLM_MODEL")

llm = ChatOpenAI(model=GENERATION_LLM_MODEL, api_key=OPEN_API_KEY)

# # Load JSON and create documents

# In[60]:

import json
from langchain.schema import Document 

with open("hotpotqa_docs_reduced.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Delete "answer" and "question" keys
cleaned_docs = [
    Document(page_content=doc["text"], metadata={"title": doc["title"]}) 
    for doc in raw_data
]

print(f"{len(cleaned_docs)} documents created.")
print("Document example:")
print(cleaned_docs[0])

# # Save embeddings in ChromaDB

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

#lighter alternative
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Index documents
vectorstore = Chroma.from_documents(cleaned_docs, embeddings)

print(f"Indexed {len(cleaned_docs)} documents in ChromaDB")

# # Generate answer with LLM

# In[ ]:

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel

output_parser = StrOutputParser()

def create_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """<?xml version="1.0" encoding="UTF-8"?>
        <Prompt>
            <Instructions>
                You are an intelligent assistant that answers questions using only the reference texts provided below.
                Use the most relevant information from these references to provide a concise and accurate answer.
            </Instructions>
            <Rules>
                <Rule>Do not add any external knowledge.</Rule>
                <Rule>Try to keep the answer under 5 words.</Rule>
                <Rule>Respond in the same language as the question and the data, English.</Rule>
                <Rule>Answer strictly what you are asked. Do not provide extra information.</Rule>
            </Rules>
            <Question>{query}</Question>
            <ReferenceTexts>{reference_texts}</ReferenceTexts>
        </Prompt>
        <Answer></Answer>
        """),
    ])

def create_chain(llm: BaseLanguageModel) -> BaseLanguageModel:
    prompt = create_prompt()
    return prompt | llm | output_parser

# # Simple FrontEnd

def get_answer(query: str) -> str:
    search_results = vectorstore.similarity_search_with_score(query, k=10) 

    reference_texts = "\n\n".join(
        [f"Title: {doc.metadata['title']}\nText: {doc.page_content}" for doc, _ in search_results]
    ) if search_results else "No relevant text found."

    chain = create_chain(llm)

    actual_answer = chain.invoke({"query": query, "reference_texts": reference_texts})

    return actual_answer

import streamlit as st

st.title("LangChain Q&A Interface")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        answer = get_answer(question)

        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question before clicking 'Ask'.")



