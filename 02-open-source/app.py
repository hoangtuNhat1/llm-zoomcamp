import openai
import streamlit as st
import json 
from elasticsearch import Elasticsearch
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',)
es_client = Elasticsearch("http://localhost:9200")
if es_client.ping():
    print("Connected to Elasticsearch")
else:
    print("Could not connect to Elasticsearch")
index_name = 'course-questions'     
def elastic_search(query) : 
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^5", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)
    result_docs = [] 
    for hit in response['hits']['hits'] : 
        result_docs.append(hit['_source'])
    return result_docs
def build_prompt(query, search_result) : 
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the the CONTEXT. 
        Use only the facts from the CONTEXT when answering the QUESTION. 
        If the CONTEXT doesn't contain the answer, output NONE 

        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()
    context = ""
    for doc in search_result: 
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
def llm(prompt):
    response = client.chat.completions.create(
        model='llama3.2',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
# Function to get response from GPT
def rag(query): 
    results = elastic_search(query)
    prompt = build_prompt(query, results)
    answer = llm(prompt)
    return answer

# Streamlit UI
st.title("ChatGPT with Streamlit")

st.write("This is a simple app to interact with ChatGPT.")

# Text input for user prompt
user_input = st.text_area("Enter your prompt here:", "")

# Button to trigger the GPT request
if st.button("Ask ChatGPT"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            # Get GPT response
            gpt_response = rag(user_input)
            st.write("**ChatGPT Response:**")
            st.write(gpt_response)
    else:
        st.write("Please enter a prompt!")

