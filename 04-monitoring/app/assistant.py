from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json 
import time 
load_dotenv()
model = SentenceTransformer("cointegrated/rubert-tiny")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

connections.connect(
    host="localhost", 
    port="19530"
)
collection = Collection(name="course_info")
def search(field, vector, course):
    # Perform the search operation
    res = collection.search(
        anns_field=f"{field}", 
        filter=f"course == {course}",
        param={"metric_type": "IP", "params": {}},
        data=[vector],
        output_fields=["text_id", "text", "section", "question", "course", "id"], 
        limit=5,  # Max. number of search results to return
    )

    # Initialize an empty list to hold the results
    result_docs = []

    # Loop through the hits
    for hits in res:
        for hit in hits:
            # Append each hit as a dictionary containing the desired fields
            hit_dict = {
                "text_id": hit.entity.get("text_id"),
                "text": hit.entity.get("text"),
                "section": hit.entity.get("section"),
                "question": hit.entity.get("question"),
                "course": hit.entity.get("course"),
            }
            result_docs.append(hit_dict)
    
    # Return the list of result documents
    return result_docs

def question_vector_search(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return search('question_vector', v_q, course)
def build_prompt(query, search_result) : 
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the the CONTEXT. 
        Use only the facts from the CONTEXT when answering the QUESTION. 
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
    response = gemini_model.generate_content(prompt)
    return response.text
def llm(prompt, model_choice):
    start_time = time.time()
    if model_choice.startswith('ollama/'):
        response = ollama_client.chat.completions.create(
            model=model_choice.split('/')[-1],
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    elif model_choice.startswith('openai/'):
        response = openai_client.chat.completions.create(
            model=model_choice.split('/')[-1],
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
    elif model_choice.startswith('gemini/') :         
        model=model_choice.split('/')[-1]
        gemini_model = genai.GenerativeModel(model_name=model)
        response = gemini_model.generate_content(prompt)
        answer = response.text
        tokens = response.usage_metadata
        tokens = {
            'prompt_tokens': response.usage_metadata.prompt_token_count,
            'completion_tokens': response.usage_metadata.candidates_token_count,
            'total_tokens': response.usage_metadata.total_token_count
        }
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
    end_time = time.time()
    response_time = end_time - start_time
    return answer, tokens, response_time

def get_answer(query, course, model_choice, search_type):
    vector = model.encode(query)
    search_results = search('question_text_vector', vector, course)

    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model_choice)
    
    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    # openai_cost = calculate_openai_cost(model_choice, tokens)
 
    return {
        'answer': answer,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'model_used': model_choice,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
        # 'openai_cost': openai_cost
    }
def evaluate_relevance(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, 'gemini/gemini-1.5-flash-8b')
    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens