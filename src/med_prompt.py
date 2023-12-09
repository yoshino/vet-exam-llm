import chromadb
import openai
from dotenv import load_dotenv

load_dotenv(verbose=True)


openai_client = openai.OpenAI()

cot_template_chromadb_path = "../chromadb/"
client = chromadb.PersistentClient(path=cot_template_chromadb_path)

collection = client.get_collection(
    name="cot_template",
)


def chat_completions(messages):
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response


USER_PROMPT = """
{cot_template1}

{cot_template2}

{cot_template3}

## Question: {question}
{answer_choices}

## Answer
model generated chain of thought explanation
Therefore, the answer is [final model answer (e.g. 1,2,3,4,5)]
"""


def build_messages(question, answer_choices, cot_template1, cot_template2, cot_template3):
    return [
        {
            "role": "user",
            "content": USER_PROMPT.format(
                question=question,
                answer_choices=answer_choices,
                cot_template1=cot_template1,
                cot_template2=cot_template2,
                cot_template3=cot_template3
            )
        }

    ]


def answer_with_cot_template(question, answer_choices):
    embedding = openai_client.embeddings.create(input = [question], model="text-embedding-ada-002").data[0].embedding
    
    cot_template = collection.query(
        query_embeddings=[embedding],
        n_results=3,
    )

    response = chat_completions(
        messages=build_messages(
            question=question,
            answer_choices=answer_choices,
            cot_template1=cot_template['documents'][0][0],
            cot_template2=cot_template['documents'][0][1],
            cot_template3=cot_template['documents'][0][2],
        )
    )
    return response


MED_PROMPT = """
Considering the following three responses, please make a decision by majority vote.

## Answer1
{response1}

## Answer2
{response2}

## Answer3
{response3}
"""

def build_med_prompt_messages(response1, response2, response3):
    return [
        {
            "role": "user",
            "content": MED_PROMPT.format(
                response1=response1,
                response2=response2,
                response3=response3,
            )
        }

    ]

def answer_with_med_prompt(question, answer_choices):
    responses = []
    for _ in range(3): # MedPrompt recommends 5 times
        response = answer_with_cot_template(question, answer_choices)
        responses.append(response.choices[0].message.content)

    response = chat_completions(
        messages=build_med_prompt_messages(
            response1=responses[0],
            response2=responses[1],
            response3=responses[2],
        )
    )
    return response
