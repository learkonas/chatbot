import numpy as np
import openai
import pandas as pd
import tiktoken

from env import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
import json
with open('document_embeddings_SEND.json', 'r') as file:
 loaded_data = json.load(file)
document_embeddings = {eval(key): value for key, value in loaded_data.items()}

count = 1

COMPLETIONS_MODEL = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL = "text-embedding-3-small" #alt: "text-embedding-ada-002"
message = "Hi! What would you like to ask about the content?\n"

df = pd.read_csv("./send_csv.csv", encoding = 'cp850')
#df = df.set_index(["Title", "Content"])

#print(f"{len(df)} articles in the data.")

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> 'list[float]':
    #print(f"getting embedding")
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> "dict['tuple[str, str]', 'list[float]']":
    print(f"computing doc embedding")
    return {
        idx: get_embedding(r.Content) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str) -> "dict['tuple[str, str]', 'list[float]']":
    print(f"loading embeddings for ${fname}")
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "Title"])
    return {
           (r.Title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

#document_embeddings = compute_doc_embeddings(df)

str_document_embeddings = {str(key): value for key, value in document_embeddings.items()}
with open('document_embeddings_SEND.json', 'w') as file:
    json.dump(str_document_embeddings, file)

def vector_similarity(x: 'list[float]', y: 'list[float]') -> float:
    global count
    #print(f"calculating vector similarity: Count: {count}")
    count = count + 1
    # Returns the similarity between two vectors. Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: 'dict[(str, str), np.array]') -> 'list[(float, (str, str))]':
    #Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    #to find the most relevant sections. 
    #print(f"ordering document sections by similarity for: {query}")
    #Return the list of document sections, sorted by relevance in descending order.
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

#order_document_sections_by_query_similarity(question, document_embeddings)[:5]

MAX_SECTION_LEN = 2500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    #print(f"constructing prompt")
    #Fetch relevant 
    
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.Content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    ## Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    #print("\n".join(chosen_sections_indexes))
    
    header = """You will provide me with answers from the given context below. Directly quote from the context wherever possible and cite the bullet point you are referencing. Answer the question as truthfully as possible using the context, in a warm, educated manner.\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0, # We use temperature close to 0.0 because it gives the most predictable, factual answer.
    "max_tokens": 2500,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: 'dict[(str, str), np.array]',
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    #print(f"answering query with context")
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


def get_question(message: str, answer:str):
    count = 0
    question = input(f"{message}")
    question = answer + " " + question
    answer = answer_query_with_context(question, df, document_embeddings)
    print(answer, "\n")
    message = "What else would you like to ask?\n"
    get_question(message, answer)

get_question(message, "Thanks.")