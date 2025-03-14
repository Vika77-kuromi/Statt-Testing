# -*- coding: utf-8 -*-
"""PolicyTesting1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LM816idhrM3U3fqcJazMxVnSnbV4C3aw
"""

from langchain_community.document_loaders import PyPDFLoader


# list of local PDF paths
local_pdf_paths = [
    "https://www.commerce.gov/sites/default/files/2025-03/DOC-Enterprise-Cybersecurity-Policy-1-1.pdf",
    "https://www.commerce.gov/sites/default/files/2025-03/Security-and-Privacy-Assessment-and-Authorization-Handbook-v1-1.pdf",
    'https://www.commerce.gov/sites/default/files/2022-02/OCIO-IT-Policy-Development-Policy.pdf',
    'https://www.commerce.gov/sites/default/files/2022-02/Internet-Protocol-Version-6-Policy.pdf',
    'https://www.commerce.gov/sites/default/files/2022-02/IT-Product-Maintenance-Support-End-of-Life-Cycle-Mgt.pdf',
    'https://www.commerce.gov/sites/default/files/2022-02/Controlled-Unclassified-Information-Policy.pdf'
]

pdf_docs =[]
# load PDFs into LangChain
for path in local_pdf_paths:
    loader = PyPDFLoader(path)
    pdf_docs.extend(loader.load())

# print to check
print(pdf_docs[0].page_content[:1000])

from langchain_text_splitters import RecursiveCharacterTextSplitter

# break the documents to different chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pdf_docs)

# check the number of chunks
len(all_splits)

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# load .env file with my api key
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    request_timeout=30,
    api_key=openai_api_key
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# use OpenAI's embedding model to convert text into vector embeddings
openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# convert the text documents (chunks) into a vectorstore using Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=openai_embeddings)

from typing_extensions import List, TypedDict
from langchain.prompts import PromptTemplate

# design the prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a policy assistant that provides concise and accurate answers based on official government documents.

Context:
{context}

Question: {question}

Provide a well-structured and informative answer below. Make sure to fully answer the question:

Answer:
"""
)


# define a dictionary structure to store application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# define the retrieve logic for the model
def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


# define the answer generation function
def generate(state: State):
    # combine all retrieved document text into a single string
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    formatted_prompt = prompt.format(question=state["question"], context=docs_content)
    response = llm.invoke(formatted_prompt)
    return {"answer": response}

from langgraph.graph import START, StateGraph

# built the workflow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

import gradio as gr

def chatbot_response(user_question):
    response = graph.invoke({"question": user_question})
    return response["answer"].content

# create a Gradio Interface
chatbot_ui = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask a policy question..."),
    outputs="text",
    title="Policy Chatbot",
    description="Ask questions about internet commerce policies.",
)


chatbot_ui.launch(share=True, inbrowser=True)