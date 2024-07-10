import subprocess
import sys

def install_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Install dependencies
install_dependencies()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import chroma
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import streamlit as st
import os
from  langchain.schema import Document
import json
from typing import Iterable
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# load_dotenv()
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
# save_docs_to_jsonl(documents,'chunk_data.jsonl')


documents=load_docs_from_jsonl("fresh_chunk.jsonl")


db = Chroma.from_documents(documents,OpenAIEmbeddings(api_key=openai_api_key))
retriever=db.as_retriever()


# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# api_key = openai_api_key
## Langmith tracking
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


st.title("SASBOT using GPT-3.5 LLM")
message = st.chat_message("assistant")
message.write("Hello SASTRAite")
input_text=st.text_input("ask your question here")

llm=ChatOpenAI(api_key=openai_api_key)

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if input_text:
    st.write(rag_chain.invoke(input_text))