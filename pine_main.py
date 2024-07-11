# import subprocess
# import sys

# def install_dependencies():
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# # Install dependencies
# install_dependencies()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_community.vectorstores import chroma
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import streamlit as st
import os
from  langchain.schema import Document
import json
from typing import Iterable
from langchain_openai import OpenAIEmbeddings
import time
# from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
load_dotenv()
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["pinecone"]["PINECONE_API_KEY"]

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





# openai_api_key=os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.environ.get("PINECONE_API_KEY")



# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


use_serverless='true'
pc = Pinecone(api_key=pinecone_api_key)
# import time
if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  

index_name = "chatsas"  # change if desired
namespace = "sasbot_pinecone"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec,
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

from langchain_pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore

embeddings=OpenAIEmbeddings(api_key=openai_api_key)
# vectorstore = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name,namespace="sasbot_pine")
vectorstore = Pinecone(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)


docsearch = vectorstore.from_documents(documents, embeddings, index_name=index_name,namespace=namespace)
retriever = vectorstore.as_retriever()

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
# print(rag_chain.invoke('what are the buildings present in SASTRA universiyty?'))