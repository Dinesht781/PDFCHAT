import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import json
from typing import Iterable
from  langchain.schema import Document
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time
import os
from dotenv import load_dotenv
load_dotenv()


def load_documents_from_jsonl(file_path: str) -> Iterable[Document]:
    """Loads documents from a JSONL file."""
    documents = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            documents.append(obj)
    return documents


def get_pinecone_index(pc: Pinecone, index_name: str, spec: str, dimension=1536, metric="cosine"):
    """Creates a Pinecone index if it doesn't exist."""
    # index=index_name
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec,
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    index_name = pc.Index(index_name)
    # print(pc.describe_index(index))
    

def create_vector_store(docs,index_name,embeddings,namespace):
    vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace='sasbot_pinecone',
    pinecone_api_key=pinecone_api_key
)
    docsearch = vectorstore.from_documents(docs,embeddings,index_name=index_name, namespace=namespace)
    return docsearch
def get_vector_store(pc:Pinecone,index_name:str,spec:str,docs,embeddings,namespace):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    # print(existing_indexes)
    if index_name not in existing_indexes:
        # print("creating new index")
        get_pinecone_index(pc,index_name,spec)
        doc_search = create_vector_store(pc,docs,index_name,embeddings,namespace)
    # index = get_pinecone_index(pc,index_name,spec)
        
    else:
        get_pinecone_index(pc,index_name,spec)
        # index = pc.Index(index_name)
        # print("using existing index")
        doc_search = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings,namespace=namespace)
    return doc_search
        
def get_rag_chain(pc,index_name,spec,docs,embeddings,namespace,llm):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """
        )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    retriever = get_vector_store(pc,index_name,spec,docs,embeddings,namespace).as_retriever()
    # rag_chain = (
    # {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # | prompt
    # | llm
    # | StrOutputParser())
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain
openai_api_key = st.secrets.get("openai.api_key")
pinecone_api_key = st.secrets.get("pinecone.api_key")
groq_api_key=os.environ['GROQ_API_KEY']
use_serverless='true'
pc = Pinecone(api_key=pinecone_api_key)

if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')
index_name = "sastrachat"  # change if desired
namespace = "sasbot_pinecone"

embeddings=OpenAIEmbeddings(api_key=openai_api_key)
docs=load_documents_from_jsonl("fresh_chunk.jsonl")
llm = ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")
# retriever = get_vector_store(pc,index_name,spec,docs,embeddings,namespace).as_retriever()
chain = get_rag_chain(pc,index_name,spec,docs,embeddings,namespace,llm)
# chain.invoke('what are areas of interests of ghousiya begum?')
st.set_page_config(page_title="SASBOT", page_icon="🤖")
st.title("SASBOT using Mistral-8x7b LLM")
with st.sidebar:
    sidebar=st.sidebar
    sidebar.title("About")
    # sidebar.markdown(f"[Google](https://www.google.com/)")
    sidebar.header('Developed By')
    sidebar.markdown(f"[Dinesh Tippavarjula](https://www.linkedin.com/in/tippavarjula-dinesh/)")
    sidebar.markdown(f"[Sai Pavan Pasupuleti](https://www.linkedin.com/in/sai-pavan-pasupuleti-78254b248/)")
    # Add a link to the sidebar using a button
    # st.sidebar.button("Wikipedia", on_click=lambda x: st.sidebar.text("https://www.wikipedia.org/(https://www.wikipedia.org/)"))
    
message = st.chat_message("assistant")
message.write("Hello SASTRAite")
input_text=st.text_input("ask your question here")
if input_text:
    if len(input_text)>6:
        st.write(chain.invoke({"input":input_text})['answer'])