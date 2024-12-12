import json
from  langchain.schema import Document
from typing import Iterable
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pinecone_api_key = ""

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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


def create_vector_store(docs, index_name, embeddings, namespace):
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace='sasbot_pinecone',
        pinecone_api_key=pinecone_api_key
    )
    docsearch = vectorstore.from_documents(docs, embeddings, index_name=index_name, namespace=namespace)
    return docsearch


def get_vector_store(pc: Pinecone, index_name: str, spec: str, docs, embeddings, namespace):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    # print(existing_indexes)
    if index_name not in existing_indexes:
        # print("creating new index")
        get_pinecone_index(pc, index_name, spec)
        doc_search = create_vector_store(pc, docs, index_name, embeddings, namespace)
    # index = get_pinecone_index(pc,index_name,spec)

    else:
        get_pinecone_index(pc, index_name, spec)
        # index = pc.Index(index_name)
        # print("using existing index")
        doc_search = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings,
                                                             namespace=namespace)
    return doc_search


def get_rag_chain(pc, index_name, spec, docs, embeddings, namespace, llm):
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = get_vector_store(pc, index_name, spec, docs, embeddings, namespace).as_retriever()
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser())
    return rag_chain