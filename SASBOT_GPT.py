import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from residue import (load_documents_from_jsonl,
                     get_pinecone_index,
                     create_vector_store,
                     get_vector_store,
                     get_rag_chain
                     )

st.sidebar.link_button("get one @ Cohere 🔗", "https://openai.com/api/")
openai_api_key = st.text_input("password", type="password", label_visibility="collapsed")
# pinecone_api_key = st.text_input("password", type="password", label_visibility="collapsed")
# openai_api_key = st.secrets.get("openai.api_key")
pinecone_api_key = st.secrets.get("pinecone.api_key")

if openai_api_key and pinecone_api_key:
    use_serverless='true'
    pc = Pinecone(api_key=pinecone_api_key)

    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
    index_name = "sastrachat"  # change if desired
    namespace = "sasbot_pinecone"

    embeddings=OpenAIEmbeddings(api_key=openai_api_key)
    docs=load_documents_from_jsonl("fresh_chunk.jsonl")
    llm = ChatOpenAI(api_key=openai_api_key)
    # retriever = get_vector_store(pc,index_name,spec,docs,embeddings,namespace).as_retriever()
    chain = get_rag_chain(pc,index_name,spec,docs,embeddings,namespace,llm)
    # chain.invoke('what are areas of interests of ghousiya begum?')
    st.set_page_config(page_title="SASBOT", page_icon="🤖")
    st.title("SASBOT using GPT-3.5 LLM")
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
        if input_text:
            st.write(chain.invoke(input_text))
else:
    st.sidebar.error(f"Please enter a Valid KEY")
    st.stop()