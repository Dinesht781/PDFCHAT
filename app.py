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
from langchain_community.vectorstores import chroma
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import streamlit as st
from langchain_openai import OpenAIEmbeddings
## Load modules from residue.py
from langchain_community.vectorstores import Chroma

from residue import save_docs_to_jsonl, load_docs_from_jsonl,format_docs

## Instead of load_dotenv visit to see how you can handle secrets in streamlit
openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

documents = load_docs_from_jsonl("fresh_chunk.jsonl")
db = Chroma.from_documents(documents, OpenAIEmbeddings(api_key=openai_api_key))
retriever = db.as_retriever()


def main():
    ## Title | Introduction
    st.title("SASBOT using GPT-3.5 LLM")
    message = st.chat_message("assistant")
    message.write("Hello SASTRAite")

    ## User Input OpenAI key
    st.header(f"Set your OpenAI API Key")
    st.sidebar.link_button("get one @ Cohere 🔗", "https://openai.com/api/")
    openai_api_key = st.text_input("password", type="password", label_visibility="collapsed")
    if openai_api_key:
        llm = ChatOpenAI(api_key=openai_api_key)
        input_text = st.text_input("ask your question here")

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        if input_text:
            st.write(rag_chain.invoke(input_text))
    else:
        st.sidebar.error(f"Please enter a Valid KEY")
        st.stop()


if __name__ == "__main__":
    main()