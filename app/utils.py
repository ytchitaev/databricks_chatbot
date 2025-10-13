import os
import streamlit as st
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks, DatabricksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests

@st.cache_resource(show_spinner=False)
def init_components():

    # get embeddings
    try:
        embeddings = DatabricksEmbeddings(endpoint=os.environ.get('EMBEDDING_ENDPOINT'))
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}'")

    # get retriever
    try:
        retriever = DatabricksVectorSearch(
            endpoint=os.environ.get('VECTOR_SEARCH_ENDPOINT'),
            index_name=os.environ.get('INDEX_NAME'),
            text_column="text",
            columns=["id", "text", "source"],
            embedding=embeddings,
            client_args={
                "workspace_url": os.environ.get('DATABRICKS_HOST'),
                "service_principal_client_id": os.environ.get('DATABRICKS_CLIENT_ID'),
                "service_principal_client_secret": os.environ.get('DATABRICKS_CLIENT_SECRET')
            }
        ).as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"DatabricksVectorSearch initialization failed: {str(e)}")
        raise
    
    # instantiate chatbot
    llm = ChatDatabricks(
        endpoint=os.environ.get('LLM_ENDPOINT'),
        temperature=0.1
    )

    # prompts based on context
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and follow-up question, rephrase to a standalone question."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document assistant. Answer based on this context: {context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    return retriever, llm, condense_prompt, answer_prompt