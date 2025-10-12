# utils.py
import os
import streamlit as st
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks, DatabricksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests

@st.cache_resource(show_spinner=False)  # Hide the spinner
def init_components():

    # Fetch access token using service principal (OAuth M2M flow for AWS Databricks)
    workspace_url = os.environ.get('DATABRICKS_HOST')
    token_url = f"{workspace_url}/oidc/v1/token"
    client_id = os.environ.get('DATABRICKS_CLIENT_ID')
    client_secret = os.environ.get('DATABRICKS_CLIENT_SECRET')
    try:
        # Get databricks token for internal auth
        response = requests.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "all-apis"
            }
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        os.environ['DATABRICKS_TOKEN'] = token
    
        # Remove OAuth env vars to avoid auth method conflict
        os.environ.pop('DATABRICKS_CLIENT_ID', None)
        os.environ.pop('DATABRICKS_CLIENT_SECRET', None)
    except Exception as e:
        st.error(f"Failed to fetch access token: {str(e)}")
        raise

    embeddings_endpoint = os.environ.get('EMBEDDING_ENDPOINT', 'databricks-bge-large-en')
    try:
        embeddings = DatabricksEmbeddings(endpoint=embeddings_endpoint)
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}, falling back to 'bge_embedding'")
        embeddings = DatabricksEmbeddings(endpoint='bge_embedding')  # Fallback

    try:
        retriever = DatabricksVectorSearch(
            endpoint=os.environ.get('VECTOR_SEARCH_ENDPOINT'),
            index_name=os.environ.get('INDEX_NAME'),
            text_column="text",
            columns=["id", "text", "source"],
            embedding=embeddings
        ).as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"DatabricksVectorSearch initialization failed: {str(e)}")
        raise

    llm = ChatDatabricks(
        endpoint=os.environ.get('LLM_ENDPOINT'),
        temperature=0.1
    )

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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever, llm, condense_prompt, answer_prompt