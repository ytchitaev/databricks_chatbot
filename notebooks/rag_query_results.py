# Databricks notebook source
# MAGIC %pip install -q databricks-langchain databricks-vectorsearch langchain-core

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch, ChatDatabricks, DatabricksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from databricks.vector_search.client import VectorSearchClient
from pprint import pprint
from pyspark.sql import Row
import warnings

# COMMAND ----------

# Suppress warnings globally
warnings.filterwarnings("ignore", message="Using a notebook authentication token")

# COMMAND ----------

# Initialize the VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

# Retrieve the index
vs_index = vsc.get_index(
    endpoint_name="document_vector_endpoint",
    index_name="document_catalog.default.document_index"
)

# Initialize the embedding model (same as used for document embeddings)
endpoint_name = "databricks-bge-large-en"  # Same endpoint used for document embeddings
try:
    embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
    print(f"Using pre-provisioned endpoint: {endpoint_name}")
except Exception as e:
    if "ENDPOINT_NOT_FOUND" in str(e):
        endpoint_name = "bge_embedding"  # Fallback to custom endpoint
        embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
        print(f"Using custom endpoint: {endpoint_name}")
    else:
        raise e

# Initialize the DatabricksVectorSearch retriever
retriever = DatabricksVectorSearch(
    endpoint="document_vector_endpoint",
    index_name="document_catalog.default.document_index",
    text_column="text",
    columns=["id", "text", "source"],
    embedding=embeddings
).as_retriever(search_kwargs={"k": 5})  # Top 5 chunks

# COMMAND ----------

def format_docs(docs):
    """Format documents by joining their page content with newlines."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, prompt, llm):
    """Build the RAG chain using the retriever, prompt, and LLM."""
    return (
        {
            "context": lambda q: format_docs(retriever.invoke(q, disable_notice=True)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def query_rag_chain(rag_chain, retriever, query: str):
    """Execute a query on the RAG chain and get response"""
    response = rag_chain.invoke(query)
    retriever_response = retriever.invoke(query, disable_notice=True)
    return response, retriever_response

def print_response(response):
    """Print the response from the RAG chain."""
    print("Response:\n")
    print(response)

def print_sources(retriever_response):
    """Print the sources from the retriever."""
    print("\nSource:\n")
    pprint([(doc.metadata["id"], doc.metadata["source"]) for doc in retriever_response])

def get_retriever_sources_list_df(retriever_response):
    """Invoke retriever and convert result sources list to a Spark DataFrame."""
    rows = [
        Row(
            id=doc.metadata.get('id'),
            source=doc.metadata.get('source'),
            content=doc.page_content
        )
        for doc in retriever_response
    ]
    return spark.createDataFrame(rows)

# COMMAND ----------

# DBTITLE 1,Get response
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-8b-instruct", temperature=0.1)
prompt = ChatPromptTemplate.from_template(
    """
    You are a document assistant. Answer based on this context: {context} 
    Question: {question}
    """
)
rag_chain = build_rag_chain(retriever=retriever, prompt=prompt, llm=llm)

# query = "What are some questions I can ask about this document?"
# query = "What were the key issues addressed in the discussion paper released by Treasury?"
# query = "List countries referenced and what the references are"
# query = "What are the main recommendations"
# query = "Are there any references to Singapore?"
query = "Is there anything the paper warns about?"
# query = "What is the title of this document?"
# query = "What are some surprising findings?"

response, retriever_response = query_rag_chain(
    rag_chain=rag_chain, 
    retriever=retriever, 
    query=query
)

print_response(response)
print_sources(retriever_response)

# COMMAND ----------

# DBTITLE 1,Sources list references
retriever_sources_df = get_retriever_sources_list_df(retriever_response)
display(retriever_sources_df)
