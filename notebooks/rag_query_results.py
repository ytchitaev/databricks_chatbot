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

# Optionally suppress warnings globally (use cautiously)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### links

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Agent**
# MAGIC - Try this:
# MAGIC   - https://chatgpt.com/share/68e8b3ed-41f4-800f-b193-4d71e158f391 
# MAGIC   - https://grok.com/share/c2hhcmQtMw%3D%3D_7d9cace0-3cad-4694-b598-51452da51008
# MAGIC
# MAGIC **Reading**
# MAGIC - https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search#create-a-vector-search-endpoint-using-the-python-sdk
# MAGIC
# MAGIC **Sources**
# MAGIC - https://treasury.gov.au/sites/default/files/2025-10/p2025-702329-fr.pdf
# MAGIC - https://treasury.gov.au/publication/p2025-702329

# COMMAND ----------

# MAGIC %md
# MAGIC ### scrap

# COMMAND ----------

retriever_result = retriever.invoke(query, disable_notice=True)

# Convert Documents to rows
rows = [
    Row(
        id=doc.metadata.get('id'),
        source=doc.metadata.get('source'),
        content=doc.page_content
    )
    for doc in retriever_result
]

# Create Spark DataFrame
spark_df = spark.createDataFrame(rows)

# Show nicely formatted output
display(spark_df)

# COMMAND ----------


# def format_docs(docs):
#     return "\n\n".join(getattr(doc, "text", doc.page_content) for doc in docs)

# def debug_docs(docs):
#     for doc in docs:
#         print(doc.__dict__)  # Or print(doc.text, doc.page_content, doc.metadata)

# COMMAND ----------

# DBTITLE 1,Scrap
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-8b-instruct", temperature=0.1)

# Define RAG prompt
prompt = ChatPromptTemplate.from_template(
    """You are a document assistant. Answer based on this context: {context}
    \n\nQuestion: {question}"""
)

# Build RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    # {"context": retriever | format_docs, "question": RunnablePassthrough()}
    {"context": lambda q: format_docs(retriever.invoke(q, disable_notice=True)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example query (run interactively or in a loop)
# query = "What are some questions I can ask about this document?"
# query = "What were the key issues addressed in the discussion paper released by Treasury?"
# query = "List countries referenced"
# query = "What are the main recommendations"
# query = "Are there any references to Singapore?"
# query = "Is there anything the paper warns about?"
# query = "What is the title of this document?"
query = "What are some surprising findings?"

response = rag_chain.invoke(query)

print("Response:\n")
print(response)

print("\nSource:\n")
pprint([(doc.metadata["id"], doc.metadata["source"]) for doc in retriever.invoke(query, disable_notice=True)])
# pprint(f"Sources: {[doc.metadata['source'] for doc in retriever.invoke(query)]}")