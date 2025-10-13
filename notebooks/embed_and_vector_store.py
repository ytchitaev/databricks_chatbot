# Databricks notebook source
# MAGIC %pip install -q databricks-langchain langchain-core

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks_langchain import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import array, to_json, col
import json

# COMMAND ----------

# Load Delta Table
delta_table = spark.table("document_catalog.default.document_chunks")

# Create embeddings using Mosaic AI endpoint
endpoint_name = "databricks-bge-large-en"  # Built-in, no setup needed
try:
    embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
    print(f"Using pre-provisioned endpoint: {endpoint_name}")
except Exception as e:
    if "ENDPOINT_NOT_FOUND" in str(e):
        endpoint_name = "bge_embedding"  # Fallback to custom
        embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
        print(f"Using custom endpoint: {endpoint_name}")
    else:
        raise e

# Compute embeddings in batch
chunk_texts = [row.text for row in delta_table.select("text").collect()]
embedded_chunks = embeddings.embed_documents(chunk_texts)

# Add embeddings to DataFrame and convert to JSON strings
pdf = delta_table.toPandas()
pdf["vector"] = embedded_chunks  # List of lists of floats
embedded_spark_df = spark.createDataFrame(pdf).withColumn("vector", col("vector").cast("array<float>"))

# Save embedded Delta Table
embedded_table_path = "document_catalog.default.document_embedded_chunks"
embedded_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "True").saveAsTable(embedded_table_path)

# Enable Change Data Feed on the Delta table
spark.sql("ALTER TABLE document_catalog.default.document_embedded_chunks SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# Set up Vector Search
vsc = VectorSearchClient(disable_notice=True)
endpoint_name_vs = "document_vector_endpoint"  # Vector Search endpoint
try:
    vsc.get_endpoint(name=endpoint_name_vs)
except:
    vsc.create_endpoint(name=endpoint_name_vs, endpoint_type="STANDARD")

# COMMAND ----------

# MAGIC %md
# MAGIC _Note: Vector store has spin-up time in Databricks._

# COMMAND ----------

index_name = "document_catalog.default.document_index"
index = vsc.create_delta_sync_index(
    endpoint_name="document_vector_endpoint",
    index_name=index_name,
    source_table_name=embedded_table_path,
    primary_key="id",
    embedding_vector_column="vector",  # Key fix: Use this for self-managed
    embedding_dimension=1024, # Ensure this matches the embedding model (BGE-large-en uses 1024)
    pipeline_type="TRIGGERED"
)

# debugging
# vsc.delete_index(endpoint_name="document_vector_endpoint", index_name="document_catalog.default.document_index")
