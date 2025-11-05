# Databricks notebook source
# Install dependencies
%pip install -q langchain-databricks langchain-community pypdf

# COMMAND ----------

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pyspark.sql.functions import lit, monotonically_increasing_id

# COMMAND ----------

# Define paths (adjust to your Unity Catalog volume)
pdf_volume_path = "/Volumes/document_catalog/default/pdf_docs"
pdf_files = dbutils.fs.ls(pdf_volume_path)  # List all PDFs in volume

# Ingest and parse PDFs using PdfReader
documents = []
for file in pdf_files:
    if file.name.endswith(".pdf"):
        pdf_path = file.path.replace("dbfs:","")
        print("Reading path:", pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""  # Handle empty pages
                # Create a document-like object similar to PyPDFLoader output
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")

print("Chunking path:", pdf_path)

# Chunking: Use recursive splitter for semantic coherence
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Balance for embedding context (bge-large-en max 512 tokens)
    chunk_overlap=200,  # Overlap for context retention
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Add metadata (e.g., source file, page) and create DataFrame
chunk_data = [
    {"id": i, "text": chunk.page_content, "source": chunk.metadata.get("source", "unknown"), "page": chunk.metadata.get("page", 0)}
    for i, chunk in enumerate(chunks)
]
df = spark.createDataFrame(chunk_data)

# COMMAND ----------

# Save to Delta Table in Unity Catalog
delta_table_path = "document_catalog.default.document_chunks"
df.write.format("delta").mode("overwrite").saveAsTable(delta_table_path)

print(f"Chunked {len(chunks)} chunks and saved to Delta Table: {delta_table_path}")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- verify
# MAGIC select * from document_catalog.default.document_chunks
# MAGIC
# MAGIC -- debug
# MAGIC -- drop table document_catalog.default.document_chunks
