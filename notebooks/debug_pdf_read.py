# Databricks notebook source
# MAGIC %pip install pypdf

# COMMAND ----------

from pypdf import PdfReader

# COMMAND ----------

pdf_path = "/Volumes/document_catalog/default/pdf_docs/p2025-702329-fr.pdf"
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
print(text)