# Databricks notebook source
# MAGIC %md
# MAGIC ### Notes
# MAGIC - Run as first step to create catalog / schema / volumes for data stores.

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS document_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS document_catalog.default")
spark.sql("CREATE VOLUME IF NOT EXISTS document_catalog.default.pdf_docs")
