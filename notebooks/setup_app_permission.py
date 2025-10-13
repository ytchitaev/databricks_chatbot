# Databricks notebook source
# MAGIC %md
# MAGIC ### Notes
# MAGIC - Run after creating Databricks App.
# MAGIC - Replace UUID with your App Service Principal ID.

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT USE CATALOG ON CATALOG document_catalog TO `9e84e630-02bb-4f17-a27f-bbae383b8761`;

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT USE SCHEMA ON SCHEMA document_catalog.default TO `9e84e630-02bb-4f17-a27f-bbae383b8761`; 

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT SELECT ON TABLE document_catalog.default.document_index TO `9e84e630-02bb-4f17-a27f-bbae383b8761`;
# MAGIC GRANT SELECT ON TABLE document_catalog.default.document_chunks TO `9e84e630-02bb-4f17-a27f-bbae383b8761`;
# MAGIC GRANT SELECT ON TABLE document_catalog.default.document_embedded_chunks TO `9e84e630-02bb-4f17-a27f-bbae383b8761`;
