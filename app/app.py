# app.py
import streamlit as st
from utils import init_components
from langchain_core.messages import AIMessage, HumanMessage

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Document Query App")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.markdown("**Sources:**")
            for i, doc in enumerate(message["sources"], 1):
                with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)
                    st.caption(f"ID: {doc.metadata.get('id')}")

# Get user input
query = st.chat_input("Enter your query about the documents:")

if query:
    # Append user message to history and chat_history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.chat_history.append(HumanMessage(content=query))

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating response..."):
            try:
                retriever, llm, condense_prompt, answer_prompt = init_components()

                # Build conversational retriever (condenses question with history)
                from langchain.chains.history_aware_retriever import create_history_aware_retriever
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain.chains.retrieval import create_retrieval_chain

                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, condense_prompt
                )

                # Build QA chain (answers with history)
                qa_chain = create_stuff_documents_chain(llm, answer_prompt)

                # Combine into full RAG chain
                rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

                # Invoke with chat_history
                response_dict = rag_chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.chat_history
                })
                response = response_dict["answer"]
                docs = response_dict["context"]
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.stop()

            # Display response
            st.markdown(response)

            # Display sources
            st.markdown("**Sources:**")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)
                    st.caption(f"ID: {doc.metadata.get('id')}")

    # Append assistant message to history (including sources for preservation)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": docs
    })

    # Append AI message to chat_history
    st.session_state.chat_history.append(AIMessage(content=response))