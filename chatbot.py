import streamlit as st
import pandas as pd

st.title("Cupid's Choice")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

data = pd.read_csv("cupid_chatbot.csv")



# React to user input
if prompt := st.chat_input("What is up?"):
    sent=prompt

    for i in range(len(data)):
        if data['keyword1'][i] in sent:
            output = data['response'][i]
            break

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = f"Cupid: {output}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})