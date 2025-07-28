import streamlit as st
from gym_guide_chatbot.chatbot import ask_bot

# UI Setup
st.set_page_config(page_title="Gym Chatbot", page_icon="💪")
st.markdown("<h2 style='text-align: center;'>Welcome to the Gym Chatbot 💪</h2>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_query = st.chat_input("Ask me something about fitness, diet or workouts...")

# Display entire history first (including the latest if it exists)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append user message to session state
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Reserve space for assistant message and show spinner
    with st.chat_message("assistant"):
        with st.spinner("💬 Typing..."):
            # Get response from bot
            response = ask_bot(user_query, session_id="streamlit-session")

            # Append assistant response to session state
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display assistant response
            st.markdown(response)
