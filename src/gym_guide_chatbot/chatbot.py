from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from gym_guide_chatbot.prompt import intent_prompt, system_prompt
from gym_guide_chatbot.utils import download_embedding_model
import os

load_dotenv()

# ---- SETUP ----

# Initialize retriever
embeddings = download_embedding_model()
index_name = os.getenv("INDEX_NAME")

retriever = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
).as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chat LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# ---- PROMPTS ----

# Main system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Intent classifier prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", intent_prompt),
    ("user", "{input}")
])

# ---- CHAINS ----

# Final parser chain
chain: Runnable = prompt | llm | StrOutputParser()
intent_chain: Runnable = intent_prompt | llm | StrOutputParser()

# In-memory chat message store (per session_id)
chat_histories = {}  # session_id -> InMemoryChatMessageHistory

def get_message_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# Final memory-supported chain with context
chat_chain = RunnableWithMessageHistory(
    RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: x["context"],
        "chat_history": lambda x: x["chat_history"]
    }) | chain,
    lambda session_id: get_message_history(session_id),
    input_messages_key="input",
    history_messages_key="chat_history"
)

# ---- MAIN FUNCTION ----

def ask_bot(user_input: str, session_id: str) -> str:
    # Step 1: Classify intent
    intent = intent_chain.invoke({"input": user_input}).strip().lower()

    if intent == "smalltalk":
        # No retrieval needed
        context_text = ""
    elif intent == "fitness_query":
        # Get relevant context from retriever
        docs = retriever.invoke(user_input)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    else:
        # Fallback to empty context
        context_text = ""

    # Step 2: Run main chat chain
    return chat_chain.invoke(
        {
            "input": user_input,
            "context": context_text
        },
        config={"configurable": {"session_id": session_id}}
    )
