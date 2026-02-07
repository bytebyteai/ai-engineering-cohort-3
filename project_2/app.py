import pathlib, streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Customer Support Chatbot")
st.title("Customer Support Chatbot")

@st.cache_resource
def init_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = Ollama(model="gemma3:4b", temperature=0.1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
    )

chain = init_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("What is in your mind?")
if question:
    with st.spinner("Thinking..."):
        response = chain(
            {
                "question": question,
                "chat_history": st.session_state.history,   # <- supply it
            }
        )
    st.session_state.history.append((question, response["answer"]))


for user, bot in reversed(st.session_state.history):
    st.markdown(f"**You:** {user}")
    st.markdown(bot)
