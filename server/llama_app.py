import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_utils.agents.LlamaGPT import LlamaGPTAgent

# openai.api_key = st.secrets.openai_key
st.header("Chat with docs 💬 📚")


@st.cache_resource(show_spinner=False)
def get_agent():
    agent = LlamaGPTAgent.from_llm(
        documents=["./example/FoodnDrinksCatalogue.txt"],
        sys_msg="You are working for Delhi Tummy a north indian food delivery company. Your name is Akshat Singh and the conversation will happen over the call.",
        verbose=True,
        retriever="sentence-window",
    )

    return agent


if "messages" not in st.session_state.keys():  # Initialize the chat message history
    ai_greetings= "How are you today? I hope you're having a great day so far. My name is Akshat Singh and I'm calling from Delhi Tummy. We are a food delivery company that specializes in delicious North Indian food and street food. How can I assist you today?"

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": ai_greetings,
        }
    ]


agent = get_agent()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.human_step(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history