import streamlit as st
from langchain_llama.agent.sales_gpt import SalesGPT

# openai.api_key = st.secrets.openai_key
st.header("Chat with docs 💬 📚")


@st.cache_resource(show_spinner=False)
def get_agent():
    agent = SalesGPT.from_llm(
        documents=["./example/FoodnDrinksCatalogue.txt"],
        sys_msg="You are working for Delhi Tummy a north indian food delivery company. Your name is Akshat Singh and the conversation will happen over the call.",
        verbose=True,
        retriever="llamaindex-sentence_window",
        use_tools = True,
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
            agent.human_step(prompt)
            response = agent._call({})
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history