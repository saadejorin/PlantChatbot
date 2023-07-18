import streamlit as st
from huggingface_hub import InferenceClient
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import requests
import os
from time import sleep

#AVATARS
av_us = '👨‍🌾'
av_ass = '🤖'

# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory.txt', 'a') as f:
        f.write(text)
        f.write('\n')

# Set HF API token
yourHFtoken = os.getenv('HUGGINGFACE_TOKEN') 
repo="HuggingFaceH4/starchat-beta"

# Initialize the model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = yourHFtoken
llm = HuggingFaceHub(repo_id=repo,
                     model_kwargs={"min_length":30,
                                   "max_new_tokens":1000, "do_sample":True,
                                   "temperature":0.2, "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})
prompt = PromptTemplate(template="{myprompt}\n", input_variables=["myprompt"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# START STREAMLIT UI
st.markdown("<h1 style='text-align: center; color: black;'>🌱PlantAI ChatBot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray; margin-top: -30px;'><i>using Starchat-beta</i></h3>", unsafe_allow_html=True)

if st.button("What are common plant crop diseases?"): myprompt = "What are common plant crop diseases?"
if st.button("How does a plant's immune system work?"): myprompt = "How does a plant's immune system work?"
if st.button("What causes leaf yellowing in plants?"): myprompt = "What causes leaf yellowing in plants?"
if st.button("How do nutrients affect plant growth?"): myprompt = "How do nutrients affect plant growth?"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])

# Accept user input
if myprompt := st.chat_input("What are common plant disease?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": myprompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar=av_us):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
        writehistory(usertext)
        # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        res  =  llm_chain.run(myprompt) # Using the pre-loaded llm_chain
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        writehistory(asstext)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
