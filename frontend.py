from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
from llama_cpp import Llama
import time

load_dotenv()

if "prev_model_selected" not in st.session_state:
    st.session_state["prev_model_selected"] = None

llm_selected = st.selectbox(
   "Model:",
   ("OpenAI", "Phi", "Zephyr"),
   index=None,
   placeholder="Choose a Model...",
   label_visibility="hidden"
)

if llm_selected == "Phi" and st.session_state["prev_model_selected"] != "Phi":
    st.session_state["llm"] = Llama(model_path="./models/phi2.gguf", chat_format="llama-2")
    progress_text = "Loading Phi 2 model..."
    local_llm_loadbar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.05)
        local_llm_loadbar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    local_llm_loadbar.empty()
    st.session_state["prev_model_selected"] = "Phi"

if llm_selected == "Zephyr" and st.session_state["prev_model_selected"] != "Zephyr":
    st.session_state["llm"] = Llama(model_path="./models/zephyr.gguf", chat_format="zephyr", n_ctx=3096)
    progress_text = "Loading Zephyr model..."
    local_llm_loadbar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.08)
        local_llm_loadbar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    local_llm_loadbar.empty()
    st.session_state["prev_model_selected"] = "Zephyr"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if llm_selected == "OpenAI":
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
        else:
            for response in st.session_state["llm"].create_chat_completion(
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            ):
                # print(response)
                # st.write(response)
                if 'content' in response['choices'][0]['delta']:
                    full_response += (response['choices'][0]['delta']['content'] or "")
                else:
                    continue
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})