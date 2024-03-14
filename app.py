# 导入所需的库
from dataclasses import asdict

import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

from interface import GenerationConfig, generate_interactive

from modelscope import snapshot_download

logger = logging.get_logger(__name__) 

model_id = 'LocknLock/ft-ietls-speaking-assistant' 
mode_name_or_path = snapshot_download(model_id, revision='master')

def on_btn_click():
    del st.session_state.messages

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def load_model():
    model = (
        AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)

    return generation_config

user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"

def combine_history(prompt):
    messages = st.session_state.messages
    total_prompt = ""
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.replace("{user}", prompt)
    return total_prompt


def main():

    print("load model begin.")
    # load model，tokenizer
    model, tokenizer = load_model()
    print("load model end.")

    user_avator = "imgs/user.png"
    robot_avator = "imgs/robot.jpeg"

    st.title("🙋 Ielts Speaking Assistant")
    st.caption("A streamlit chatbot powered by InternLM2 QLora")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=103028,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()