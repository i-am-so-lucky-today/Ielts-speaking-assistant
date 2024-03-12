# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st

from modelscope import snapshot_download

model_id = 'LocknLock/ft-ietls-speaking-assistant'
mode_name_or_path = snapshot_download(model_id, revision='master')

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)

    return generation_config

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()  
    return tokenizer, model

def main():

    print("load model begin.")
    # load model，tokenizer
    tokenizer, model = get_model()
    print("load model end.")

    user_avator = "imgs/user.png"
    robot_avator = "imgs/robot.jpeg"

    st.title("🙋 Ielts Speaking Assistant")
    st.caption("A streamlit chatbot powered by InternLM2 QLora")
    # generation_config = prepare_generation_config()
    # 侧边栏
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # 遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message("user").write(msg[0])
        st.chat_message("assistant").write(msg[1])

    # 如果用户在聊天输入框中输入了内容，则执行以下操作
    if prompt := st.chat_input():
        # 在聊天界面上显示用户的输入
        st.chat_message("user").write(prompt)
        # 构建输入     
        response, history = model.chat(tokenizer, prompt, meta_instruction=system_prompt, history=st.session_state.messages)
        # 将模型的输出添加到session_state中的messages列表中
        st.session_state.messages.append((prompt, response))
        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(response)

    return

if __name__ == "__main__":
    main()