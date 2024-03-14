![image](https://github.com/Sixdes/Ielts-speaking-assistant/assets/84364070/31befa01-3125-4599-99b7-c96f2bddffea)# Ielts-speaking-assistant

## 简介


![Uploading IELTS COACH.png…]()


基于YouTube、b站等相关雅思口语模拟考试以及真实测试视频， 通过InternLM2微调得到的雅思口语测试助手。雅思口语测试助手旨在帮助模拟雅思口语测试与课程学习。

本项目将介绍关于数据获取、清洗、处理，使用InternLM2 微调、LMDeploy量化与推理，最后部署至 OpenXLab。

## openxlab模型

模型链接[ielts-speaking-assistant model](https://openxlab.org.cn/models/detail/LocknLock/ft-ietls-speaking-assistant/tree/main)

应用链接 [ielts-speaking-assistant](https://openxlab.org.cn/apps/detail/lumine/ielts-speaking-assistant)

## 数据集

从YouTube，b站爬取了200多个雅思口语对话模拟或者真实视频，从中提取对应音频，通过音频提取获得对话的原始文档，整理对话数据为 XTuner 多轮对话数据格式，如下：

```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

进一步清洗数据获得最终的多轮对话数据集，用于后续指令微调。



## 微调

基座模型：InternLM2-chat-7b

考虑到雅思口语测试过程中存在不同需求：

1. 模拟考官对使用者进行引导与提问
2. 帮助使用者进行提示与回答建议

分别针对不同的目标需求修改指令对话数据集，使用 XTuner 进行微调。

xtuner 安装：

```shell
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
```

### SFT训练

整理好数据后，即可进行微调。具体微调的 config 已经放置在 `train/config` 目录下，在安装好 xtuner 后可以进行训练：

```shell
xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
```

### 模型转换

将得到的 PTH 模型转换为 HuggingFace 模型，生成 Adapter 文件夹

```shell
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
```

### 模型合并

将 HuggingFace adapter 合并到大语言模型：

```shell
 xtuner convert merge \
     ${NAME_OR_PATH_TO_LLM} \
     ${NAME_OR_PATH_TO_ADAPTER} \
     ${SAVE_PATH} \
     --max-shard-size 2GB
```

可以通过运行接口文件或者通过 xtuner chat 进行模型对话：

```shell 
# 运行文件
python ./cli_demo.py 
# 加载 Adapter 模型对话（Float 16）
xtuner chat ${NAME_OR_PATH_TO_ADAPTER} --prompt-template internlm2_chat
```


## 部署

## 量化

## OpenCompass 评测

## 鸣谢

## 特别感谢
