import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import random

# 定义特殊 token
BEGIN_OF_THOUGHT = "<|begin_of_thought|>"
END_OF_THOUGHT = "<|end_of_thought|>"
BEGIN_OF_SOLUTION = "<|begin_of_solution|>"
END_OF_SOLUTION = "<|end_of_solution|>"

def generate_distilled_data(base_model, tokenizer, questions, num_rollouts=5):
    """
    使用 o1-like 模型生成蒸馏数据。

    Args:
        base_model: 基础模型（例如 Qwen2.5-32B-Instruct）
        tokenizer: 分词器
        questions: 问题列表
        num_rollouts: 每个问题生成的 rollout 次数

    Returns:
        一个列表，包含蒸馏数据，每个数据是一个字典，包含问题、思考过程、答案
    """
    distilled_data = []
    for question in questions:
        for _ in range(num_rollouts):
            inputs = tokenizer(question, return_tensors="pt").to(base_model.device)
            outputs = base_model.generate(**inputs, max_length=2048)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 假设o1-like模型会按格式输出，这里需要你自己根据实际情况做解析。
            try:
                thought_start = response.index(BEGIN_OF_THOUGHT) + len(BEGIN_OF_THOUGHT)
                thought_end = response.index(END_OF_THOUGHT)
                solution_start = response.index(BEGIN_OF_SOLUTION) + len(BEGIN_OF_SOLUTION)
                solution_end = response.index(END_OF_SOLUTION)

                thought = response[thought_start:thought_end].strip()
                solution = response[solution_start:solution_end].strip()

                distilled_data.append({
                    "question": question,
                    "thought": thought,
                    "solution": solution,
                })
            except ValueError:
                print(f"格式解析错误，跳过此轮： {response}")
                continue

    return distilled_data
def format_data_for_training(distilled_data):
    """
    格式化蒸馏数据，用于模型训练。
     Args:
        distilled_  蒸馏数据，每个数据是一个字典，包含问题、思考过程、答案
    Returns:
       一个列表，包含格式化的数据，每个数据是一个字符串，包含问题、思考过程和答案的拼接
    """
    formatted_data = []
    for item in distilled_data:
        formatted_prompt = f"""你的角色是一个助手，需要通过系统的长思考过程来解决问题，然后再给出最终的准确答案。

请将你的回答分为两个部分：思考过程和答案。
在思考过程中，请按照指定格式详细描述你的推理过程：

{BEGIN_OF_THOUGHT}
{{详细的思考步骤，每一步用 \\n\\n 分隔}}
{END_OF_THOUGHT}

每一步骤都应包含详细的思考过程，如分析问题、总结相关信息、提出新想法、验证步骤准确性、修正任何错误，以及回顾之前的步骤。
在答案部分，请根据你思考部分的尝试、探索和反思，系统地给出最终你认为正确的答案。答案应保持逻辑、准确、简洁的表达风格，并包含得出结论的必要步骤，格式如下：

{BEGIN_OF_SOLUTION}
{{最终格式化、准确和清晰的答案}}
{END_OF_SOLUTION}

现在，请根据以上指导，尝试解决以下问题：
{item['question']}
"""

        formatted_output = f"""{BEGIN_OF_THOUGHT}\n{item['thought']}\n{END_OF_THOUGHT}\n{BEGIN_OF_SOLUTION}\n{item['solution']}\n{END_OF_SOLUTION}"""

        formatted_data.append(
            {"input": formatted_prompt, "output": formatted_output}
        )
    return formatted_data




import os
from openai import OpenAI
# api_key="sk-c2262ab303d14ff78f97d9cfa1cdb4ab"
import ast
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-c2262ab303d14ff78f97d9cfa1cdb4ab", # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 通过 messages 数组实现上下文管理
messages = []

def respose_llm(prompt,i, ask=False):
    print("*" * 20 + f"第{i}轮对话" + "*" * 20)
    if ask:
        print("=" * 20 + "提问" + "=" * 20)
        print(prompt)
    messages.append({'role': 'user', 'content': prompt})
    completion = client.chat.completions.create(
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=messages,
        stream=True
    )
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering == False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content
    # 输出内容
    print("=" * 20 + "思考过程" + "=" * 20)
    print(reasoning_content)
    print("=" * 20 + "最终答案" + "=" * 20)
    print(answer_content)
    # 将内容添加型上下文当中
    messages.append({'role': 'assistant', 'content': answer_content})

    # return messages
