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


def create_dataset(formatted_data, tokenizer):
    """
    创建 PyTorch Dataset 对象。

    Args:
        formatted_ 格式化后的数据列表
        tokenizer: 分词器

    Returns:
        PyTorch Dataset 对象
    """
    input_texts = [item['input'] for item in formatted_data]
    output_texts = [item['output'] for item in formatted_data]

    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = tokenizer(output_texts, return_tensors="pt", padding=True, truncation=True)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs

        def __len__(self):
            return len(self.inputs["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.inputs["input_ids"][idx],
                "attention_mask": self.inputs["attention_mask"][idx],
                "labels": self.outputs["input_ids"][idx],
            }

    return CustomDataset(inputs, outputs)


def train_model(model, tokenizer, dataset, output_dir="sft_model"):
    """
    使用 SFT 微调模型。

    Args:
        model: 基础模型
        tokenizer: 分词器
        dataset: PyTorch Dataset 对象
        output_dir: 输出模型目录
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)


def rollout_and_check(model, tokenizer, question, ground_truth_answer, max_rollouts=20):
    """
    使用多次采样（rollout）和答案比对来验证输出。

     Args:
        model: 基础模型
        tokenizer: 分词器
        question: 问题
        ground_truth_answer:  正确答案
        max_rollouts: 最大 rollout 次数

    Returns:
        如果找到正确答案，则返回对应的思考过程和答案；否则返回 None
    """
    for _ in range(max_rollouts):
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=2048)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            thought_start = response.index(BEGIN_OF_THOUGHT) + len(BEGIN_OF_THOUGHT)
            thought_end = response.index(END_OF_THOUGHT)
            solution_start = response.index(BEGIN_OF_SOLUTION) + len(BEGIN_OF_SOLUTION)
            solution_end = response.index(END_OF_SOLUTION)

            thought = response[thought_start:thought_end].strip()
            solution = response[solution_start:solution_end].strip()

            if solution == ground_truth_answer:
                return {"thought": thought, "solution": solution}
        except ValueError:
            print(f"格式解析错误，跳过此轮： {response}")
            continue

    return None  # 如果所有 rollout 都没找到正确答案，返回 None


def dpo_training(model, tokenizer, dataset, output_dir='dpo_model', num_epochs=2):
    """
  使用 DPO 微调模型。
  Args:
      model: 基础模型
      tokenizer: 分词器
      dataset: 数据集
      output_dir: 输出模型目录
      num_epochs: 训练轮数
  """
    print("DPO Training Placeholder: 由于DPO训练代码较为复杂，此处仅为占位符")
    #  在这里实现 DPO 训练， 包括选择正负样本，计算 DPO loss,  更新模型
    # 这部分代码较为复杂，需要根据实际情况进行编写，这里先不做详细实现
    print("DPO 训练正在进行，但是是一个占位符，因此跳过...")
    #  可以使用 trl 库来简化 DPO 的实现
    #  具体实现可以参考： https://huggingface.co/docs/trl/main/en/dpo_trainer#dpotrainer
    #  确保你已经安装了trl库： pip install trl

    # 在这里添加保存模型的代码
    torch.save(model.state_dict(), f'{output_dir}/dpo_model.pth')
    print(f"DPO 微调完成, 模型已保存至 {output_dir} 目录")


if __name__ == "__main__":

    # 1. 加载基础模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 你需要下载这个模型到本地
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

    # 2. 准备一些模拟的训练数据（你可以替换为你自己的数据）
    questions = [
        "请问 1+1 等于几？",
        "请问 2 的平方是多少？",
        "如果一个苹果的价格是 2 元，三个苹果的价格是多少？",
        "请你写一段python代码来实现冒泡排序",
        "如果地球是平的，会发生什么？",
        "什么是量子纠缠？",
        "猜一个谜语：什么东西早上四条腿，中午两条腿，晚上三条腿？",
        "编写一个函数判断一个字符串是否是回文串"
    ]
    ground_truth_answers = [
        "2",
        "4",
        "6 元",
        """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
        "如果地球是平的，那么就不会有重力，并且所有东西都会飘在空中。",
        "量子纠缠是一种量子力学现象，指多个粒子之间存在着一种特殊的关联，即便它们相隔很远。",
        "谜底是：人",
        """
def is_palindrome(s):
    s = ''.join(filter(str.isalnum, s)).lower()
    return s == s[::-1]
"""
    ]

    # 3. 生成蒸馏数据
    print("正在生成蒸馏数据...")
    distilled_data = generate_distilled_data(base_model, tokenizer, questions, num_rollouts=5)

    # 4. 格式化蒸馏数据，用于模型训练
    formatted_data = format_data_for_training(distilled_data)

    # 5. 创建 PyTorch Dataset 对象
    dataset = create_dataset(formatted_data, tokenizer)

    # 6. 使用 SFT 微调模型
    print("使用 SFT 微调模型...")
    train_model(base_model, tokenizer, dataset)

    # 7. 加载 SFT 后的模型
    sft_model_path = 'sft_model'
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # 8. 进行探索
    print("进行探索，并使用正确答案来训练...")
    refined_data = []
    for i, question in enumerate(questions):
        ground_truth_answer = ground_truth_answers[i]
        result = rollout_and_check(sft_model, tokenizer, question, ground_truth_answer)
        if result:
            print(f"问题: {question}, 找到正确答案。")
            refined_data.append({
                "question": question,
                "thought": result["thought"],
                "solution": result["solution"],
            })
        else:
            print(f"问题: {question}, 未找到正确答案。")

    # 9. 将找到的正确轨迹进行 SFT 再训练
    if refined_data:
        print("使用探索到的正确轨迹进行再训练（SFT）...")
        formatted_refined_data = format_data_for_training(refined_data)
        refined_dataset = create_dataset(formatted_refined_data, tokenizer)
        train_model(sft_model, tokenizer, refined_dataset, output_dir="sft_refined_model")

    # 10. 使用 DPO 进行训练 (这里需要你自己实现，这里仅为占位符)
    print("进行 DPO 训练（这里只是占位符）...")
    if refined_data:
        dpo_training(sft_model, tokenizer, refined_dataset)
    print("程序运行结束")