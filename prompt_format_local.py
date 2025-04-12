from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import csv
import re
import pandas as pd
from tqdm import tqdm


def read_csv_to_list(file_path):
    """
    读取CSV文件并将其内容保存为一个列表。
    
    :param file_path: CSV文件的路径
    :return: 包含CSV文件内容的列表
    """
    data_list = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_list.append(row)
        return data_list
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")
    return data_list


def read_txt_to_list(file_path):
    """
    读取TXT文件并将其内容保存为一个列表。
    
    :param file_path: TXT文件的路径
    :return: 包含TXT文件内容的列表
    """
    data_list = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line in file:
                data_list.append(line.strip())
        return data_list
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")
    return data_list


def extract_content(text):
    """
    提取 <think1>, <query>, <think2>, 和 <summary> 部分的内容。
    
    :param text: 输入文本
    :return: 包含提取内容的字典
    """
    patterns = {
        "think1": r"<think1>(.*?)</think1>",
        "query": r"<query>(.*?)</query>",
        "think2": r"<think2>(.*?)</think2>",
        "summary": r"<summary>(.*?)</summary>"
    }
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        extracted_data[key] = match.group(1).strip() if match else None
    return extracted_data


def save_or_append_to_excel(file_path, new_data):
    """
    将数据保存或追加到Excel文件中。
    
    :param file_path: Excel文件路径
    :param new_data: 新数据（字典列表）
    """
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(new_data)
    
    combined_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"数据已保存/更新到 {file_path}")


def response_local_model(prompt, tokenizer, model, max_length=32768):
    """
    使用本地模型生成回复。
    
    :param prompt: 输入提示
    :param tokenizer: 分词器
    :param model: 模型
    :param max_length: 最大生成长度
    :return: 生成的文本
    """
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print(f"输入长度: {len(inputs['input_ids'][0])}")
    
    # test_input = "请解释一下什么是深度学习。"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 生成推理输出
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去除输入 prompt 的部分
    # if generated_text.startswith("<｜User｜>\n"+prompt):
    #     generated_text = generated_text[len(prompt):].strip()
    index = generated_text.find("<think>")
    generated_text = generated_text[index:]
    # print(generated_text)
    return generated_text


def process_prompt(source, modifier, target):
    """
    构造输入提示。
    
    :param source: 源图像描述
    :param modifier: 转换指令
    :param target: 目标图像描述
    :return: 格式化的提示文本
    """
    prompt = f"""
Input Information:
Source Image Description: {source}
Transformation Instruction: {modifier}
Target Image Description: {target}

Output Format (to be generated):
<summary>Summary: [Strictly follow this structure: Summarize the transformation process from the source to the target image in fluent language, including the logic of key element changes and design choices.]</summary>

Please generate the Chain of Thought strictly following the above output format.
"""
    return prompt


if __name__ == "__main__":
    # 加载本地模型和分词器
    model_name = "/root/commusim/model-zoo/DeepSeek"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda:1")
    model.eval()

    # 数据集路径
    root_path = "/root/commusim/fashionIQ"
    split = "train"
    type_ = None

    # 源和目标句子文件路径
    source_file = "./output/source_sentence.txt"
    target_file = "./output/target_sentence.txt"

    # 读取源和目标句子
    source_list = read_txt_to_list(source_file)
    target_list = read_txt_to_list(target_file)

    # 确保源和目标句子数量一致
    if len(source_list) != len(target_list):
        raise ValueError("源句子和目标句子的数量不一致，请检查输入文件。")

    # Excel 文件路径
    excel_file_path = "extracted_data1.xlsx"

    # 处理数据
    for i, (source, target) in enumerate(tqdm(zip(source_list, target_list), total=len(source_list))):
        if i < 10570:
            continue
        print(i)
        # 构造提示
        prompt = process_prompt(source, "Transformation Instruction", target)
        
        # 使用模型生成响应
        generated_text = response_local_model(prompt, tokenizer, model)
        # print(generated_text)
        # import pdb; pdb.set_trace()
        # 提取内容
        extracted_data = extract_content(generated_text)
        extracted_data["think1"] = source
        extracted_data["think2"] = target
        extracted_data["query"] = i
        with open("data1.txt", "a") as file:
            file.write(str(extracted_data))
        with open("restore1.txt", "w") as file:
            file.write(str(extracted_data))
        # 保存到 Excel
        # import pdb; pdb.set_trace()
        save_or_append_to_excel(excel_file_path, [extracted_data])
    
    print("处理完成！")