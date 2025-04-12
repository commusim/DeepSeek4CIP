from openai import OpenAI
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt
import csv
from data import FashionIQDataset
import re
import pandas as pd
from tqdm import tqdm


def read_csv_to_list(file_path):
    """
    读取CSV文件并将其内容保存为一个列表。
    
    :param file_path: CSV文件的路径
    :return: 包含CSV文件内容的列表
    """
    data_list = []  # 用于存储CSV文件的内容
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            # 创建CSV读取器
            csv_reader = csv.reader(file)
            
            # 遍历每一行并将内容添加到列表中
            for row in csv_reader:
                data_list.append(row)  # 将每一行作为子列表添加到主列表中
            
            return data_list  # 返回包含所有内容的列表
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")
    
    return data_list  # 如果发生异常，返回当前已读取的数据


def read_txt_to_list(file_path):
    """
    读取TXT文件并将其内容保存为一个列表。
    
    :param file_path: TXT文件的路径
    :return: 包含TXT文件内容的列表
    """
    data_list = []  # 用于存储TXT文件的内容
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            # 逐行读取文件内容
            for line in file:
                # 去除行末的换行符，并将每一行添加到列表中
                data_list.append(line.strip())
            
            return data_list  # 返回包含所有内容的列表
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")
    
    return data_list  # 如果发生异常，返回当前已读取的数据


#########################
#      normal formation
#########################


client = OpenAI(
    api_key="sk-27c60d9760fe49089e2848cfb61f9415",
    base_url=f"https://api.deepseek.com",
    # stream=False
)



# Function to extract content from the provided text
def extract_content(text):
    """
    Extracts <think1>, <query>, <think2>, and <summary> sections from the input text.
    :param text: Input text containing <think1>, <query>, <think2>, and <summary>.
    :return: A dictionary containing the extracted sections.
    """
    # Define regex patterns for each section
    think1_pattern = r"<think1>(.*?)</think1>"
    query_pattern = r"<query>(.*?)</query>"
    think2_pattern = r"<think2>(.*?)</think2>"
    summary_pattern = r"<summary>(.*?)</summary>"
    
    # Extract content using regex
    think1_match = re.search(think1_pattern, text, re.DOTALL)
    query_match = re.search(query_pattern, text, re.DOTALL)
    think2_match = re.search(think2_pattern, text, re.DOTALL)
    summary_match = re.search(summary_pattern, text, re.DOTALL)
    
    # Store extracted content in a dictionary
    extracted_data = {
        "think1": think1_match.group(1).strip() if think1_match else None,
        "query": query_match.group(1).strip() if query_match else None,
        "think2": think2_match.group(1).strip() if think2_match else None,
        "summary": summary_match.group(1).strip() if summary_match else None
    }
    
    return extracted_data

# Function to save or append data to an Excel file
def save_or_append_to_excel(file_path, new_data):
    """
    Save or append data to an Excel file.
    :param file_path: Path to the Excel file.
    :param new_data: List of dictionaries containing new data to be appended.
    """
    # Check if the file already exists
    if os.path.exists(file_path):
        # Read existing data from the file
        existing_df = pd.read_excel(file_path)
        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        # Append new data to existing data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        combined_df = pd.DataFrame(new_data)
    
    # Save the combined DataFrame to the Excel file
    combined_df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"Data saved/updated in {file_path}")


def response_llm(client, prompt, messages, reason_message, output_path, i, ask=False,):
    # i = len(messages) // 2 + 1
    # pdb.set_trace()
    print("\n" + "*" * 20 + f"第{i}轮对话" + "*" * 20)

    if ask:
        print("=" * 20 + "提问" + "=" * 20)
        print(prompt)

    messages.append({'role': 'user', 'content': prompt})
    completion = client.chat.completions.create(
        model="deepseek-reasoner",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
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
                # print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering == False:
                    # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                # print(delta.content, end='', flush=True)
                answer_content += delta.content
    with open(os.path.join(output_path, 'LLM_onsestep.txt'), 'a', encoding='utf-8') as file:
        file.write("\n\n" + "=" * 20 + f"第{i}轮对话" + "=" * 20)
        file.write("\n" + '-' * 10 + "提问" + '-' * 10 + "\n")
        file.write(prompt)
        file.write("\n" + '-' * 10 + "思考" + '-' * 10 + "\n")
        file.write(reasoning_content)
        file.write("\n" + '-' * 10 + "回答" + '-' * 10 + "\n")
        file.write(answer_content)
    # 将内容添加型上下文当中
    return answer_content



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
<summary>Summary: Summarize the transformation process from the source to the target image in fluent language, including the logic of key element changes and design choices.</summary>

Please generate the Chain of Thought strictly following the above output format.
"""
    return prompt

if __name__ == "__main__":
    # 示例调用
    # 替换为您的CSV文件的实际路径
    data = FashionIQDataset(root_path="/root/commusim/fashionIQ",split="train", type=None)
    source = "./output/source_sentence.txt"
    target = "./output/target_sentence.txt"

    # 调用函数读取CSV文件并保存为列表
    source_list = read_txt_to_list(source)
    target_list = read_txt_to_list(target)
    # cip_list = data[:][2]
    # import pdb; pdb.set_trace()
    # cip = read()

    for i in tqdm(range(len(data))):
        prompt = process_prompt(source_list[i], data[i][2], target_list[i])
        # print(prompt)
        input_text = response_llm(client, prompt, [], [], "output", i, ask=False)
        print(input_text)
        # File path for the Excel file
        excel_file_path = "extracted_data.xlsx"
        extracted_data = extract_content(input_text)
        # Append the extracted data to the Excel file
        save_or_append_to_excel(excel_file_path, [extracted_data])
    print(1)