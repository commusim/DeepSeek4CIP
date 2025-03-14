from openai import OpenAI
import os
import base64
from PIL import Image
import matplotlib.pyplot as plt
#  base 64 编码格式


# message = []
# message.append()

#########################
#      normal formation
#########################
# 定义特殊 token
BEGIN_OF_THOUGHT = "<|begin_of_thought|>"
END_OF_THOUGHT = "<|end_of_thought|>"
BEGIN_OF_SOLUTION = "<|begin_of_solution|>"
END_OF_SOLUTION = "<|end_of_solution|>"

prompt_cn = \
f"""你的角色是一个助手，需要通过系统的长思考过程来解决问题，然后再给出最终的准确答案。

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
{'question'}
"""

prompt_en = f"""Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. 

Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: 

“‘
{BEGIN_OF_THOUGHT}
{{thought with steps separated with "\n\n"}} 
{BEGIN_OF_THOUGHT} 
”’
 
Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: 

“‘ 
{BEGIN_OF_THOUGHT} 
{{final formatted, precise, and clear solution}}
{END_OF_SOLUTION} 
”’ 

Now, try to solve the following question through the above guidelines"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key='sk-c2262ab303d14ff78f97d9cfa1cdb4ab',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def message_unit(image_path, prompt, slow_think_prompt=prompt_en):
    image = encode_image(image_path)
    ask_dict = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{slow_think_prompt}"},
            {
                "type": "image_url",
                # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                # PNG图像：  f"data:image/png;base64,{base64_image}"
                # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                # WEBP图像： f"data:image/webp;base64,{base64_image}"
                "image_url": {"url": f"data:image/png;base64,{image}"},
            },
            {"type": "text", "text": f"{prompt}"},
        ],
    }
    return ask_dict
init_dict = {
    	    "role": "system",
            "content": [{"type":"text","text": "You are a helpful assistant."}]}


def response_llm_api(message, client):
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=message,
    )

    content = completion.choices[-1].message.content

    return content

def extratc_key(start_token, end_token):
    return

if __name__ == "__main__":
    root_path = "D:/dataset/fashionIQ/fashionIQ/image_data"
    path1 =  "D:/dataset/fashionIQ/fashionIQ/captions_pairs/fashion_iq-train-cap.txt"
    # image1 = "dress/B003FGW7MK.jpg"
    # image2 = "dress/B008BHCT58.jpg"
    # text = "is solid black with no sleeves"

    image1 = "dress/B003FGW7MK.jpg"
    image2 = "dress/B008BHCT58.jpg"
    text = "is black with straps "
    I_s = os.path.join(root_path,image1)
    I_t = os.path.join(root_path,image2)
    plt.imshow(Image.open(I_s))
    plt.title(image1)
    plt.show()
    plt.imshow(Image.open(I_t))
    plt.title(image2)
    plt.show()
    source_prompt = f"""change the style of this shirt/dress/toptee to {text} Desribe this modified shirt/dress/toptee in one word based on its style:"""
    target_prompt = """Describe this shirt/dress/toptee in one word based on its style:"""
    message_s = message_unit(I_s,source_prompt)
    message_t = message_unit(I_t, target_prompt)
    x1 = response_llm_api([init_dict,message_s],client)
    x2 = response_llm_api([init_dict,message_t],client)

    print(1)