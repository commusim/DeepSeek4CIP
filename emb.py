import os
import json
import torch
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key='sk-c2262ab303d14ff78f97d9cfa1cdb4ab',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
#tensor(0.5705)

#tensor(0.6203)
#tensor(0.6562)

completion = client.embeddings.create(
    model="text-embedding-v3",
    input='Elegant',
    dimensions=1024,
    encoding_format="float"
)
data1 = torch.tensor(completion.data[0].embedding)
completion = client.embeddings.create(
    model="text-embedding-v3",
    input="""Minimalist
""",
    dimensions=1024,
    encoding_format="float"
)
data2 = torch.tensor(completion.data[0].embedding)
# data = json.loads(completion.model_dump_json())
print(data2@data1.T)