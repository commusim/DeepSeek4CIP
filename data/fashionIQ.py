from typing import Dict, List
from torch.utils.data import Dataset
import numpy as np
import torch
import os



class FashionIQDataset(Dataset):
    """
    Fashion200K dataset.
    Image pairs in {root_path}/image_pairs/{split}_pairs.pkl

    """

    def __init__(self,root_path, split, type):
        super().__init__()
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.split = split
        type = "-" + type if type is not None else ""
        self.source_img = []
        self.target_img = []
        self.modifier = []
        # load image pairs
        path = os.path.join(self.root_path, 'captions_pairs', f'fashion_iq-{split}-cap.txt')
        with open(path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                # 去除行末的换行符，并使用分号分割字符串
                items = line.strip().split(';')
                # 将分割后的数据添加到列表中
                self.source_img.append(items[0])
                self.target_img.append(items[1])
                self.modifier.append(items[2])
        print(f"Load {split} pairs from {path}")

    def __getitem__(self, idx):
        source_img_path = os.path.join(self.img_root_path, self.source_img[idx])
        target_img_path = os.path.join(self.img_root_path, self.target_img[idx])
        modifier = self.modifier[idx]
        return source_img_path, target_img_path, modifier

    def __len__(self):
        return len(self.source_img)

if __name__ == "__main__":
    # 定义用于存放解析后数据的列表
    data = FashionIQDataset("D:/dataset/fashionIQ/fashionIQ", 'train', 'dress')
    for i in data:
        print(i)