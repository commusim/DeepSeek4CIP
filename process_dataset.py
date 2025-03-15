from data import FashionIQDataset
from LLM_response import LLM_init,message_unit,process_input,inference,process_output
import argparse
import numpy as np
from tqdm import tqdm
def main(args):
    processer,model = LLM_init(model_path = args.model_path)
    source = []
    target = []
    cip = []
    data = FashionIQDataset(root_path=args.data_path,split= args.spilt, type=None)
    if args.save_long:
        save_long = "sentence"
    else:
        save_long = "word"
    # for i in tqdm(range(len(data))):
    for i in range(2):
        source_img_path = data[i][0]
        target_img_path = data[i][1]
        modifier = data[i][2]
        cip_prompt = f"""change the style of this shirt/dress/toptee to {modifier}. Desribe this modified shirt/dress/toptee in one {save_long} based on its style:"""
        img_prompt = f"""Describe this shirt/dress/toptee in one {save_long} based on its style:"""
        # print(img_prompt)
        if args.task in ["cip"]:
            message = message_unit(source_img_path,cip_prompt)
            inputs = process_input(message, processer)
            generated_ids_trimmed,output = inference(model,inputs)
            # import pdb; pdb.set_trace()
            text = process_output(processer,generated_ids_trimmed)
            # print("{}".format(text))
            cip.append(text)

        if args.task in ["target"]:
            message = message_unit(target_img_path,img_prompt)
            inputs = process_input(message,processer)
            generated_ids_trimmed,output = inference(model,inputs)
            text = process_output(processer,generated_ids_trimmed)
            # print("{}".format(text))
            target.append(text)
            
        if args.task in ["source"]:
            message = message_unit(source_img_path,img_prompt)
            inputs = process_input(message,processer)
            generated_ids_trimmed,output = inference(model,inputs)
            text = process_output(processer,generated_ids_trimmed)
            # print("{}".format(text))
            source.append(text)




    np.savetxt(f"{args.task}_{save_long}.csv",np.array(cip),fmt='%s', delimiter=',', encoding='utf-8')
    return None

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="/root/commusim/model-zoo/Qwen2-VL-7B-Instruct")
    args.add_argument("--spilt", type=str, default="train")
    args.add_argument("--data_path", type=str, default="/root/commusim/fashionIQ")
    args.add_argument("--task", type=str, default="cip")
    args.add_argument("--save_long", type=bool, default=True)
    args = args.parse_args()
    main(args)



