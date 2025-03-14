from data import FashionIQDataset
from LLM_response import LLM_init,message_unit,process_input,inference,process_output
import argparse

def main(args):
    processer,model = LLM_init(model_path = args.model_path)
    source = []
    target = []

    data = FashionIQDataset(root_path=args.data_path,split= args.spilt, type=None)
    for i in range(100):
        source_img_path = data[i][0]
        target_img_path = data[i][1]
        modifier = data[i][2]
        source_prompt = f"""change the style of this shirt/dress/toptee to {modifier} Desribe this modified shirt/dress/toptee in one word based on its style:"""
        target_prompt = """Describe this shirt/dress/toptee in one word based on its style:"""
        message_unit = message_unit(source_img_path,source_prompt)
        inputs = process_input(message_unit)
        generated_ids_trimmed = inference(model,inputs)
        text = process_output(processer,generated_ids_trimmed)
        print("".format(text))
        source.append(text)

        message_unit = message_unit(target_img_path,target_prompt)
        inputs = process_input(message_unit)
        generated_ids_trimmed = inference(model,inputs)
        text = process_output(processer,generated_ids_trimmed)
        print("".format(text))
        source.append(text)
        
    return None

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="/root/commusim/model-zoo/Qwen2-VL-7B-Instruct")
    args.add_argument("--spilt", type=str, default="train")
    args.add_argument("--data_path", type=str, default="/root/commusim/data/fashionIQ")
    # args.add_argument("--batch_size", type=int, default=1)
    args = args.parse_args()
    main(args)



