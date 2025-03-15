from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

def LLM_init(model_path):
    # model_path = "/root/commusim/model-zoo/Qwen2-VL-7B-Instruct"
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
    return processor, model

def message_unit(image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages

def process_input(messages, processor):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs


def inference(model, inputs):
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return generated_ids_trimmed, generated_ids


def process_output(processor, generated_ids_trimmed):

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def main():
    processer,model = LLM_init(model_path = "/root/commusim/model-zoo/Qwen2-VL-7B-Instruct")
    prompt = "describe this image"
    image_path =  "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    message = message_unit(image_path,prompt)
    inputs = process_input(message,processer)
    generated_ids_trimmed = inference(model,inputs)
    text = process_output(processer,generated_ids_trimmed)
    print("{}".format(text))
    return None


if __name__ == "__main__":
    main()