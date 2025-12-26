import os
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from swift.llm import (
get_model_tokenizer, get_template, safe_snapshot_download, BaseArguments, PtEngine, RequestConfig, InferRequest, load_image
)
from swift.utils import seed_everything
from swift.tuners import Swift



def eval_model(args):
    seed_everything(42)

    lora_checkpoint = safe_snapshot_download(args.model_path)  # 修改成checkpoint_dir
    new_args = BaseArguments.from_pretrained(lora_checkpoint)
    model = new_args.model
    template_type = new_args.template  # None: 使用对应模型默认的template_type
    default_system = None  # None: 使用对应模型默认的default_system

    # 加载模型和对话模板
    model, tokenizer = get_model_tokenizer(model, model_type=new_args.model_type)
    model = Swift.from_pretrained(model, lora_checkpoint)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    for i in tqdm(range(len(questions))):
        image_file=questions[i]['image_id']
        question_id=questions[i]['question_id']
        bbox = questions[i]['bbox']
        name, ext = os.path.splitext(image_file)
        new_image_name = f"{name}{ext}"
        image = load_image(os.path.join(args.image_folder, new_image_name))

        # Distance
        infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>\nHow far is the camera from the car<bbox>[{bbox}]</bbox>? (Unit: meter)"""}], images=[image])  
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        ans_file.write(json.dumps({
                                "question_id": questions[i]["question_id"],
                                "qtype": "distance",
                                "image_id": questions[i]["image_id"],
                                "answer": response,
                                "ground_truth": questions[i]['distance'],
                                # "question":questions[i]['question'],                      
                                }) + "\n")
        ans_file.flush()

        # Width
        infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>\nWhat is the width of the car<bbox>[{bbox}]</bbox>? (Unit: millimeter)"""}], images=[image])  
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        ans_file.write(json.dumps({
                                "question_id": questions[i]["question_id"],
                                "qtype": "width",
                                "image_id": questions[i]["image_id"],
                                "answer": response,
                                "ground_truth": questions[i]['width'],
                                # "question":questions[i]['question'],                      
                                }) + "\n")
        ans_file.flush()

        
        # length
        infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>\nWhat is the length of the car<bbox>[{bbox}]</bbox>? (Unit: millimeter)"""}], images=[image])  
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        ans_file.write(json.dumps({
                                "question_id": questions[i]["question_id"],
                                "qtype": "length",
                                "image_id": questions[i]["image_id"],
                                "answer": response,
                                "ground_truth": questions[i]['length'],
                                # "question":questions[i]['question'],                      
                                }) + "\n")
        ans_file.flush()


        # Height
        infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>\nWhat is the height of the car<bbox>[{bbox}]</bbox>? (Unit: millimeter)"""}], images=[image])  
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        ans_file.write(json.dumps({
                                "question_id": questions[i]["question_id"],
                                "qtype": "height",
                                "image_id": questions[i]["image_id"],
                                "answer": response,
                                "ground_truth": questions[i]['height'],
                                # "question":questions[i]['question'],                      
                                }) + "\n")
        ans_file.flush()
        
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    # parser.add_argument("--prompt", type=str, default="bbox")
    args = parser.parse_args()

    eval_model(args)