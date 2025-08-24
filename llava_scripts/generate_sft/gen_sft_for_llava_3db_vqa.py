import json
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from skimage.measure import block_reduce


COLOR_TEMPLATE = [
    "<image>\n[vqa] What color is this car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Output the color of this vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] What is the color of the car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Please tell me the color of the vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] Can you provide the color of this car<bbox>[{prompt}]</bbox> is?"
]

TYPE_TEMPLATE = [
    "<image>\n[vqa] What type is this car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Output the type of this vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] What is the type of the car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Please tell me the type of the vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] Can you provide the type of this car<bbox>[{prompt}]</bbox> is?"
]

COLOR_TYPE_TEMPLATE = [
    "<image>\n[vqa] What color and type is this car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Output the color and type of this vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] What is the color and type of the car<bbox>[{prompt}]</bbox>?",
    "<image>\n[vqa] Please tell me the color and type of the vehicle<bbox>[{prompt}]</bbox>",
    "<image>\n[vqa] Can you provide the color and type of this car<bbox>[{prompt}]</bbox> is?"
]

DEPTH_TEMPLATE = [
    "<image>\n[vqa] What is the depth value of the car<bbox>[{prompt}]</bbox>? (Unit: meter)",
    "<image>\n[vqa] Please tell me the depth of the car<bbox>[{prompt}]</bbox> in the image. (Unit: meter)",
    "<image>\n[vqa] Can you calculate the depth information of the vehicle<bbox>[{prompt}]</bbox> in the image? (Unit: meter)",
    "<image>\n[vqa] How deep is the vehicle<bbox>[{prompt}]</bbox> in the image? (Unit: meter)",
    "<image>\n[vqa] Can you provide the depth of the car<bbox>[{prompt}]</bbox> in the image? (Unit: meter)",
]

DISTANCE_TEMPLATE = [
    "<image>\n[vqa] How far is the camera from the car<bbox>[{prompt}]</bbox>? (Unit: meter)",
    "<image>\n[vqa] How far is vehicle<bbox>[{prompt}]</bbox> from the camera? (Unit: meter)",
    "<image>\n[vqa] What is the distance between the vehicle<bbox>[{prompt}]</bbox> and the camera? (Unit: meter)",
    "<image>\n[vqa] Can you measure the distance between the camera and the vehicle<bbox>[{prompt}]</bbox>? (Unit: meter)",
    "<image>\n[vqa] Please tell me how far away is the vehicle<bbox>[{prompt}]</bbox> and the camera. (Unit: meter)"
]

WIDTH_TEMPLATE = [
    "<image>\n[vqa] What is the width of the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] How wide is the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] Please tell me the width of the vehicle<bbox>[{prompt}]</bbox>. (Unit: millimeter)",
    "<image>\n[vqa] Do you know how wide this vehicle<bbox>[{prompt}]</bbox> is? (Unit: millimeter)",
    "<image>\n[vqa] Can you provide the width of this car<bbox>[{prompt}]</bbox> is? (Unit: millimeter)"
]

LENGTH_TEMPLATE = [
    "<image>\n[vqa] What is the length of the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] How long is the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] Please tell me the length of the vehicle<bbox>[{prompt}]</bbox>. (Unit: millimeter)",
    "<image>\n[vqa] Do you know how long this vehicle<bbox>[{prompt}]</bbox> is? (Unit: millimeter)",
    "<image>\n[vqa] Can you provide the length of this car<bbox>[{prompt}]</bbox> is? (Unit: millimeter)"
]

HEIGHT_TEMPLATE = [
    "<image>\n[vqa] What is the height of the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] How high is the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] Please tell me the height of the vehicle<bbox>[{prompt}]</bbox>. (Unit: millimeter)",
    "<image>\n[vqa] Do you know how high this vehicle<bbox>[{prompt}]</bbox> is? (Unit: millimeter)",
    "<image>\n[vqa] Can you provide the height of this car<bbox>[{prompt}]</bbox> is? (Unit: millimeter)"
]

WIDTH_LENGTH_HEIGHT_TEMPLATE = [
    "<image>\n[vqa] What is the size of the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] How long, wide, and high is the car<bbox>[{prompt}]</bbox>? (Unit: millimeter)",
    "<image>\n[vqa] Please tell me the size of the vehicle<bbox>[{prompt}]</bbox>. (Unit: millimeter)",
    "<image>\n[vqa] Do you know how long, wide, and high this vehicle<bbox>[{prompt}]</bbox> is? (Unit: millimeter)",
    "<image>\n[vqa] Can you provide the size of this car<bbox>[{prompt}]</bbox> is? (Unit: millimeter)"
]

TYPE_WIDTH_LENGTH_HEIGHT_TEMPLATE = [
    "<image>\n[vqa] What is the type, length, width, and height of the car<bbox>[{prompt}]</bbox>? (Unit: meter)",
    "<image>\n[vqa] What is the type of car<bbox>[{prompt}]</bbox>? And how long, wide, and high is it? (Unit: meter)",
    "<image>\n[vqa] Please tell me the type, length, width, and height of the vehicle<bbox>[{prompt}]</bbox>. (Unit: meter)",
    "<image>\n[vqa] Do you know the type, length, width, and height of this vehicle<bbox>[{prompt}]</bbox> is? (Unit: meter)",
    "<image>\n[vqa] Can you provide the type, length, width, and height of this car<bbox>[{prompt}]</bbox> is? (Unit: meter)"
]


def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data

metainfo_list = [
    # "rsvg_train.jsonl",
    # "dior_rsvg_train.jsonl",
    # "geochat_train.jsonl",
    # "vrsbench_train.jsonl",
    # "geoground_train.jsonl",
    "airspatial_qa_train.jsonl",
]
metainfo_dir = "/path/to/data/metadata"
image_dir = "/path/to/data/images"
save_file = "/path/to/data/sft_llava/llava_airspatial_vqa.json"
mask_dir = "/path/to/data/masks"
# PATCH_SIZE = 16

dict_list = []
ii = 0
for j in range(5):
    for metainfo in metainfo_list:
        data = load_jsonl(os.path.join(metainfo_dir, metainfo))
        for _, meta in tqdm(enumerate(data), total=len(data)):
            image_id = meta['image_id']

            # HBB
            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"
            with Image.open(image_file_path) as img:
                width, height = img.size  # 获取图片的宽度和高度

            bbox = [int(x * 1000 / width) if i % 2 == 0 else int(x * 1000 / height) for i, x in enumerate(meta['bbox'])]
            # xmin 和 ymin 不能小于 0
            bbox[0] = max(0, bbox[0])   # xmin
            bbox[1] = max(0, bbox[1])  # ymin
            # xmax 和 ymax 不能大于 100
            bbox[2] = min(1000, bbox[2])   # xmax
            bbox[3] = min(1000, bbox[3])  # ymax

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": DEPTH_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['depth']
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": DISTANCE_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['distance']
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": COLOR_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['color']
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": TYPE_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['type']
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": COLOR_TYPE_TEMPLATE[j].format(prompt=meta['bbox'])
                },
                {
                    "from": "gpt",
                    "value": meta['color'] + " " + meta['type']
                }, 
            ]
            dict_list.append(dict)
            ii += 1



            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": WIDTH_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['width']
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": LENGTH_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['length']
                }, 
            ]
            dict_list.append(dict)
            ii += 1

            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"
                
            dict['conversations'] = [
                {
                    "from": "human",
                    "value": HEIGHT_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": meta['height']
                }, 
            ]
            dict_list.append(dict)
            ii += 1

            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"
                
            dict['conversations'] = [
                {
                    "from": "human",
                    "value": WIDTH_LENGTH_HEIGHT_TEMPLATE[j].format(prompt=bbox)
                },
                {
                    "from": "gpt",
                    "value": f"<size>{int(meta['length'])},{int(meta['width'])},{int(meta['height'])}</size>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            dict = {}
            dict['id'] = str(ii)
            if "geoground3d" in metainfo:
                image_file_path = f"{image_dir}/{meta['image_id']}"
                dict['image'] = f"{meta['image_id']}"
            else:
                image_file_path = f"{image_dir}/{meta['dataset']}/{meta['image_id']}"
                dict['image'] = f"{meta['dataset']}/{meta['image_id']}"
                
            dict['conversations'] = [
                {
                    "from": "human",
                    "value": TYPE_WIDTH_LENGTH_HEIGHT_TEMPLATE[j].format(prompt=meta['bbox'])
                },
                {
                    "from": "gpt",
                    "value": f"{meta['type']},<size>{int(meta['length'])},{int(meta['width'])},{int(meta['height'])}</size>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1


with open(save_file, 'w') as json_file:    
    json.dump(dict_list, json_file, indent=4)  
