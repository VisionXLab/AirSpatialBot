import json
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from skimage.measure import block_reduce


HBB_TEMPLATE = [
    "<image>\n[refer] give me the bounding box of <ref>{prompt}</ref>",
    "<image>\n[refer] output the bounding box of the <ref>{prompt}</ref> in the image.",
    "<image>\n[refer] from this image, provide the bounding box for <ref>{prompt}</ref>",
    "<image>\n[refer] please provide the bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>",
    "<image>\n[refer] can you locate and provide the bounding box for <ref>{prompt}</ref> in the given image?"
]

B3D_TEMPLATE = [
    "<image>\n[refer] give me the 3D bounding box of <ref>{prompt}</ref>",
    "<image>\n[refer] output the 3D bounding box of the <ref>{prompt}</ref> in the image.",
    "<image>\n[refer] from this image, provide the 3D bounding box for <ref>{prompt}</ref>.",
    "<image>\n[refer] please provide the 3D bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>",
    "<image>\n[refer] can you locate and provide the 3D bounding box for <ref>{prompt}</ref> in the given image?"
]

OBB_TEMPLATE = [
    "<image>\n[refer] give me the oriented bounding box of <ref>{prompt}</ref>",
    "<image>\n[refer] output the oriented bounding box of the <ref>{prompt}</ref> in the image.",
    "<image>\n[refer] from this image, provide the oriented bounding box for <ref>{prompt}</ref>.",
    "<image>\n[refer] please provide the oriented bounding box coordinate of the region this sentence describes: <ref>{prompt}</ref>.",
    "<image>\n[refer] can you locate and provide the oriented bounding box for <ref>{prompt}</ref> in the given image?"
]


OBB2B3D_TEMPLATE = [
    "<image>\n[refer] give me the 3D bounding box of <ref>{prompt}</ref><obb>[{obb}]</obb>",
    "<image>\n[refer] output the 3D bounding box of the <ref>{prompt}</ref><obb>[{obb}]</obb> in the image.",
    "<image>\n[refer] from this image, provide the 3D bounding box for <ref>{prompt}</ref><obb>[{obb}]</obb>",
    "<image>\n[refer] please provide the 3D bounding box of the region this sentence describes: <ref>{prompt}</ref><obb>[{obb}]</obb>",
    "<image>\n[refer] can you locate and provide the 3D bounding box for <ref>{prompt}</ref><obb>[{obb}]</obb> in the given image?"
]

HBB2B3D_TEMPLATE = [
    "<image>\n[refer] give me the 3D bounding box of <ref>{prompt}</ref><box>[{bbox}]</box>",
    "<image>\n[refer] output the 3D bounding box of the <ref>{prompt}</ref><box>[{bbox}]</box> in the image.",
    "<image>\n[refer] from this image, provide the 3D bounding box for <ref>{prompt}</ref><box>[{bbox}]</box>",
    "<image>\n[refer] please provide the 3D bounding box of the region this sentence describes: <ref>{prompt}</ref><box>[{bbox}]</box>",
    "<image>\n[refer] can you locate and provide the 3D bounding box for <ref>{prompt}</ref><box>[{bbox}]</box> in the given image?"
]




def encode_mask(mask_list):
    rows = []
    for row in mask_list:
        encoded_row = []
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                encoded_row.append(f"{row[j-1]} *{count}")
                count = 1
        encoded_row.append(f"{row[-1]} *{count}")
        rows.append(", ".join(encoded_row))
    return "; ".join(rows) + ";"


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
    "airspatial_rec_train.jsonl",
]
metainfo_dir = "/path/to/data/metadata"
image_dir = "/path/to/data/images"
save_file = "/path/to/data/sft_llava/llava_airspatial_rec.json"
mask_dir = "/path/to/data/masks"
# PATCH_SIZE = 16

dict_list = []
ii = 0
for j in range(3):
    for metainfo in metainfo_list:
        data = load_jsonl(os.path.join(metainfo_dir, metainfo))
        for _, meta in tqdm(enumerate(data), total=len(data)):
            image_id = meta['image_id']


            # HBB
            dict = {}
            dict['id'] = str(ii)
            if "geoground" in metainfo:
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
                    "value": HBB_TEMPLATE[j].format(prompt=meta['question'])
                },
                {
                    "from": "gpt",
                    "value": f"<box>[{bbox}]</box>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1

            # 3DBB
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

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": B3D_TEMPLATE[j].format(prompt=meta['question'])
                },
                {
                    "from": "gpt",
                    "value": f"<3db>[{meta['bbox_3d']}]</3db>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1

            # OBB
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

            bboxps = np.array(meta['poly']).astype(int)
            rbbox = cv2.minAreaRect(bboxps)
            x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
            if w < h:
                w, h = h, w
                a += 90
            while not 90 > a >= -90:
                if a >= 90:
                    a -= 180
                else:
                    a += 180
            assert 90 > a >= -90

            obb = [int(x * 100 / width), int(y * 100 / height), int(w  * 100 / width), int(h * 100 / width), int(a)]

            # xmin 和 ymin 不能小于 0
            obb[0] = max(0, obb[0])   # xmin
            obb[1] = max(0, obb[1])  # ymin
            # xmax 和 ymax 不能大于 100
            obb[2] = min(100, obb[2])   # xmax
            obb[3] = min(100, obb[3])  # ymax

            dict['conversations'] = [
                {
                    "from": "human",
                    "value": OBB_TEMPLATE[j].format(prompt=meta['question'])
                },
                {
                    "from": "gpt",
                    "value": f"<obb>[{obb}]</obb>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            # OBB23DB
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
                    "value": OBB2B3D_TEMPLATE[j].format(prompt=meta['question'], obb=obb)
                },
                {
                    "from": "gpt",
                    "value": f"<3db>[{meta['bbox_3d']}]</3db>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            # HBB23DB
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
                    "value": HBB2B3D_TEMPLATE[j].format(prompt=meta['question'], bbox=bbox)
                },
                {
                    "from": "gpt",
                    "value": f"<3db>[{meta['bbox_3d']}]</3db>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1

            # 3DB2OBB
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
                    "value": B3D_TEMPLATE[j].format(prompt=meta['question'])
                },
                {
                    "from": "gpt",
                    "value": f"<3db>[{meta['bbox_3d']}]</3db>"
                }, 
                {
                    "from": "human",
                    "value": "The oriented bounding box corresponding to the 3D bounding box is"
                },
                {
                    "from": "gpt",
                    "value": f"<obb>[{obb}]</obb>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1


            # 3DB2HBB
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
                    "value": B3D_TEMPLATE[j].format(prompt=meta['question'])
                },
                {
                    "from": "gpt",
                    "value": f"<3db>[{meta['bbox_3d']}]</3db>"
                }, 
                {
                    "from": "human",
                    "value": "The bounding box corresponding to the 3D bounding box is"
                },
                {
                    "from": "gpt",
                    "value": f"<box>[{bbox}]</box>"
                }, 
            ]
            dict_list.append(dict)
            ii += 1

 

with open(save_file, 'w') as json_file:    
    json.dump(dict_list, json_file, indent=4)  
