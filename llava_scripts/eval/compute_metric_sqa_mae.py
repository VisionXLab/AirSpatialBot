import re
import math
from tqdm import tqdm
import os
from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import inflect
import ast
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square


def extract_number(text):
    numbers = re.findall(r"-?\d*\.\d+|-?\d+", text)  # 提取整数和小数
    if not numbers:
        return False  # 没有数字返回 0
    return numbers[-1]

def process_size(size_string: str):
    match = re.search(r"<size>(.*?)</size>", size_string)
    if match:
        size_content = match.group(1)
        try:
            size_list = ast.literal_eval(size_content)
            return size_list
        except (ValueError, SyntaxError):
            raise ValueError("size 内容格式不正确，无法转换为列表。")
    else:
        raise ValueError("未找到 <size> 标签或内容。")
    
# 读取JSONL文件并将每行解析为Python字典，存入列表
def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data

def folder_creat_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--answers-file', required=True, help='Target jsonl directory')

    args = parser.parse_args()

    # Create an engine instance
    convert = inflect.engine()

    # 从 jsonl 文件中加载数据
    predict = load_jsonl(args.answers_file)
    total_cnt = len(predict)

    # tolerance = 0.01

    # correct = 0
    # total = 0
    all_types = ['depth', 'distance', 'length', 'width', 'height', 'size']
    print('number of question types:', len(all_types) - 1)

  


    for cur_type in all_types:
        invalid_format = 0
        gt_list = []
        pred_list = []
        predict = load_jsonl(args.answers_file)
        for i, predict in tqdm(enumerate(predict), total=total_cnt):
            pred_ans = predict['answer']
            gt_ans = predict['ground_truth']
            q_type = predict['qtype']
            img_id = predict['image_id']
            if q_type == cur_type:
                if q_type == "size":
                    try:
                        pred_ans = process_size(pred_ans)
                        gt_ans = process_size(gt_ans)
                    except:
                        invalid_format += 1
                        continue
                    # gt_list.append(gt_ans)
                    # pred_list.append(pred_ans)
                    # if (abs(gt_ans[0] - pred_ans[0]) / gt_ans[0] < tolerance) and (abs(gt_ans[1] - pred_ans[1]) / gt_ans[1] < tolerance) and (abs(gt_ans[2] - pred_ans[2]) / gt_ans[2] < tolerance):
                    #     correct += 1
                    #     if q_type in all_types:
                    #         correct_per_type[q_type] += 1
                    # print(pred_ans, gt_ans)
                    gt_list.append(sum(gt_ans))
                    pred_list.append(sum(pred_ans))

                else:
                    
                    gt_ans = float(gt_ans)
                    pred_ans = extract_number(pred_ans)
                    if pred_ans == False:
                        invalid_format += 1
                        continue   
                    pred_ans = float(pred_ans)
                    # print(pred_ans)
                    if pred_ans > 10000:
                        invalid_format += 1
                        continue   
                    gt_list.append(gt_ans)
                    pred_list.append(pred_ans)
        try:
            mse = mean_squared_error(gt_list, pred_list)
            mae = mean_absolute_error(gt_list, pred_list)
            r2 = r2_score(gt_list, pred_list)
            print('Type:', cur_type, 'invalid_ratio:', invalid_format / len(gt_list), 'MAE:', mae, 'RMSE:', mse ** 0.5, 'R2:', r2)
        except:
            print('Type:', cur_type, 'invalid_number:', invalid_format)



    invalid_format = 0
    gt_list = []
    pred_list = []
    predict = load_jsonl(args.answers_file)
    for i, predict in tqdm(enumerate(predict), total=total_cnt):
        pred_ans = predict['answer']
        gt_ans = predict['ground_truth']
        q_type = predict['qtype']
        img_id = predict['image_id']
        if q_type == "size":
            try:
                pred_ans = process_size(pred_ans)
                gt_ans = process_size(gt_ans)
            except:
                invalid_format += 1
                continue
            gt_list.append(sum(gt_ans))
            pred_list.append(sum(pred_ans))

        else:
            
            gt_ans = float(gt_ans)
            pred_ans = extract_number(pred_ans)
            if pred_ans == False:
                invalid_format += 1
                continue   
            pred_ans = float(pred_ans)
            if pred_ans > 10000:
                invalid_format += 1
                continue   
            gt_list.append(gt_ans)
            pred_list.append(pred_ans)
    mse = mean_squared_error(gt_list, pred_list)
    mae = mean_absolute_error(gt_list, pred_list)
    r2 = r2_score(gt_list, pred_list)
    print('invalid_ratio:', invalid_format / len(gt_list), 'MAE:', mae, 'RMSE:', mse ** 0.5, 'R2:', r2)
