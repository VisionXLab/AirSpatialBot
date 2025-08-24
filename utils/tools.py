import re
import json
import ast

def custom_format(template: str, values: dict) -> str:
    """
    对 {{}} 包裹的内容进行格式化，对单 {} 保持不变。
    :param template: 输入的字符串模板
    :param values: 用于格式化的键值对
    :return: 格式化后的字符串
    """
    
    def replace_match(match):
        key = match.group(1)
        return values.get(key, f"{{{{{key}}}}}")

    result = re.sub(r"{{(.*?)}}", replace_match, template)
    return result

def _try_find_json(last_stream_response):
    try:
        last_stream_response = last_stream_response.strip().replace("\\_", '_')# .replace('\\"', '"').replace("\n", "")
        action_json = json.loads(last_stream_response)
        return action_json
    except json.JSONDecodeError as e:
        return None

def parse_action_from_text(last_stream_response):
    actions = []
    one_maybe_json = _try_find_json(last_stream_response)
    if one_maybe_json is not None:
        actions.extend(one_maybe_json)
    else:
        # search json block
        pattern = r'```json.*?```'
        res = re.findall(pattern, last_stream_response, re.S)
        if len(res) > 0:
            for one_maybe_json in res:
                one_maybe_json = one_maybe_json.replace("```json", "").replace("```", "").strip()
                maybe_action = _try_find_json(one_maybe_json)
                if maybe_action is not None:
                    actions.extend(maybe_action)
                else:
                    # 有可能是好几行的json，需要按行分开
                    for one_line in one_maybe_json.split("\n"):
                        maybe_action = _try_find_json(one_line)
                        if maybe_action is not None:
                            actions.extend(maybe_action)

    actions = [action for action in actions if action is not None]
    return actions

def process_bbox(bbox_string: str):
    match = re.search(r"<box>(.*?)</box>", bbox_string)
    if match:
        bbox_content = match.group(1)
        try:
            bbox_list = ast.literal_eval(bbox_content)
            return bbox_list
        except (ValueError, SyntaxError):
            raise ValueError("box 内容格式不正确，无法转换为列表。")
    else:
        raise ValueError("未找到 <box> 标签或内容。")
    
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
    
def process_3db(bbox_string: str):
    match = re.search(r"<3db>(.*?)</3db>", bbox_string)
    if match:
        bbox_content = match.group(1)
        try:
            bbox_list = ast.literal_eval(bbox_content)
            return bbox_list
        except (ValueError, SyntaxError):
            raise ValueError("3db 内容格式不正确，无法转换为列表。")
    else:
        raise ValueError("未找到 <3db> 标签或内容。")
    

def process_obb(bbox_string: str):
    match = re.search(r"<obb>(.*?)</obb>", bbox_string)
    if match:
        bbox_content = match.group(1)
        try:
            bbox_list = ast.literal_eval(bbox_content)
            return bbox_list
        except (ValueError, SyntaxError):
            raise ValueError("obb 内容格式不正确，无法转换为列表。")
    else:
        raise ValueError("未找到 <obb> 标签或内容。")