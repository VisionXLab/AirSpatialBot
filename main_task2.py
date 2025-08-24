from llm_client import LanguageModelClient
from utils.tools import custom_format, parse_action_from_text, process_bbox
import asyncio
import json
import pandas as pd
from tqdm.asyncio import tqdm
import os
from typing import List, Dict, Any, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

image_path  = "/path/to/images/geoground"
gt_json = "airspatial_agent_test_task2_v2.jsonl"
action_json = "action_json/output_task2_geoground+internvl_8b.json"
answer_json = "execute_json/answer_task2_geoground+internvl_8b.json"
eval_json = "evaluate_json/evaluate_task2_geoground+internvl_8b.json"

vlm_name = "geoground"
llm_name = "internvl2-8b"

semaphore = asyncio.Semaphore(1)

csv_files = [
    "csv_file/output-19-add.csv",
    "csv_file/output-20-add.csv",
    "csv_file/output-40-add.csv",
    "csv_file/output-42-add.csv",
    "csv_file/output-43-add.csv",
    "csv_file/output-44-add.csv",
    "csv_file/output-45-add.csv",
    "csv_file/output-46-add.csv",
    "csv_file/output-47-add.csv",
    "csv_file/output-48-add.csv",
    "csv_file/output-49-add.csv",
]
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)


class ActionExecutor:
    def __init__(
        self, action=None, img_path: str = "", query: str = "", index: int = None
    ):
        self.action: List[Dict[str, Union[str, Any]]] = action
        self.query: str = query
        self.img_path: str = img_path
        self.response: Dict = {}
        self.sliced_img_path: List[str] = []
        self.fuction_dict = {
            "img_slice": self.single_img_slice,
            "image_understanding": self.image_understanding,
            "spatial_understanding": self.spatial_understanding,
            "query_table": self.query_table,
            "web_search": self.web_search,
            "query_size": self.query_size,
            "disambiguation": self.disambiguation,
        }
        self.executor = ThreadPoolExecutor()
        self.index = index
        self.df = df
        self.size = {}
        self.bboxes = []
        self.gt_bbox = None
        self.understanding = []

    async def execute(self):
        try:
            for action in self.action:
                fuction_name = action["action_type"]
                execute_fuction = self.fuction_dict.get(fuction_name)
                if execute_fuction is None:
                    raise Exception(f"Function {fuction_name} not found.")
                args = action.get("params", {})
                self.response[fuction_name] = await execute_fuction(**args)
        except Exception as e:
            self.response = {"error": str(e)}

    async def single_img_slice(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()

        def crop_image():
            with Image.open(self.img_path) as img:
                min_x = kwargs['min_x']
                min_y = kwargs['min_y']
                max_x = kwargs['max_x']
                max_y = kwargs['max_y']
                try:
                    index = kwargs['index']
                except KeyError:
                    index = None

                cropped_img = img.crop((min_x, min_y, max_x, max_y))
                if index is not None:
                    out_path = self.img_path.replace(".JPG", f"_{index}_cropped.JPG")
                else:
                    out_path = self.img_path.replace(".JPG", "_cropped.JPG")
                cropped_img.save(out_path)
                self.sliced_img_path.append(out_path)
                return out_path

        result = await loop.run_in_executor(self.executor, crop_image)
        return result

    async def img_slice(self, bboxes: List) -> List[str]:
        for i, bbox in enumerate(bboxes):
            await self.single_img_slice(
                min_x=bbox[0], min_y=bbox[1], max_x=bbox[2], max_y=bbox[3], index=i
            )

    async def image_understanding(self, **kwargs) -> str:

        answers = []
        for index, img_path in enumerate(self.sliced_img_path):
            try:
                llm_client = LanguageModelClient(model_name=vlm_name)

                with open(
                    "prompts/image_understanding.txt", "r", encoding="utf-8"
                ) as f:
                    prompt = f.read()
                    prompt = custom_format(prompt, {"user_input": self.query})

                response = await llm_client.send_request_to_server(prompt=prompt, img_path=img_path)
                answers.append({str(index): response})
            except Exception as e:
                answers.append({"error": str(e)})

        return answers

    async def spatial_understanding(self, **kwargs):
        anno = kwargs["anno"]
        if anno == "get_size":
            min_x = kwargs['min_x']
            min_y = kwargs['min_y']
            max_x = kwargs['max_x']
            max_y = kwargs['max_y']

            bbox = [min_x, min_y, max_x, max_y]
            img_url = self.img_path.split("/")[-1]
            row = self.df[
                (self.df["Image URL"] == img_url)
                & (self.df["x_min"] == bbox[0])
                & (self.df["y_min"] == bbox[1])
                & (self.df["x_max"] == bbox[2])
                & (self.df["y_max"] == bbox[3])
            ]
            if row.empty:
                self.size = {"length": 0, "width": 0, "height": 0}
            else:
                length, width, height = (
                    row.iloc[0]["length"],
                    row.iloc[0]["width"],
                    row.iloc[0]["height"],
                )
                self.size = {"length": length, "width": width, "height": height}
            return self.size
        else:
            data = pd.read_json("airspatial_agent_test_task2.jsonl", lines=True)
            for _, row in data.iterrows():
                if row["question_id"] == self.index:
                    self.bboxes = process_bbox(row["answer"])
                    self.gt_bbox = row["bbox"]
            return self.bboxes

    async def query_table(self, **kwargs):
        ans = {}
        row = self.df[
            (self.df["length"] == self.size["length"])
            & (self.df["width"] == self.size["width"])
            & (self.df["height"] == self.size["height"])
        ]
        if row.empty:
            ans = {
                "error": "No matching row found in the DataFrame for the given size."
            }
        else:
            type = kwargs["type"]
            for sub in type:
                ans[sub] = str(row[sub].values[0])

        return ans

    async def web_search(self, **kwargs):
        return "calling_web_search"

    async def query_size(self, **kwargs):
        brand, model = kwargs["brand"], kwargs["model"]
        try:
            color = kwargs["color"]
        except KeyError:
            color = None
        finally:
            if color:
                row = self.df[
                    (self.df["brand"] == brand)
                    & (self.df["model"] == model)
                    & (self.df["color"] == color)
                ]
            else:
                row = self.df[(self.df["brand"] == brand) & (self.df["model"] == model)]
            if row.empty:
                return {"error": "No matching row found in the DataFrame."}
            else:
                length, width, height = (
                    row.iloc[0]["length"],
                    row.iloc[0]["width"],
                    row.iloc[0]["height"],
                )
                self.size = {"length": length, "width": width, "height": height}
                return self.size

    async def disambiguation(self, **kwargs):

        async def llm_disambiguation(result, gt_color, gt_type):
            with open("prompts/disambiguation.txt", "r", encoding="utf-8") as f:
                prompt = f.read()
                prompt = custom_format(
                    prompt,
                    {
                        "discription": str(result),
                        "gt_color": gt_color,
                        "gt_type": gt_type,
                    },
                )
            client = LanguageModelClient(model_name=llm_name)
            response = await client.send_request_to_server(prompt=prompt)
            return response

        try:
            color = kwargs["color"]
        except KeyError:
            color = None
        await self.img_slice(bboxes=self.bboxes)
        result = await self.image_understanding()
        gt_row = self.df[
            (self.df["Image URL"] == Path(self.img_path).name)
            & (self.df["x_min"] == self.gt_bbox[0])
            & (self.df["y_min"] == self.gt_bbox[1])
            & (self.df["x_max"] == self.gt_bbox[2])
            & (self.df["y_max"] == self.gt_bbox[3])
        ]
        gt_color, gt_type = gt_row["color"].values[0], gt_row["type"].values[0]
        response = await llm_disambiguation(result, gt_color, gt_type)
        try:
            index = int(response)
        except ValueError:
            return {"error": "Disambiguation failed."}
        return self.bboxes[index]



async def generate_plan(data: pd.DataFrame, llm_name: str):
    output_data = {}
    with open("prompts/visual_agent.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    async def process_row(index, row):
        async with semaphore:
            result = {
                "question_id": row["question_id"],
                "question": row["question"],
                "gt": row["gt"],
                "image_id": row["image_id"],
            }
            try:
                prompt = custom_format(
                    prompt_template, {"user_input": result["question"]}
                )
                client = LanguageModelClient(model_name=llm_name)
                response = await client.send_request_to_server(prompt=prompt)
                actions = parse_action_from_text(response)
                result["actions"] = actions
            except Exception as e:
                result["actions"] = {"error": str(e)}

            return index, result

    tasks = [process_row(index, row) for index, row in data.iterrows()]
    for index, result in await tqdm.gather(*tasks, desc="Processing rows"):
        output_data[index] = result

    return output_data



async def process_single_action(index, row):
    async with semaphore:
        action_executor = ActionExecutor(
            action=row["actions"],
            img_path=os.path.join(image_path, row["image_id"]),
            query=row["question"],
            index=row["question_id"],
        )
        await action_executor.execute()
        return index, action_executor.response


async def process_action(action_dict: Dict):
    tasks = [process_single_action(index, row) for index, row in action_dict.items()]
    for index, response in await tqdm.gather(*tasks, desc="Processing actions"):
        action_dict[index]["response"] = response
    return action_dict


if __name__ == "__main__":
    # generate planning
    data = pd.read_json(gt_json, lines=True)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(generate_plan(data, llm_name))
    with open(action_json, "w") as f:
        json.dump(results, f, indent=4)

    # execute planning
    # with open("action_json/output_task2_internvl2_8b.json", "r", encoding="utf-8") as f:
    #     action_dict = json.load(f)
    loop = asyncio.get_event_loop()
    answer_dict = loop.run_until_complete(process_action(results))
    with open(answer_json, "w", encoding="utf-8") as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    # evaluate
    # with open("execute_json/answer_task2_internvl2_8b.json", "r", encoding="utf-8") as f:
    #     answer_dict = json.load(f)

    data = pd.read_json(gt_json, lines=True)

    final_dict = {}
    q1_true = 0
    q2_true = 0
    q3_true = 0
    q1_false = 0
    q2_false = 0
    q3_false = 0

    for index, row in answer_dict.items():
        question = row["question"]
        for r in data.iterrows():
            if str(r[1]["question_id"]) == str(index):
                data_row = r[1]
                break
        q_type = data_row["qtype"]
        gt = data_row["bbox"]

        answer = ""
        try:
            last_action = row["actions"][-1]["action_type"]
            tmp_answer = row["response"][last_action]
        except Exception as e:
            tmp = {
                "question": question,
                "gt": gt,
                "answer": answer,
                "evaluate": False,
            }
            final_dict[index] = tmp
            if q_type == "1p":
                q1_false += 1
            elif q_type == "2p":
                q2_false += 1
            else:
                q3_false += 1
            continue

        if isinstance(tmp_answer, list):
            answer = tmp_answer

        # 
        tmp = {
            "question": question,
            "gt": gt,
            "answer": answer,
            "evaluate": True if str(answer) == str(gt) else False,
            "number": q_type,
        }
        if q_type == "1p":
            if tmp["evaluate"]:
                q1_true += 1
            else:
                q1_false += 1
        elif q_type == "2p":
            if tmp["evaluate"]:
                q2_true += 1
            else:
                q2_false += 1
        else:
            if tmp["evaluate"]:
                q3_true += 1
            else:
                q3_false += 1
        final_dict[index] = tmp

    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)


    print(f"p1_true: {q1_true}, p1_false: {q1_false}, Acc: {q1_true / (q1_true + q1_false)}")
    print(f"p2_true: {q2_true}, p2_false: {q2_false}, Acc: {q2_true / (q2_true + q2_false)}")
    print(f"p3_true: {q3_true}, p3_false: {q3_false}, Acc: {q3_true / (q3_true + q3_false)}")
