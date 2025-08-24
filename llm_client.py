# Description: This file is used to test the language model client.
import requests
from PIL import Image
import aiohttp
import asyncio
import json

from utils.tools import custom_format, parse_action_from_text
import yaml
import base64

with open("config.yml", "r", encoding="utf-8") as file:
    llm_server_config = yaml.safe_load(file)

class LanguageModelClient:
    def __init__(self, model_name: str):

        self.model_name = llm_server_config[model_name].get("model_name")
        self.openai_api_key = llm_server_config[model_name].get("openai_api_key")
        self.target_url = llm_server_config[model_name].get("target_url")
        # self.openai_api_key = "sk-2e76b1116d274cacafe4b0971f07aac1"
        # self.target_url = "https://api.deepseek.com"
        # self.model_name = "qwen2.5-3b-instruct"
        # self.target_url = "https://api.302.ai/v1/chat/completions"
        # self.openai_api_key = "sk-ZpBa2ulJG0wsaI9mlLGdsVDgUiWpDRDzooyMHqXsrXQOy0Gb"

        self.temperature = llm_server_config.get("temperature", 0.9)
        self.top_p = llm_server_config.get("top_p", 0.9)
        self.max_tokens = llm_server_config.get("max_tokens", 500)

    async def send_request_to_server(self, prompt: str, img_path: str=None):

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        if img_path:
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": self.max_tokens,
                }
        else:
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.target_url, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    response_json = await response.json()
                    return response_json["choices"][0]["message"]["content"]
                else:
                    raise Exception(
                        f"Server returned status code {response.status}, response: {await response.text()}"
                    )
