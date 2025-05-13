import json
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # try load_dotenv(override=True) for different behavior


# OpenAI Models
from openai import OpenAI

client = OpenAI()

models = client.models.list()

print("--------------------------------")
print("--------OpenAI Models:----------")
print("--------------------------------")
for model in models:
    print(model.id)
print("--------------------------------")
print("--------------------------------")


# Together Models
import requests

print("--------------------------------")
print("--------Together Models:---------")
print("--------------------------------")

url = "https://api.together.xyz/v1/models"

headers = {
    "accept": "application/json",
    "authorization": "Bearer " + os.getenv("TOGETHER_API_KEY")
}

response = requests.get(url, headers=headers)
for model in response.json():
    print(model["id"])
print("--------------------------------")
print("--------------------------------")
