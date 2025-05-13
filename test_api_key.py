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
    "authorization": "Bearer cfb73dca729628106d333426a418ca08ace664ad6f0512e31c130d3b765240ff"
}

response = requests.get(url, headers=headers)
for model in response.json():
    print(model["id"])
print("--------------------------------")
print("--------------------------------")
