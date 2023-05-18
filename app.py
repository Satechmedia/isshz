"""
Created on Thur May 18 2:00pm (WAT)

@author: Sam Olubunmi
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    inputText: str
    healthCheck: str


# loading the saved model
model_id = pickle.load(open('dreamlike-AI-1.0.pkl','rb'))


@app.get('/')
def about():
    healthCheck = "Version 1.0, Health 100%"
    return healthCheck


# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "3D, a female detective robot stading in front of big company, bright cinematic lighting, gopro"
# image = pipe(prompt).images[0]

# image.save("./result1.jpg")

@app.post('/generate')
def generate(inputText):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = inputText
    image = pipe(prompt).images[0]
    result = image.save("./result.jpg")
    return  result
