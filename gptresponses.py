import random
import json
from transformers import pipeline
def processData(database):
    processed = []
    for x in database["data"]:
        for y in x["paragraphs"]:
            initi = y["context"]
            for qa in y["qas"]:
                qe = qa["question"]
                if len(qa["answers"]) > 0:
                    ans = qa["answers"][0]["text"]
                    processed.append({"question": qe, "context": initi, "human_answer": ans})
    return processed
def gpt(model, data):
    for i in range(len(data)):
        instance = data[i]
        qe = instance["question"]
        dataAnswer = instance["human_answer"]
        gpt_answer = model(question=qe, context=instance["context"])["answer"]
        if i < 100:
            print(f"Question {i+1}: {qe}")
            print(f"Human Answer: {dataAnswer}")
            print(f"GPT Answer: {gpt_answer}\n")
        if i == 100:
            break
with open("train-v2.0.json", "r") as f:
    database = json.load(f)
data = processData(database)
random.shuffle(data)
model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
gpt(model, data)