import openai
import json


path_ids_selected = "../clinical_qa_/data/synthesized_tests/ids_30.json"
api_key_path = '../recsum_/za/api_key.txt'

with open(path_ids_selected) as f:
    ids_selected = json.load(f)

with open(api_key_path) as f:
    openai.api_key = f.read().strip()


def generate_from_davinci(prompt):
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=500)
    return response["choices"][0]['text'].strip().replace('\n', ' ')


def generate_from_turbo(messages):
    response = openai.ChatCompletion.create(model=" ", messages=messages)
    return response['choices'][0]["message"]['content']




