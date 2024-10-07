# -*- coding: utf-8 -*-
# @Time    : 2024/8/11 上午10:07
# @Author  : Yancan Chen
# @Email   : yancan@u.nus.edu
# @File    : all_process_gpt4_clip.py

import pandas as pd
import faiss
import torch
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import base64
import requests
import json
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY =""


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)


def encode_text(query: list, model, processor):
    # the max_length for clip is 75
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True, max_length=72)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if hasattr(model, 'module'):
        # If it's wrapped, access methods through the 'module' attribute
        text_embedding = model.module.get_text_features(**inputs)
    else:
        # If it's not wrapped, access methods directly
        text_embedding = model.get_text_features(**inputs)
    text_embedding = text_embedding.cpu()
    return text_embedding

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def retrieve_topk_images(query: list,
                         topk=10,
                         faiss_model=None,
                         blip_model=None,
                         id2image=None,
                         processor=None, ):
    query_vec = encode_text(query=query, model=blip_model, processor=processor)
    query_vec = query_vec.to(torch.float32).detach().numpy()
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    distance, indices = faiss_model.search(query_vec, topk)
    indices = np.array(indices)
    image_paths = [[id2image.get(idx, 'path/not/found') for idx in row] for row in indices]
    return image_paths, indices, distance


# Find the ranking of target images
def find_index_in_list(element, my_list):
    return my_list.index(element) if element in my_list else 50000

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json(text):
    pattern = r'{.*}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group()
    else:
        return 'parse incorrectly'


#回答
def answer(question,base64_target_image):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"according to the image, answer the question:{question}，Your answer must be direct and simple",
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_target_image}",
              },
            }
          ],
        }
      ],
      max_tokens=100,
    )
    return response.choices[0].message.content

#总结
def summary(query, question, answer):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"""
        Your task is to summarize the information from the image's question and answer and add this information to the original image description.
        Remember: the summarized information must be concise, and the original description should not be altered.

        <question>
        {question}
        <answer>
        {answer}
        <image description>
        {query}
        The information extracted from the question and answer should be added to the original description as an attribute or a simple attributive clause.
        """
        },
        {"role": "user", "content": ""}
    ])
    return response.choices[0].message.content


def question_attribute_gpt4(query):
    client = OpenAI(api_key=API_KEY)
    prompt = f"""Ask a new question in the following dialog, assume that the questions are designed to 
        help us retrieve this image from a large collection of images:
          Caption: 2 full grown zebras standing by a brick building with a steel door
          Question: is this picture in color?
          Answer: yes
          Question: do you see people?
          Answer: no
          Question: are the animals in a pen?
          Caption: a group of people standing on a snowy slope
          Question: Are there any trees visible in the background of the image?
          Answer: no
          Question: How many people are in the group?
          Answer: four

          Caption: {query}
          Your question is: 
                    """
    message = [{"type": "text", "text": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        max_tokens=200,
    )
    output_text = response.choices[0].message.content
    return output_text


def chat_search_step1(recall_res, output_file_path_step1, sampling_strategy,
                      image_path_base_dir, faiss_model, blip_model, blip_processor):
    """
    Run LLaVA-Next-Video in the AutoDL system,
    :return:
    """
    save_data = []
    start_index = 0  # 处理的最后一条数据索引 # 685
    for i in tqdm(range(start_index, len(recall_res))):
        query = recall_res.loc[i, 'option']
        target_image_path = image_path_base_dir + recall_res.loc[i, 'target_image']
        image_rank = recall_res.loc[i, 'image_rank']
        # try:
        # Generate question using Pllava
        images_path, indices, query_distance = retrieve_topk_images([query],
                                                                    topk=100 if sampling_strategy == 'clustering' else 50,
                                                                    faiss_model=faiss_model,
                                                                    blip_model=blip_model,
                                                                    id2image=id2image,
                                                                    processor=blip_processor)
        images_path, indices, query_distance = images_path[0], indices[0], query_distance[0]  # 消除batch
        top50_query_distance = query_distance[49]
        # 计算ranking
        # Downsample retrieved image
        # 提问题
        question= question_attribute_gpt4(query)

        record_dict = {
                       'top50_query_distance': str(top50_query_distance),  # 排名在第50位的图片的余弦数值
                       'query_distance': str(list(query_distance)),  # query在召回图片中的距离
                       'one_tune_questions': question,
                       'target_image_path': target_image_path,
                       'image_rank': int(image_rank),
                       'option': query
                       }
        # print(record_dict)
        save_data.append(record_dict)
        # 跑完一条数据后写入文件中
        with open(output_file_path_step1, 'a') as f:
            f.write(json.dumps(record_dict) + '\n')
        # except:
        #     print('-' * 130, 'passing')
        #     print(query)
        #     pass


def chat_search_step2(recall_res, output_file_path_step2, faiss_model, blip_model, blip_processor):
    """
    Connect the ChatGPT
    :return:
    """
    save_data = []
    start_index = 0
    top10 = better = 0
    for i in tqdm(range(start_index, len(recall_res))):
        target_image_path = recall_res[i]['target_image_path']
        question = recall_res[i]['one_tune_questions']
        query = recall_res[i]['option']
        image_rank = recall_res[i]['image_rank']

        # 回答问题
        base64_target_image = encode_image(target_image_path)
        answer_of_question = answer(question, base64_target_image)
        # 总结和/hpctmp/e1143641/imageRetrieval/chat_search_suggestionquery 和 Q-A 问答对
        summary_of_question_and_option = summary(query, question, answer_of_question)  # CLIP max_length = 70

        # 查看扩写query之后的图片排名
        image_paths_new, _, summary_distance = retrieve_topk_images([summary_of_question_and_option],
                                                                    topk=10000,
                                                                    faiss_model=faiss_model,
                                                                    blip_model=blip_model,
                                                                    id2image=id2image,
                                                                    processor=blip_processor, )
        image_rank_new = find_index_in_list(target_image_path, image_paths_new[0])

        is_top10 = is_better = 0
        if image_rank_new <= 10:
            top10 += 1
            is_top10 = 1
        if image_rank_new < image_rank:
            better += 1
            is_better = 1

        # add more information
        top50_query_distance = recall_res[i]['top50_query_distance']
        query_distance = recall_res[i]['query_distance']

        record_dict = {
            'top50_query_distance': top50_query_distance,  # 排名在第50位的图片的余弦数值
            'query_distance': query_distance,  # query在召回图片中的距离
            'one_tune_questions': question,
            'image_rank': image_rank,
            'target_image_path': target_image_path,
            'option': query,
            'answer_of_question': answer_of_question,
            'summary_of_question_and_option': summary_of_question_and_option,
            'image_rank_new': int(image_rank_new),
            'is_top10': int(is_top10),
            'is_better': int(is_better)
        }
        # print(record_dict)
        save_data.append(record_dict)
        # 跑完一条数据后写入文件中
        with open(output_file_path_step2, 'a') as f:
            f.write(json.dumps(record_dict) + '\n')


def load_image_faiss(faiss_index_path, id2image_path, image_vector_path):
    faiss_model = faiss.read_index(faiss_index_path)
    with open(id2image_path, 'rb') as f:
        id2image = pickle.load(f)
    with open(image_vector_path, 'rb') as f:
        image_vector = pickle.load(f)
    return faiss_model, id2image, image_vector

def load_models_processors_clip(blip_model_id):

    clip_processor = CLIPProcessor.from_pretrained(blip_model_id)
    clip_model = CLIPModel.from_pretrained(blip_model_id,
                                           torch_dtype=torch.bfloat16,
                                           device_map="auto")
    clip_model.eval()

    # 打印显存使用情况
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

    return clip_processor, clip_model

if __name__ == "__main__":
    seed_everything(seed=42)

    # Modify the model path
    dataset_name = 'visdial' # 'visdial'
    dataset_fold = 'css_data' #'css_data'
    faiss_index_path = f'./checkpoints/clip_faiss_{dataset_name}.index'
    id2image_path = f'./checkpoints/id2image_clip_{dataset_name}.pickle'
    id2image_path = f'./checkpoints/id2image_clip_{dataset_name}.pickle'
    id2image_path = f'./checkpoints/id2image_clip_{dataset_name}.pickle'
    image_vector_path = f'./checkpoints/clip_image_embedding_{dataset_name}.pickle'
    sampling_strategy =''
    # Load data
    recall_res = pd.read_csv(f'./playground/data/{dataset_fold}/{dataset_name}_image_captions_with_top10_tag.csv')
    output_file_path_step1 = f'./rank_res/step1_clip_gpt4o_{dataset_name}_prompt_only.jsonl'
    output_file_path_step2 = f'./rank_res/clip_gpt4o_{dataset_name}_prompt_only.jsonl'
    image_path_base_dir = f'./playground/data/{dataset_fold}/'
    llava_model_id =  'lmsys/vicuna-7b-v1.5'
    blip_model_id = 'openai/clip-vit-large-patch14-336'  # clip or blip
    run_autodl = False  # Generate question
    run_nushpc = True  # Connect to ChatGPT
    absence_of_data_chat_step1 = False  # 跑出来的数据
    absence_of_data_chat_step2 = False

    clip_processor, clip_model = load_models_processors_clip(blip_model_id)
    faiss_model, id2image, image_vector = load_image_faiss(faiss_index_path, id2image_path, image_vector_path)
    recall_res = recall_res[recall_res['is_top10'] == 0].reset_index()

    if absence_of_data_chat_step1:
        add_recall_res = read_jsonl(output_file_path_step1)
        add_recall_res = pd.DataFrame(add_recall_res)['option'].tolist()
        recall_res = recall_res[~recall_res['option'].isin(add_recall_res)].reset_index()

    # chat_search_step1 runs on the AutoDL platform
    if run_autodl:
        chat_search_step1(recall_res, output_file_path_step1, sampling_strategy,
                          image_path_base_dir=image_path_base_dir, faiss_model=faiss_model, blip_model=clip_model,
                          blip_processor=clip_processor)
    if absence_of_data_chat_step2:
        recall_res = pd.DataFrame(read_jsonl(output_file_path_step1))
        add_recall_res = read_jsonl(output_file_path_step2)
        add_recall_res = pd.DataFrame(add_recall_res)['option'].tolist()
        recall_res = recall_res[~recall_res['option'].isin(add_recall_res)].reset_index(drop=True).to_dict(
            orient='records')

    if run_nushpc:
        if absence_of_data_chat_step2:
            recall_res = read_jsonl(output_file_path_step2)
        else:
            recall_res = read_jsonl(output_file_path_step1)
        chat_search_step2(recall_res, output_file_path_step2, faiss_model=faiss_model,
                          blip_model=clip_model, blip_processor=clip_processor)

"""
python all_process_gpt4_clip_promptonly.py
"""
