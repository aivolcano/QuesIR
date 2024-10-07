# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 下午3:04
# @Author  : Yancan Chen
# @Email   : yancan@u.nus.edu
# @File    : all_process_without_summary.py

import pandas as pd
import faiss
import torch
import numpy as np
import pickle
from tqdm import tqdm
import random
from transformers import CLIPProcessor, CLIPModel, BitsAndBytesConfig
from collections import defaultdict
import json
import  argparse
import time
import os


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

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


def load_models_processor_clip(blip_model_id):
    clip_processor = CLIPProcessor.from_pretrained(blip_model_id)
    clip_model = CLIPModel.from_pretrained(blip_model_id, torch_dtype=torch.bfloat16,
                                           device_map="auto")
    clip_model.eval()
    return clip_processor, clip_model

def load_image_faiss(faiss_index_path, id2image_path, image_vector_path):
    faiss_model = faiss.read_index(faiss_index_path)
    with open(id2image_path, 'rb') as f:
        id2image = pickle.load(f)
    with open(image_vector_path, 'rb') as f:
        image_vector = pickle.load(f)
    return faiss_model, id2image, image_vector


def retrieve_topk_images(query: list,
                         topk=10,
                         faiss_model=None,
                         blip_model=None,
                         id2image=None,
                         processor=None, ):
    query_vec = encode_text(query=query, model=blip_model, processor=processor)
    # print(query_vec.dtype)
    query_vec = query_vec.to(torch.float32).detach().numpy()
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    distance, indices = faiss_model.search(query_vec, topk)
    indices = np.array(indices)
    image_paths = [[id2image.get(idx, 'path/not/found') for idx in row] for row in indices]
    return image_paths, indices, distance


# Find the ranking of target images
def find_index_in_list(element, my_list):
    return my_list.index(element) if element in my_list else 50000



def chat_search_step3(recall_res, output_file_path_step2, faiss_model, blip_model, blip_processor):
    """
    Connect the ChatGPT
    :return:
    """
    save_data = []
    start_index = 0
    top10 = better = 0
    for i in tqdm(range(start_index, len(recall_res))):
        target_image_path = recall_res[i]['target_image_path']
        if not target_image_path.startswith('./playground'):
            target_image_path = './playground/data/css_data/' + target_image_path
        question = recall_res[i]['one_tune_questions'].strip()
        query = recall_res[i]['option'].strip()
        image_rank = recall_res[i]['image_rank']
        # try:
        # 回答问题
        # base64_target_image = encode_image(target_image_path)
        # answer_of_question = answer(question, base64_target_image)
        answer_of_question = recall_res[i]['answer_of_question'] # load the answer from the dataset
        # 总结和/hpctmp/e1143641/imageRetrieval/chat_search_suggestionquery 和 Q-A 问答对
        summary_of_question_and_option = query + ' ' + query + ' ' + answer_of_question

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

        if args.sampling_strategy == 'prompt_only':
            record_dict = {
                # 'downsampled_images_path': sampled_image_paths,
                # 'sampled_distances': sampled_distances,  # 采样出的图片的距离
                # 'top50_query_distance': recall_res[i]['top50_query_distance'],  # 排名在第50位的图片的余弦数值
                # 'query_distance': recall_res[i]['query_distance'],  # query在召回图片中的距离
                'one_tune_questions': question,
                # 'common_entities': common_entities,
                # 'diff_entities': diff_entities,
                'image_rank': image_rank,
                'target_image_path': target_image_path,
                'option': query,
                'answer_of_question': answer_of_question,
                'summary_of_question_and_option': summary_of_question_and_option,
                'image_rank_new': int(image_rank_new),
                'is_top10': int(is_top10),
                'is_better': int(is_better)
            }
        else:
            # add more information
            sampled_image_paths = recall_res[i]['downsampled_images_path']
            sampled_distances = recall_res[i]['sampled_distances']
            top50_query_distance = recall_res[i]['top50_query_distance']
            query_distance = recall_res[i]['query_distance']
            common_entities = recall_res[i]['common_entities']
            diff_entities = recall_res[i]['diff_entities']

            record_dict = {
                           'downsampled_images_path': sampled_image_paths,
                           'sampled_distances': sampled_distances,  # 采样出的图片的距离
                           'top50_query_distance': top50_query_distance,  # 排名在第50位的图片的余弦数值
                           'query_distance': query_distance,  # query在召回图片中的距离
                           'one_tune_questions': question,
                           'common_entities': common_entities,
                           'diff_entities': diff_entities,
                           'image_rank': image_rank,
                           'target_image_path': target_image_path,
                           'option': query,
                           'answer_of_question': answer_of_question,
                           'summary_of_question_and_option': summary_of_question_and_option,
                           'image_rank_new': int(image_rank_new),
                           'is_top10': int(is_top10),
                           'is_better': int(is_better)
                           }
        save_data.append(record_dict)
        # 跑完一条数据后写入文件中
        with open(output_file_path_step2, 'a') as f:
            f.write(json.dumps(record_dict) + '\n')
        # except:
        #     pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sample script")

    # 添加参数
    parser.add_argument('--sampling_strategy', type=str, choices=['random_sampled', 'interval', 'clustering', 'prompt_only','gpt4o'],
                        default='clustering', help="Choose the sampling strategy")
    parser.add_argument('--dataset_name', type=str, choices=['visdial', 'mscoco', 'flickr30k'], default='visdial',
                        help="Select the dataset name")
    parser.add_argument('--dataset_fold', type=str, choices=['css_data', 'mscoco', 'flickr30k'], default='css_data',
                        help="Specify the dataset fold")

    return parser.parse_args()


if __name__ == '__main__':
    start = time.time()
    seed_everything(seed=42)

    args = parse_arguments()

    # 使用这些参数
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Dataset fold: {args.dataset_fold}")

    # Modify the model path
    dataset_name = args.dataset_name  # 'mscoco'
    dataset_fold = args.dataset_fold  # 'mscoco'
    faiss_index_path = f'./checkpoints/clip_faiss_{dataset_name}.index'
    id2image_path = f'./checkpoints/id2image_clip_{dataset_name}.pickle'
    image_vector_path = f'./checkpoints/clip_image_embedding_{dataset_name}.pickle'
    # Load data
    recall_res = pd.read_csv(f'./playground/data/{dataset_fold}/{dataset_name}_image_captions_with_top10_tag.csv')
    sampling_strategy = args.sampling_strategy  # 'topk_cos_similiarity' 'interval' 'clustering'
    output_file_path_step1 = f'./rank_res/step1_clip_pllava-7b_{dataset_name}_{sampling_strategy}.jsonl'
    if args.sampling_strategy == 'prompt_only':
        output_file_path_step2 = f'./rank_res/clip_vicuna-7b_{dataset_name}_{sampling_strategy}.jsonl'
        output_file_path_step3 = f'./rank_res/clip_vicuna-7b_{dataset_name}_{sampling_strategy}_cqa_concat.jsonl'
    elif args.sampling_strategy == 'gpt4o':
        output_file_path_step2 = f'./rank_res/clip_gpt4o_{dataset_name}_interval.jsonl'
        output_file_path_step3 = f'./rank_res/clip_gpt4o_{dataset_name}_interval_cqa_concat.jsonl'
    else:
        output_file_path_step2 = f'./rank_res/clip_pllava-7b_{dataset_name}_{sampling_strategy}.jsonl'
        output_file_path_step3 = f'./rank_res/clip_pllava-7b_{dataset_name}_{sampling_strategy}_cqa_concat.jsonl'


    llava_model_id = 'MODELS/pllava-7b'
    blip_model_id = 'openai/clip-vit-large-patch14-336'  # clip or blip
    run_autodl = False  # Generate question
    run_nushpc = False  # Connect to ChatGPT
    run_summary_strategy = True  # concat the query question answer directly
    error_append = False

    # llava_model, llava_processor, clip_processor, clip_model = load_models_processors(llava_model_id, blip_model_id)
    # if run_autodl:
    #     llava_processor, llava_model = load_models_processor_llava(llava_model_id)
    clip_processor, clip_model = load_models_processor_clip(blip_model_id)
    faiss_model, id2image, image_vector = load_image_faiss(faiss_index_path, id2image_path, image_vector_path)
    recall_res = recall_res[recall_res['is_top10'] == 0].reset_index()
    # if error_append:
    #     non_recall_res = pd.DataFrame(read_jsonl(output_file_path_step2))
    #     recall_res = recall_res[~recall_res['option'].isin(non_recall_res['option'].to_list())].reset_index()


    if run_summary_strategy:
        recall_res = read_jsonl(output_file_path_step2)
        chat_search_step3(recall_res, output_file_path_step3, faiss_model=faiss_model,
                          blip_model=clip_model, blip_processor=clip_processor)

"""
python all_process_without_summary.py --sampling_strategy interval --dataset_name mscoco --dataset_fold mscoco
python all_process_without_summary.py  --sampling_strategy random_sampled --dataset_name mscoco --dataset_fold mscoco
python all_process_without_summary.py  --sampling_strategy clustering --dataset_name mscoco --dataset_fold mscoco

python all_process_without_summary.py  --sampling_strategy interval --dataset_name flickr30k --dataset_fold flickr30k
python all_process_without_summary.py  --sampling_strategy random_sampled --dataset_name flickr30k --dataset_fold flickr30k
python all_process_without_summary.py  --sampling_strategy clustering --dataset_name flickr30k --dataset_fold flickr30k


python all_process_without_summary.py --sampling_strategy interval --dataset_name visdial --dataset_fold css_data
python all_process_without_summary.py  --sampling_strategy random_sampled --dataset_name visdial --dataset_fold css_data
python all_process_without_summary.py  --sampling_strategy clustering --dataset_name visdial --dataset_fold css_data

python all_process_without_summary.py --sampling_strategy prompt_only --dataset_name visdial --dataset_fold css_data
python all_process_without_summary.py --sampling_strategy prompt_only --dataset_name flickr30k --dataset_fold flickr30k
python all_process_without_summary.py --sampling_strategy prompt_only --dataset_name mscoco --dataset_fold mscoco

python all_process_without_summary.py --sampling_strategy gpt4o --dataset_name visdial --dataset_fold css_data
python all_process_without_summary.py --sampling_strategy gpt4o --dataset_name flickr30k --dataset_fold flickr30k
python all_process_without_summary.py --sampling_strategy gpt4o --dataset_name mscoco --dataset_fold mscoco

"""
