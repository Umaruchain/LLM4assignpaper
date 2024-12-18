import torch
from llm2vec import LLM2Vec

from location import parse_jsonl

import pandas as pd

def extract_oral_presentations(csv_file):
    # 读取CSV文件
    oral_df = pd.read_excel(csv_file)
    
    # 筛选出Decision列中包含"ORAL"的行
    #oral_df = df[df['Decision'].str.contains('ORAL', na=False)]
    
    #oral_df.to_csv('hksts_oral.csv', index=False)

    # 初始化一个空列表来存放结果
    results = []
    
    # 遍历每一条记录
    for _, row in oral_df.iterrows():
        # 格式化文章描述
        article_description = f"Title:\n{row['Title']}\n\n" \
                              f"Topics:\n{row['Topics']}\n\n" \
                              f"Keywords:\n{row['Keywords']}\n\n" \
                              f"Abstract:\n{row['Abstract']} \n\n"
        
        # 将格式化好的字符串添加到结果列表中
        results.append(article_description)
    
    return results




def append_instruction(instruction,text_lists):
    res = []
    for text in text_lists:
        res.append([instruction, text,0])
    return res

# l2v = LLM2Vec.from_pretrained(
#     "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
#     peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
#     device_map="cuda" if torch.cuda.is_available() else "cpu",
#     torch_dtype=torch.bfloat16,
# )

# Encoding queries using instructions
instruction = (
    "Identify the topic or theme of the given conference papers:"
)


queries_tmp = extract_oral_presentations('./oral_all_demo.xlsx')
#queries,pos_list = parse_jsonl('./GeoLLM/prompts/bay_area_prompts.jsonl')
queries = append_instruction(instruction,queries_tmp)
print('papers: ', len(queries))

# import pdb
# pdb.set_trace()
# queries = queries[:50]
#print('papers: ', len(queries))
# import pdb
# pdb.set_trace()


l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

import numpy as np
# clustering_batch_size = 200
q_reps_norm = torch.nn.functional.normalize(l2v.encode(queries), p=2, dim=1)

corpus_embeddings = np.asarray(q_reps_norm.cpu())

np.save('hksts.npy',corpus_embeddings)


# print("Fitting Mini-Batch K-Means model...")
# clustering_model = sklearn.cluster.MiniBatchKMeans(
#     n_clusters=20, batch_size=clustering_batch_size
# )
# clustering_model.fit(corpus_embeddings)
# cluster_assignment = clustering_model.labels_
