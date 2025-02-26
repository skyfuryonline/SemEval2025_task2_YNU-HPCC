import json

SPLIT = ["sample","validation"]
dataPath = "ea-mt-eval/data/references/"
langList = ["ja_JP",'ar_AE','de_DE','es_ES','fr_FR','it_IT','ko_KR','th_TH','tr_TR','zh_TW']
def generate_parallel_corpus(data,field='source'):
    return [
    {"source": item["source"],"source_locale":item["source_locale"],"target_locale":item["target_locale"],"targets": target["translation"]}
    for item in data
    for target in item["targets"]
    ]

def getSourceFile():
    whole_data = []
    for l in langList:
        with open(dataPath+"sample"+f"/{l}.jsonl", 'r', encoding='utf-8') as data_file:
            tmp = [json.loads(line.strip()) for line in data_file]
            for i in tmp:
                whole_data.append(i)
        with open(dataPath+"validation"+f"/{l}.jsonl", 'r', encoding='utf-8') as data_file:
            tmp = [json.loads(line.strip()) for line in data_file]
            for i in tmp:
                whole_data.append(i)
    return whole_data


data = getSourceFile()
corpus = generate_parallel_corpus(data)

with open('all_data_for_llama.jsonl', 'w', encoding='utf-8') as output_file:
    for idx, item in enumerate(corpus):
        merged_data = {
            "source_locale": item["source_locale"],
            "target_locale": item["target_locale"],
            "source": item["source"],
            "target": item["targets"]
        }
json.dump(merged_data, output_file, ensure_ascii=False)
output_file.write('\n')



