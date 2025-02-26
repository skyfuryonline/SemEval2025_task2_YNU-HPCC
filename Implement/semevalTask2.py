import openai
openai.api_key="your api_key"
from openai import OpenAI
import os.path
import ast
from transformers import pipeline
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from fuzzywuzzy import fuzz


langList = ['ar_AE','de_DE','fr_FR','es_ES','it_IT','ko_KR','th_TH','tr_TR','zh_TW',"ja_JP"]
lang = "fr_FR"

 pipe = pipeline("token-classification", model="dslim/bert-large-NER")
 m2m =  pipeline("text2text-generation", model="facebook/m2m100_418M")

api_url = 'your api_url'
api_key_QwenMax = 'your api_key'

CountryDict={
    "ar_AE":"ar",
    "de_DE":"de",
    "es_ES":"es",
    "fr_FR":"fr",
    "it_IT":"it",
    "ja_JP":"ja",
    "ko_KR":"ko",
    "th_TH":"th",
    "tr_TR":"tr",
    "zh_TW":"zh"
}

CountryDict2 = {
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "tr": "Turkish",
    "zh": "Chinese (Traditional)"
}

cache = {}


def clean_translated_name(translated_name):
    """
    """
    if not translated_name:
        return None
    
    cleaned_name = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", translated_name)
    
    cleaned_name = re.sub(r"（.*?）|【.*?】", "", cleaned_name)
    
    cleaned_name = cleaned_name.strip()
   
    cleaned_name = re.sub(r"[\"'“”‘’]", "", cleaned_name)
   
    cleaned_name = re.sub(r"\s+", " ", cleaned_name)
   
    cleaned_name = re.sub(r"[#$%@&]", "", cleaned_name)
    
    cleaned_name = re.sub(r"[.,;!?]", "", cleaned_name)
    return cleaned_name

def translate_with_slidingWindow(ne,translated_sentence,translated_name):
    
    lne = len(ne)
    max_similarity = 0
    best_start = 0
    best_end = 0
    
    for i in range(len(translated_sentence) - lne + 1):
        window = translated_sentence[i:i + lne]
        similarity = fuzz.ratio(window, ne)

       
        if similarity > max_similarity:
            max_similarity = similarity
            best_start = i
            best_end = i + lne

    
    while best_start > 0 and translated_sentence[best_start - 1] not in [' ', '\n', '\t']:
        best_start -= 1

    
    while best_end < len(translated_sentence) and translated_sentence[best_end] not in [' ', '\n', '\t']:
        best_end += 1
    
    translated_sentence = (translated_sentence[:best_start] +
                           translated_name +
                           translated_sentence[best_end:])
    return translated_sentence


def crawl(sentence,targetLang):
    ner_result = pipe(sentence)
    
    translated_sentence = sentence
    try:
        entityList = merge_entities(ner_result)
    
    except Exception as e:
        return translated_sentence

    
    for entity in entityList:
        entity_name = entity["entity"]
        label = entity["label"]

        # Wizard ' s  First Rule
        entity_name = re.sub(r'\s([.,;!?()[]{}])', r'\1', entity_name)
        entity_name = re.sub(r'([.,;!?()[]{}])\s', r'\1', entity_name)

      
       
        ne,translated_name = query_wikidata_translation(entity_name, targetLang)

        
        if translated_name is None:
            return translated_sentence
        if ne is None:
            return  translated_sentence
        if ne== "":
            return translated_sentence
        if translated_name == "":
            return translated_sentence
       
        elif entity_name in translated_name:
            return translated_sentence

        if translated_name:
           
            ne = clean_translated_name(ne)

           
            translated_sentence = translate_with_slidingWindow(ne,translated_sentence,translated_name)

    return translated_sentence


def query_wikidata_translation(entity_name, target_lang):
   
    search_url = f"https://www.wikidata.org/w/index.php?search={entity_name}&ns0=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Accept-Language": target_lang,
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to search {entity_name} on Wikidata.")
        return None,None

    soup = BeautifulSoup(response.text, "html.parser")

 
    results = soup.select(".mw-search-results a")  #
    
    states = soup.select(".mw-search-results .mw-search-result-data")

    if len(results)==0:
        return None,None
    
    best_match = None
    highest_similarity = 0
    ne = None
    for result,stateCount in zip(results,states):
        text = result.get_text()
        
        text = clean_translated_name(text)

        
        stat = stateCount.get_text()
       
        match = re.search(r"(\d+)\s+statements,\s+(\d+)\s+sitelinks", stat)
        statements_count = 0
        sitelinks_count = 0
        if match:
            statements_count = int(match.group(1))
            sitelinks_count = int(match.group(2))

        
        similarity =0.2*statements_count+0.3*sitelinks_count+0.5*fuzz.ratio(entity_name.lower(), text.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = result
            ne = text

    
    if best_match:
       link = best_match
    else:
        link = None
   
    if not link:
        print(f"No results found for {entity_name}.")
        return None,None

    entity_url = "https://www.wikidata.org" + link["href"]+f"?uselang={target_lang}"

    
    response = requests.get(entity_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to access the entity page for {entity_name}.")
        return None,None

    
    soup = BeautifulSoup(response.text, "html.parser")

    lang_section =soup.find('span', class_="wikibase-title-label")
   

    if lang_section:
        
        return ne,lang_section.get_text()

    print(f"No translation found for {entity_name} in {target_lang}.")
    return None,None



def getSourceFile():
   
    with open(jsonl_file, 'r', encoding='utf-8') as data_file:
        data = [json.loads(line.strip()) for line in data_file]
    return data


def getTranslationFile():
    with open(output_file, 'r', encoding='utf-8') as translation_file:
        translations = translation_file.readlines()
    return translations


def generateSubmitFile():

    data = getSourceFile()
   
    translations = getTranslationFile()

    with open(save_jsonl_file, 'w', encoding='utf-8') as output_file:
        for idx, item in enumerate(data):
          
            translation = translations[idx].strip()
            merged_data = {
                "id": item["id"],
                "source_language": item["source_locale"],
                "target_language": item["target_locale"],
                "text": item["source"],
                "prediction": translation
            }
            json.dump(merged_data, output_file, ensure_ascii=False)
           
            output_file.write('\n')


def translate_sentence_with_m2m(sentence,targetLang,sourceLang='en'):
   
    swithSentence = crawl(sentence, targetLang=targetLang)
    m2m.tokenizer.src_lang = sourceLang
    
    m2m.generation_config.forced_bos_token_id = m2m.tokenizer.get_lang_id(targetLang)
    translation = m2m(swithSentence)
    return translation[0]['generated_text']



def switch_ne_with_Qwen(sentence,neList,targetLang):
    
    translated_sentence = sentence
    for ne in neList:
       
        if ne in cache:
            translated_sentence = translated_sentence.replace(ne,cache[ne])
        else:
            entity_name = re.sub(r'\s([.,;!?()[]{}])', r'\1', ne)
            entity_name = re.sub(r'([.,;!?()[]{}])\s', r'\1', entity_name)
        
        
            _, translated_name = query_wikidata_translation(entity_name, targetLang)

           
            if translated_name is None:
                continue
            if translated_name == "":
                continue
            
            if entity_name in translated_name:
                continue
            cache[ne] = translated_name
            translated_sentence = translated_sentence.replace(ne,translated_name)
    return translated_sentence

def extract_ne_with_QwenMax(sentence):
    client = OpenAI(
        
        api_key=api_key_QwenMax,
        base_url=api_url,
    )
    completion = client.chat.completions.create(
       
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that extracts named entities from text. Always return the named entities in a Python list format, such as ["Entity1", "Entity2"]. If no entities are found, return an empty list [].'},
            {'role': 'user', 'content': sentence}
        ]
    )
    response = completion.choices[0].message.content
   
    try:
        response = ast.literal_eval(response)
    except Exception as e:
        return None
    return response


def translate_sentence_with_QwenMT(sentence, targetLang):
    neList = extract_ne_with_QwenMax(sentence)


    if neList is None:
       switchSentence = sentence
    else:
        switchSentence = switch_ne_with_Qwen(sentence,neList,targetLang=targetLang)

    
    messages = [
        {
            "role": "user",
            "content":switchSentence
        }
    ]
    translation_options = {
        "source_lang": "English",
        "target_lang": f"{CountryDict[lang]}",
    }
    completion = client.chat.completions.create(
        
        messages=messages,
        extra_body={
            "translation_options": translation_options
        }
    )
    return completion.choices[0].message.content


def extract_sentences(field='source'):
    data = getSourceFile()
    return [entry[field] for entry in data]


def extract_lang(field='target_locale'):
    data = getSourceFile()
    return [entry[field] for entry in data]


def extract_id(field='id'):
    data = getSourceFile()
    return [entry[field] for entry in data]


def merge_entities(ner_result):
    entities = []
    current_entity = []
    entity_type = None

    for token in ner_result:
        word = token['word']
        label = token['entity']

       
        if word.startswith("##"):
            word = word[2:]
            current_entity[-1] += word
            continue

        
        if label.startswith('B-'):
           
            if current_entity:
                entities.append({'entity': ' '.join(current_entity), 'label': entity_type})
            current_entity = [word]
            entity_type = label[2:]
        elif label.startswith('I-') and entity_type:
            current_entity.append(word)

   
    if current_entity:
        entities.append({'entity': ' '.join(current_entity), 'label': entity_type})

    return entities



def write_to_txt(sentences, file_path):
    with open(file_path, 'w',encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
def main():

    translated_sentences = []
    sentences = extract_sentences()
    # id = extract_id()
    targetLang = extract_lang()[0]


    cnt = 0
    for sentence in sentences:
        try:
           
            translated = translate_sentence_with_QwenMT(sentence, targetLang)
          

            translated_sentences.append(translated)
           
            cnt+=1
        except Exception as e:
            print(f"{e}")

    # 将翻译后的句子写入txt文件
    write_to_txt(translated_sentences, output_file)
    print(f"{output_file}")

if __name__=="__main__":
     main()
    generateSubmitFile()
