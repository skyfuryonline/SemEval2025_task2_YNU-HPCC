import json
import requests
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz

langList = ['tr_TR','zh_TW']
lang = "zh_TW"
modelName = "DeepSeek(zh&tr)"

myDict = {}

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
    "zh_TW":"zh-hant",
    "zh":"zh-hant"
}

with open(NerDict_jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        
        data = json.loads(line.strip())
        
        key = data.get('ne')
        if key:
            myDict[key] = data

def clean_translated_name(translated_name):
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

    results = soup.select(".mw-search-results a")  
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
    print("Merged data has been saved in jsonl.")

def getSourceFile():
    with open(jsonl_file, 'r', encoding='utf-8') as data_file:
        data = [json.loads(line.strip()) for line in data_file]
    return data

def getTranslationFile():
    with open(output_file, 'r', encoding='utf-8') as translation_file:
        translations = translation_file.readlines()
    return translations

def extract_sentences(field='source'):
    data = getSourceFile()
    return [entry[field] for entry in data]

def extract_lang(field='target_locale'):
    data = getSourceFile()
    return [entry[field] for entry in data]

def extract_id(field='id'):
    data = getSourceFile()
    return [entry[field] for entry in data]

def write_to_txt(sentences, file_path):
    with open(file_path, 'w',encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

def translate_with_DeepSeek(sentence,targetLang):
    url = "http://localhost:11434/api/generate"
    tmpNeTrans = ""
    tmpNe = ""
    for ne in myDict.keys():
        if ne in sentence:
            tmpNeTrans = myDict[ne][targetLang]
            tmpNe = ne
            break
    data = {
        "model": "deepseek-r1:70b",  
        "prompt": f"""Translate the following English text to {CountryDict[targetLang]}, using the given entity translation dictionary.
    - Dictionary:
      - "{tmpNe}" -> "{tmpNeTrans}"

    Text: '{sentence}'
    """,
        "stream": False  
    }

    response = requests.post(url, json=data)
    translated = response.json()["response"]
    translated = re.sub(r"</think>.*?</think>", "", translated, flags=re.DOTALL)
    translated = [line.strip() for line in translated.split("\n") if line.strip()]
    return translated[-1] if translated else ""
def main():
    translated_sentences = []
    sentences = extract_sentences()
    targetLang = extract_lang()[0]
    

    cnt = 0
    for sentence in sentences:
        try:
            translated = translate_with_DeepSeek(sentence, targetLang)
            translated_sentences.append(translated)
            cnt+=1
        except Exception as e:
            print(f"{e}")

    write_to_txt(translated_sentences, output_file)
    print(f"{output_file}")

if __name__=="__main__":
    main()
    generateSubmitFile()