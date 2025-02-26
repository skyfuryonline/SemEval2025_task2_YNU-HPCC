import openai
openai.api_key="your api_key"
import os.path
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Tongyi

langList = ['ar_AE','de_DE','fr_FR','es_ES','it_IT','ko_KR','th_TH','tr_TR','zh_TW',"ja_JP"]
lang = "tr_TR"
api_url = 'your api_url'
api_key = 'your api_key'
myDict = {}
with open(nerDict_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        key = data.get('ne')
    if key:
        myDict[key] = data
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

def translated_with_CoT(sentence,target_lang):
    prompt = PromptTemplate(
    input_variables=["sentence", "myDict","target_lang","tl"],
    template="""
    Please translate the sentence according to the following rules and only output the translated result without any additional content:
    1. For named entities, if there is a corresponding translation in the dictionary, use the translation from the dictionary and select the appropriate translation based on the specified language ({target_lang}).
    2. If no corresponding translation is found in the dictionary, use the model's translation as a substitute.
    3. Translate the other parts normally.
    4. Please convert the translation to Turkish (tr_TR).
    Sentence: {sentence}
    Named Entity Dictionary: {myDict}
    Target Language: {tl}

    Only output the translated sentence:
    """
    )
    llm = Tongyi(
        model_name="your model",
        api_key="your api_key",
        base_url="your base_url"
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    translated_sentence = llm_chain.run({"sentence": sentence, "myDict": myDict,"target_lang":target_lang,"tl":CountryDict2[target_lang]})
    translated_lines = translated_sentence.strip().split("\n")
    last_line = translated_lines[-1] if translated_lines else ""
    processed_result = last_line.strip()
    return processed_result

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
def main():
    translated_sentences = []
    sentences = extract_sentences()
    targetLang = extract_lang()[0]
    
    cnt = 0
    for sentence in sentences:
        try:
            translated = translated_with_CoT(sentence, targetLang)
            translated_sentences.append(translated)
            cnt+=1
        except Exception as e:
            print(f"{e}")

    write_to_txt(translated_sentences, output_file)
    print(f"{output_file}")

if __name__=="__main__":
        main()
        generateSubmitFile()
