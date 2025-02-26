from transformers import pipeline
pipe = pipeline("text2text-generation", model="facebook/m2m100_418M")

pipe.tokenizer.src_lang = "en"
pipe.generation_config.forced_bos_token_id = pipe.tokenizer.get_lang_id('de')

language_codes = {
    "ar": {"Arabic": "ar"},
    "de": {"German": "de"},
    "es": {"Spanish": "es"},
    "fr": {"French": "fr"},
    "it": {"Italian": "it"},
    "ja": {"Japanese": "ja"},
    "ko": {"Korean": "ko"},
    "th": {"Thai": "th"},
    "tr": {"Turkish": "tr"},
    "zh": {"Chinese": "zh"}
}

text = "What is the seventh tallest mountain in North America?"
translation = pipe(text)
print(translation[0]['generated_text'])
