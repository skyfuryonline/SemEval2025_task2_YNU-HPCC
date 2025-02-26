from transformers import pipeline

pipe = pipeline("token-classification", model="dslim/bert-large-NER")

# text = "What type of place is the Po Lin Monastery?"

# text = "Can you describe the color scheme used on the coat of arms of Egypt?"
text = "What can readers expect to learn from The Prize: The Epic Quest for Oil, Money, and Power?"

ner_result = pipe(text)
# for entity in ner_result:
#     print(f"Entity: {entity['word']} | Label: {entity['entity']} | Score: {entity['score']}")
# print("---------------------------------------------------------------------------------")



def merge_entities(ner_result):
    entities = []
    current_entity = []
    entity_type = None

    for token in ner_result:
        word = token['word']
        label = token['entity']

        
        if word.startswith("##"):
            word = word[2:]
            current_entity[-1] += word  #
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

merged_entities = merge_entities(ner_result)

for entity in merged_entities:
    print(f"Entity: {entity['entity']} | Label: {entity['label']}")

'''
text = "Hugging Face is creating a tool that democratizes AI. The company was founded by Clément Delangue."
Entity: Hugging Face | Label: ORG
Entity: AI | Label: MISC
Entity: Clément Delangue | Label: PER

What type of place is the Po Lin Monastery?
Entity: Po Lin Monastery | Label: LOC
'''
