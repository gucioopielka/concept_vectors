import deepl
import os
import json
from tqdm import tqdm
import time
from utils.globals import DATASET_DIR

translator = deepl.Translator(os.environ.get('DEEPL_API_KEY'))
target_langs = ['FR', 'ES']
tasks_to_translate = ['causal_eng']

for target_lang in target_langs:
    for task in tasks_to_translate:
        data = json.load(open(os.path.join(DATASET_DIR, f'{task}.json'), 'r'))
        target_file_path = os.path.join(DATASET_DIR, f'{task.replace("_eng", "")}_{target_lang.lower()}.json')

        if os.path.exists(target_file_path):
            translated_data = json.load(open(target_file_path, 'r'))
        else:
            translated_data = []
        
        if len(translated_data) == len(data):
            print(f'{task} already translated')
            continue
        else:
            print(f'Translating {task} to {target_lang}')

        for i, d in enumerate(tqdm(data)):
            if i < len(translated_data):
                continue
            while True:
                try:
                    translated_data.append({
                        'input': translator.translate_text(d['input'], source_lang='EN', target_lang=target_lang, model_type='quality_optimized').text,
                        'output': translator.translate_text(d['output'], source_lang='EN', target_lang=target_lang, model_type='quality_optimized').text
                    })
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            # Save progress every 50 iterations
            if (i + 1) % 50 == 0:
                json.dump(translated_data, open(target_file_path, 'w'), indent=4)

        # Final save
        json.dump(translated_data, open(target_file_path, 'w'), indent=4)