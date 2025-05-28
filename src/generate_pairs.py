import sys
import time
import json
import re
import os
from openai import OpenAI
from utils.globals import DATASET_DIR

client = OpenAI()
concept = sys.argv[1]

text = {
    'categorical': {
        'example': "\n1. america: country\n2. chocolate: food\n3. apple: fruit\n4. blue: colour\n5. ",
        'pairs_text': 'exemplar:category pairs'
    },
    'causal': {
        'example': "\n1. stumble: fall\n2. storm: flood\n3. medication: cure\n4. heat: fire\n5. ",
        'pairs_text': 'cause:effect pairs'
    }
}

if concept not in text:
    raise ValueError(f"Invalid concept: {concept}")

def unique_input(data):
    unique_dicts = {}
    for entry in data:
        if entry['input'] not in unique_dicts:
            unique_dicts[entry['input']] = entry  # Store the first occurrence

    # Convert back to a list
    return list(unique_dicts.values())

def clean_data(data):
    # Remove any entries with underscores
    data = [pair for pair in data if ('_' not in pair['input']) and ('_' not in pair['output'])]
    # Remove any entries with numbers
    data = [pair for pair in data if not any(char.isdigit() for char in pair['input']) and not any(char.isdigit() for char in pair['output'])]
    # Remove any entries with more than one space
    data = [pair for pair in data if pair['input'].count(' ') <= 1 and pair['output'].count(' ') <= 1]
    return data

def generate_category_dataset(N: int, example: str, pairs_text: str):
    '''
    Generates 100 pairs of antonyms, in the form of a list of 2-tuples.
    '''
    t0 = time.time()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me {N} one word {pairs_text}."},
            {"role": "assistant", "content": f"Sure! Here are {N} one word {pairs_text} satisfying this specification: {example}"},
        ]
    )

    # Add our examples to the response
    response_text: str = example + response.choices[0].message.content
    
    # Parse the response
    pattern = r'\d+\.\s*([^:]+):\s*(.+)'
    matches = re.findall(pattern, response_text)

    # Create word pairs, by splitting on commas and colons
    word_pairs = [{'input': exemplar.lower().strip(), 'output': category.lower().strip()} for exemplar, category in matches]

    print(f"Finished in {time.time()-t0:.2f} seconds.")

    return word_pairs


TOTAL_N = 1000
GENERATE_N = 100
N_RETRIES = 1
print(f"Generating {TOTAL_N} {concept} examples...")

file_path = os.path.join(DATASET_DIR, f'{concept}_eng.json')
if os.path.exists(file_path):
    pairs = json.load(open(file_path))
    print(f"Loaded {len(pairs)} {concept} examples from {file_path}")
else:
    pairs = []

while True:
    generated_pairs = generate_category_dataset(GENERATE_N, text[concept]['example'], text[concept]['pairs_text'])

    # Add the generated examples to the list
    pairs.extend(generated_pairs)

    # Remove duplicates
    pairs = unique_input(pairs)

    # Clean the pairs
    pairs = clean_data(pairs)

    print(f"Total number of examples: {len(pairs)}")
    if len(pairs) >= TOTAL_N or N_RETRIES == 0:
        break

    N_RETRIES -= 1
    print(f"Retrying... {N_RETRIES} retries left.")

# Save the dataset
json.dump(pairs, open(file_path, 'w'), indent=4)