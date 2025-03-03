import time
import json
import re
from openai import OpenAI
client = OpenAI()

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

def generate_category_dataset(N: int):
    '''
    Generates 100 pairs of antonyms, in the form of a list of 2-tuples.
    '''
    t0 = time.time()

    # Define a few examples (for our dataset, and for our prompt)
    example_antonyms = "\n1. america: country\n2. chocolate: food\n3. apple: fruit\n4. blue: colour\n5. "

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give me {N} one word exemplar:category pairs."},
            {"role": "assistant", "content": f"Sure! Here are {N} exemplar:category pairs satisfying this specification: {example_antonyms}"},
        ]
    )

    # Add our examples to the response
    response_text: str = example_antonyms + response.choices[0].message.content
    
    # Parse the response
    pattern = r'\d+\.\s*([^:]+):\s*(.+)'
    matches = re.findall(pattern, response_text)

    # Create word pairs, by splitting on commas and colons
    word_pairs = [{'input': exemplar.lower().strip(), 'output': category.lower().strip()} for exemplar, category in matches]

    print(f"Finished in {time.time()-t0:.2f} seconds.")

    return word_pairs


TOTAL_N = 1000
GENERATE_N = 100
N_RETRIES = 5
print(f"Generating {TOTAL_N} categorical examples...")
category_pairs = []
while True:
    generated_cats = generate_category_dataset(GENERATE_N)

    # Add the generated examples to the list
    category_pairs.extend(generated_cats)

    # Remove duplicates
    category_pairs = unique_input(category_pairs)

    # Clean the category pairs
    category_pairs = clean_data(category_pairs)

    print(f"Total number of examples: {len(category_pairs)}")
    if len(category_pairs) >= TOTAL_N or N_RETRIES == 0:
        break

    N_RETRIES -= 1
    print(f"Retrying... {N_RETRIES} retries left.")

# Save the dataset
json.dump(category_pairs, open('data/datasets/categorical_eng.json', 'w'), indent=4)