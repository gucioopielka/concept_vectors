import pandas as pd
import os, json
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--conceptnet_path', type=str, default='data/analogies/conceptnet.csv')
    argparser.add_argument('--save_path', type=str, default='data/ICL/conceptnet')
    args = argparser.parse_args()

    conceptnet_path = args.conceptnet_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    conceptnet_df = pd.read_csv(conceptnet_path, sep=';')

    for rel in conceptnet_df['rel'].unique():
        rel_df = conceptnet_df[conceptnet_df['rel'] == rel]
        rel_dict = []
        for row in rel_df.iterrows():
            rel_dict.append({'input': row[1]['start'], 'output': row[1]['end']})
        rel_name = rel.lower().replace(' ', '_')
        json.dump(rel_dict, open(f'{save_path}/{rel_name}.json', 'w'), indent=4, separators=(',', ': '))



