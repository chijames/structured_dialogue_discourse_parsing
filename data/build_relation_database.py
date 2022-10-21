import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, required=True)
args = parser.parse_args()

mapping = {}

with open(os.path.join(args.dir_path, 'train/train.json')) as infile:
    data = json.load(infile)

for d in data:
    for relation in d['relations']:
        if relation['type'] not in mapping:
            mapping[relation['type']] = len(mapping)

mapping['Special'] = len(mapping)

with open(os.path.join(args.dir_path,'relation_database.json'), 'w') as outfile:
    json.dump(mapping, outfile)
