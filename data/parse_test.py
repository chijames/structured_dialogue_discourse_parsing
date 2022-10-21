import os
from collections import defaultdict
import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_contexts", type=int)
parser.add_argument("--dir_path", type=str, required=True)
parser.add_argument("--mode", type=str, default='test')
parser.add_argument("--link_only", action='store_true')
args = parser.parse_args()

outfile = open(os.path.join(args.dir_path, '{}.txt'.format(args.mode)), 'w')

with open(os.path.join(args.dir_path, 'relation_database.json')) as infile:
    relation_database = json.load(infile)
with open(os.path.join(args.dir_path, '{}/{}.json'.format(args.mode, args.mode))) as infile:
    raw_data = json.load(infile)

all_links = []
for data in raw_data:
    edges = []
    relations = data['relations']
    if len(relations) == 0:
        continue
    parents = {}
    for relation in relations:
        x, y = relation['x'] + 1, relation['y'] + 1
        if x >= args.num_contexts or y >= args.num_contexts: # skip too long dev instances
            continue
        parents[y] = x
        if args.link_only:
            edges.append((x, y, 0))
        else:
            edges.append((x, y, relation_database[relation['type']]))

    for i in range(len(data['edus'])):
        if i+1 not in parents:
            if args.link_only:
                edges.append((0, i+1, 0))
            else:
                edges.append((0, i+1, relation_database['Special']))
    #assert (0, 1) in edges # caution, this may not be true is test data is changed
    all_links.append(list(set(edges))) # remove duplicates
    edus = data['edus'][:args.num_contexts] # skip too long dev instances
    res = []
    for edu in edus:
        res.append(edu['speaker'] + ' , ' + edu['text'])
    diff = args.num_contexts-len(res)
    res = res + ['dummy']*diff
    outfile.write('\n'.join(res)+'\n')
    outfile.write('-1 | {} | \n'.format(args.num_contexts-diff)) # -1 is just an identifier used in dataloader

with open(os.path.join(args.dir_path, '{}_links.json'.format(args.mode)), 'w') as outfile:
    json.dump(all_links, outfile)

print ('total golden pairs:', sum([len(link) for link in all_links]))
