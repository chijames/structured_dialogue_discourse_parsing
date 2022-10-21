import os
from collections import defaultdict
import tqdm
import json
import argparse
import networkx as nx
import itertools
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num_contexts", type=int)
parser.add_argument("--slide", type=int, default=15)
parser.add_argument("--dir_path", type=str, required=True)
parser.add_argument("--link_only", action='store_true')
args = parser.parse_args()

outfile = open(os.path.join(args.dir_path, 'train.txt'), 'w')

with open(os.path.join(args.dir_path, 'relation_database.json')) as infile:
    relation_database = json.load(infile)

with open(os.path.join(args.dir_path, 'train/train.json')) as infile:
    raw_data = json.load(infile)

for data in tqdm.tqdm(raw_data):
    relations = data['relations']
    edus = data['edus']
    if len(relations) == 0: # no useful information
        continue
    parsed = set()
    subtrees = []
    connections = []
    relation_mapping = {}
    for relation in relations:
        if relation['x'] < relation['y']:
            connections.append((relation['x']+1, relation['y']+1))
            relation_mapping[connections[-1]] = relation_database[relation['type']]

    parents = {}
    for connection in connections:
        parents[connection[1]] = connection[0]
    
    # ensure connectivity
    # pick up all dangling edus
    for i in range(len(edus)):
        if i+1 not in parents:
            connections.append((0, i+1))
            relation_mapping[connections[-1]] = relation_database['Special']

    # remove multiple parents, randomly
    # also remove duplicates
    G = nx.DiGraph()
    G.add_edges_from(connections)
    G = nx.algorithms.tree.minimum_spanning_arborescence(G)

    # ensure all edus are present
    nodes = [n for n in G.nodes()]
    for i in range(len(edus)+1):
        if i not in nodes:
            print ('error 60')
            exit()

    connections = [[k,v] for k, v in G.edges()] # arborescence

    for i in range(0, len(edus), args.slide): # chunk the tree to fit in gpu
        # inclusive
        # + 1 to account for the dummy root in arborescence node representations
        min_id = i + 1
        max_id = min(i + args.num_contexts, len(edus)) - 1 + 1
        cur_connections = []
        for x, y in connections:
            if min_id <= x <= max_id and min_id <= y <= max_id:
                if x == y:
                    print ('error 74')
                    exit()
                cur_connections.append([x, y])
        
        G = nx.DiGraph()
        G.add_edges_from(cur_connections)
        G.add_nodes_from(range(min_id, max_id+1))
        subgraphs = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
        for subgraph in subgraphs:
            assert nx.algorithms.tree.recognition.is_arborescence(subgraph)
            subroot = ([n for n, d in subgraph.in_degree() if d==0])
            assert len(subroot) == 1
            subroot = subroot[0]
            cur_connections.append([0, subroot])
        G = nx.DiGraph()
        G.add_edges_from(cur_connections)
        assert nx.algorithms.tree.recognition.is_arborescence(G)
        subtrees.append(G)

        if i + args.num_contexts >= len(edus):
            break
    
    for subtree in subtrees:
        nodes = sorted([n for n in subtree.nodes()])
        assert nodes[0] == 0
        text = []
        # contract the node idxs
        mapping = {0:0} # 0 is reserved for the dummy root
        for node in nodes[1:]:
            mapping[node] = len(mapping)
            text.append('{} , {}'.format(edus[node-1]['speaker'], edus[node-1]['text'])) # node-1 to go back to edus idx
        arcs = []
        relation_types = []
        for k, v in subtree.edges():
            if (k,v) in relation_mapping:
                arcs.append([mapping[k], mapping[v], relation_mapping[(k,v)]])
            else:
                arcs.append([mapping[k], mapping[v], relation_database['Special']])

        arcs.sort(key=lambda x:x[1])
        for arc in arcs:
            if arc[0] >= arc[1]:
                print ('error 116')
        diff = args.num_contexts-len(text)
        text = text + ['dummy']*diff
        outfile.write('\n'.join(text)+'\n')
        arcs = arcs + [[1,1,1]]*diff
        outfile.write('-1 {} | {} | '.format(' '.join([str(arc[0]) for arc in arcs]), args.num_contexts-diff))
        if args.link_only:
            outfile.write('{}\n'.format(' '.join([str(0) for arc in arcs])))
        else:
            outfile.write('{}\n'.format(' '.join([str(arc[2]) for arc in arcs])))
