import os
from collections import defaultdict
import pickle

import numpy as np

from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph

output_folder = "ssl-relation-prediction/data"
name_list = ["FB15k-237-betae", "FB15k-betae", "NELL-betae"]

def convert(name):
    kgidx = KGIndex.load(f"data/{name}/kgindex.json")
    train_kg = KnowledgeGraph.create(triple_files=f"data/{name}/train_kg.tsv", kgindex=kgidx)
    valid_kg = KnowledgeGraph.create(triple_files=f"data/{name}/valid_kg.tsv", kgindex=kgidx)
    test_kg = KnowledgeGraph.create(triple_files=f"data/{name}/test_kg.tsv", kgindex=kgidx)
    train_triples = train_kg.get_triples_without_betae_type_reciprocal_relation(exclusion=[])
    valid_triples = valid_kg.get_triples_without_betae_type_reciprocal_relation(exclusion=train_kg.triples)
    test_triples = test_kg.get_triples_without_betae_type_reciprocal_relation(exclusion=valid_kg.triples)
    train_triple_array = np.asarray(train_triples, dtype=np.int64)
    valid_triple_array = np.asarray(valid_triples, dtype=np.int64)
    test_triple_array  = np.asarray(test_triples, dtype=np.int64)

    os.makedirs(os.path.join(output_folder, name), exist_ok=True)
    np.save(os.path.join(output_folder, name, "train.npy"), train_triple_array)
    np.save(os.path.join(output_folder, name, "valid.npy"), valid_triple_array)
    np.save(os.path.join(output_folder, name, "test.npy"), test_triple_array)

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for lhs, rel, rhs in train_triples + valid_triples + test_triples:
        to_skip['lhs'][(rhs, rel + kgidx.num_relations // 2)].add(lhs)  # reciprocals
        to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for pos, skip in to_skip.items():
        for query, ans in skip.items():
            to_skip_final[pos][query] = sorted(list(ans))

    with open(os.path.join(output_folder, name, "to_skip.pickle"), 'wb') as out:
        pickle.dump(to_skip_final, out)
    print('Done processing!')


for name in name_list:
    print("converting " + name + " ...")
    convert(name)
