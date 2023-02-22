import os
import yaml

from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph

output_folder = "kge/data"
name_list = ["FB15k-237-betae", "FB15k-betae", "NELL-betae"]

def convert(name):
    kgidx = KGIndex.load(f"data/{name}/kgindex.json")
    train_kg = KnowledgeGraph.create(triple_files=f"data/{name}/train_kg.tsv", kgindex=kgidx)
    valid_kg = KnowledgeGraph.create(triple_files=f"data/{name}/valid_kg.tsv", kgindex=kgidx)
    test_kg = KnowledgeGraph.create(triple_files=f"data/{name}/test_kg.tsv", kgindex=kgidx)
    config = {"dataset": {
        "name": name,
        "num_entities": kgidx.num_entities,
        "num_relations": kgidx.num_relations // 2,

        "files.train.filename": "train.tsv",
        "files.train.size": train_kg.num_triples // 2,
        "files.train.type": "triples",

        "files.valid_without_unseen.filename": "valid.tsv",
        "files.valid_without_unseen.size": valid_kg.num_triples // 2,
        "files.valid_without_unseen.type": "triples",

        "files.valid.filename": "valid.tsv",
        "files.valid.size": valid_kg.num_triples // 2,
        "files.valid.type": "triples",

        "files.test_without_unseen.filename": "test.tsv",
        "files.test_without_unseen.size": test_kg.num_triples // 2,
        "files.test_without_unseen.type": "triples",

        "files.test.filename": "test.tsv",
        "files.test.size": test_kg.num_triples // 2,
        "files.test.type": "triples",
    }}

    os.makedirs(os.path.join(output_folder, name), exist_ok=True)
    kgidx.dump_id2name(entity_id_file=os.path.join(output_folder, name, "entity_ids.del"),
                    relation_id_file=os.path.join(output_folder, name, "relation_ids.del"),
                    rel_dup=False)
    train_kg.dump_without_betae_type_reciprocal_relation(os.path.join(output_folder, name, "train.tsv"))
    valid_kg.dump_without_betae_type_reciprocal_relation(os.path.join(output_folder, name, "valid.tsv"))
    test_kg.dump_without_betae_type_reciprocal_relation( os.path.join(output_folder, name, "test.tsv"))
    f = open(os.path.join(output_folder, name, "valid.del"), "wt")
    f.close()
    with open(os.path.join(output_folder, name, "dataset.yaml"), "wt") as f:
        f.write(yaml.dump(config))

for name in name_list:
    print("converting " + name + " ...")
    convert(name)
