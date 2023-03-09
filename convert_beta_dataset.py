import os
import os.path as osp
import json
import pickle
from typing import Dict

from tqdm import tqdm

from src.language import fof
from src.language.grammar import parse_lstr_to_lformula
from src.structure.knowledge_graph_index import KGIndex
from src.structure.knowledge_graph import KnowledgeGraph

beta_types_key_list = [
    ('e', ('r',)),
    ('e', ('r', 'r')),
    ('e', ('r', 'r', 'r')),
    (('e', ('r',)), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
    ((('e', ('r',)), ('e', ('r',))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r', 'n'))),
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
    (('e', ('r', 'r')), ('e', ('r', 'n'))),
    (('e', ('r', 'r', 'n')), ('e', ('r',))),
    (('e', ('r',)), ('e', ('r',)), ('u',)),
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)),
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')),
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
    ]

labeled_beta_types_list = [
    ('s1', ('r1',)),
    ('s1', ('r1', 'r2')),
    ('s1', ('r1', 'r2', 'r3')),
    (('s1', ('r1',)), ('s2', ('r2',))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('s3', ('r3',))),
    ((('s1', ('r1',)), ('s2', ('r2',))), ('r3',)),
    (('s1', ('r1', 'r2')), ('s2', ('r3',))),
    (('s1', ('r1',)), ('s2', ('r2', 'n'))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('s3', ('r3', 'n'))),
    ((('s1', ('r1',)), ('s2', ('r2', 'n'))), ('r3',)),
    (('s1', ('r1', 'r2')), ('s2', ('r3', 'n'))),
    (('s1', ('r1', 'r2', 'n')), ('s2', ('r3',))),
    (('s1', ('r1',)), ('s2', ('r2',)), ('u',)),
    ((('s1', ('r1',)), ('s2', ('r2',)), ('u',)), ('r3',)),
    ((('s1', ('r1', 'n')), ('s2', ('r2', 'n'))), ('n',)),
    ((('s1', ('r1', 'n')), ('s2', ('r2', 'n'))), ('n', 'r3')),
    ((('s1', ('r1',)), ('s2', ('r2',)), ('u',)), ('r3',))
    ]

beta_lstr_list = [
    "r1(s1,f)",  # 1p
    "r1(s1,e1)&r2(e1,f)",  # 2p
    "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "r1(s1,f)&r2(s2,f)",  # 2i
    "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "r1(s1,e1)&r2(s2,e1)&r3(e1,f)", ## ip
    "r1(s1,e1)&r2(e1,f)&r3(s2,f)", # pi
    "r1(s1,f)&!r2(s2,f)", #2in
    "r1(s1,f)&r2(s2,f)&!r3(s3,f)", # 3in
    "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)", # inp
    "r1(s1,e1)&r2(e1,f)&!r3(s2,f)", # pin
    "r1(s1,e1)&!r2(e1,f)&r3(s2,f)", # pni
    "r1(s1,f)|r2(s2,f)", # 2u
    "(r1(s1,e1)|r2(s2,e1))&r3(e1,f)", # up
    "!(!r1(s1,f)&!r2(s2,f))", # 2u-dm
    "!(!r1(s1,e1)&!r2(s2,e1))&r3(e1,f)",# up-dm
    "(r1(s1,e1)&r3(e1,f))|(r2(s2,e1)&r3(e1,f))" # up-dnf
]

beta_names = [
    '1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u', 'up', '2u-dm', 'up-dm', 'up-dnf'
]

beta_lstr2name = {}
for s, n in zip(beta_lstr_list, beta_names):
    print(s, n)
    beta_lstr2name[
        parse_lstr_to_lformula(s).lstr()
    ] = n


def beta_type_to_ldict():
    pass

def align_entities_relations(labeled_beta_type, beta_sample) -> Dict:
    d = {}
    def _align(labeled_beta_type, beta_sample):
        for sub_type, sub_sample in zip(labeled_beta_type, beta_sample):
            if not isinstance(sub_type, str):
                _align(sub_type, sub_sample)
            else:
                if sub_type[0] in 'sr':
                    d[sub_type] = sub_sample
    _align(labeled_beta_type, beta_sample)
    return d

def convert_beta_folder(beta_folder, output_folder):
    """
    Convert the folder of beta dataset into the output data
    the structure of the beta folder
        indices
        - ent2id.pkl
        - id2ent.pkl
        - id2rel.pkl
        - rel2id.pkl

        knowledge graphs and queries
        {test/valid/train}.txt

        train_queries
        train-queries.pkl
        train-answers.pkl

        evaluation queries
        {test/valid}-queries.pkl
        {test/valid}-easy-answers.pkl
        {test/valid}-hard-answers.pkl

    the structure of output folder
        kgindex.json: aggregrates the indices files into one
                      used by the KGIndex.load()
        {train/test/valid}.tsv: triple information in the tsv
                                used by KnowledgeGraph.create()
        {train/test/valid}_qaa.json: dictionary
            key: lstr of each type
            value: List of triples (mapping dict, easy_answer, hard_answer)
            for train, the hard answer is empty list

    """

    # build knowledge graph indices

    print("converting KGIndex")

    os.makedirs(output_folder, exist_ok=True)

    kgidx = KGIndex()
    with open(osp.join(beta_folder, 'ent2id.pkl'), 'rb') as f:
        entity_to_id = pickle.load(f)
    eids = []
    for name, eid in entity_to_id.items():
        kgidx.register_entity(name, eid)
        eids.append(eid)
    assert max(eids) - min(eids) + 1 == len(kgidx.map_entity_name_to_id)

    with open(osp.join(beta_folder, 'rel2id.pkl'), 'rb') as f:
        relation_to_id = pickle.load(f)

    rids = []
    for name, rid in relation_to_id.items():
        kgidx.register_relation(name, rid)
        rids.append(rid)
    assert max(rids) - min(rids) + 1 == len(kgidx.map_relation_name_to_id)

    print("dump converted KGIndex")
    kgidx.dump(osp.join(output_folder, 'kgindex.json'))
    kgidx = KGIndex.load(osp.join(output_folder, 'kgindex.json'))

    # train knowledge graphs
    print("converting train KnowledgeGraph")
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(beta_folder, 'train.txt'),
        kgindex=kgidx)
    print("dump converted train KnowledgeGraph")
    train_kg.dump(osp.join(output_folder, 'train_kg.tsv'))

    print("converting valid KnowledgeGraph")
    valid_kg = KnowledgeGraph.create(
        triple_files=[osp.join(beta_folder, 'train.txt'),
                      osp.join(beta_folder, 'valid.txt')],
        kgindex=kgidx)
    print("dump converted valid KnowledgeGraph")
    valid_kg.dump(osp.join(output_folder, 'valid_kg.tsv'))

    print("converting test KnowledgeGraph")
    test_kg = KnowledgeGraph.create(
        triple_files=[osp.join(beta_folder, 'train.txt'),
                      osp.join(beta_folder, 'valid.txt'),
                      osp.join(beta_folder, 'test.txt')],
        kgindex=kgidx)
    print("dump converted test KnowledgeGraph")
    test_kg.dump(osp.join(output_folder, 'test_kg.tsv'))

    # knowledge graph queries

    # train queries
    with open(osp.join(beta_folder, "train-queries.pkl"), 'rb') as f:
        train_queries = pickle.load(f)

    with open(osp.join(beta_folder, "train-answers.pkl"), 'rb') as f:
        train_answers = pickle.load(f)

    lstr_xy_dict = {}
    for key, labeled_type, lstr in zip(
        beta_types_key_list, labeled_beta_types_list, beta_lstr_list):
        samples = list(train_queries[key])
        if len(samples) == 0:
            print(key, lstr, "not found in the dataset")
            continue

        lformula = parse_lstr_to_lformula(lstr)
        folf = fof.FirstOrderFormula(lformula)
        print(folf.formula.lstr())

        lstr_xy_dict[lstr] = []
        for sample in tqdm(samples, desc='train query answer processing'):
            d = align_entities_relations(labeled_type, sample)
            answer = {'f': list(train_answers[sample])}
            folf.append_relation_and_symbols(d)
            lstr_xy_dict[lstr].append(
                (d, answer, [])
            )
    with open(osp.join(output_folder, 'train-qaa.json'), 'wt') as f:
        print([k for k in lstr_xy_dict])
        json.dump(lstr_xy_dict, f)

    # valid queries
    with open(osp.join(beta_folder, "valid-queries.pkl"), 'rb') as f:
        valid_queries = pickle.load(f)

    with open(osp.join(beta_folder, "valid-easy-answers.pkl"), 'rb') as f:
        valid_easy_answers = pickle.load(f)

    with open(osp.join(beta_folder, "valid-hard-answers.pkl"), 'rb') as f:
        valid_hard_answers = pickle.load(f)

    lstr_xy_dict = {}
    for key, labeled_type, lstr in zip(
        beta_types_key_list, labeled_beta_types_list, beta_lstr_list):
        samples = list(valid_queries[key])

        lformula = parse_lstr_to_lformula(lstr)
        folf = fof.FirstOrderFormula(lformula)
        print(folf.formula.lstr())

        lstr_xy_dict[lstr] = []
        for sample in tqdm(samples, desc="valid query answer processing"):
            d = align_entities_relations(labeled_type, sample)
            easy_answer = {'f': list(valid_easy_answers[sample])}
            hard_answer = {'f': list(valid_hard_answers[sample])}
            folf.append_relation_and_symbols(d)
            lstr_xy_dict[lstr].append(
                (d, easy_answer, hard_answer)
            )

    with open(osp.join(output_folder, 'valid-qaa.json'), 'wt') as f:
        json.dump(lstr_xy_dict, f)


    # test queries
    with open(osp.join(beta_folder, "test-queries.pkl"), 'rb') as f:
        test_queries = pickle.load(f)

    with open(osp.join(beta_folder, "test-easy-answers.pkl"), 'rb') as f:
        test_easy_answers = pickle.load(f)

    with open(osp.join(beta_folder, "test-hard-answers.pkl"), 'rb') as f:
        test_hard_answers = pickle.load(f)

    lstr_xy_dict = {}
    for key, labeled_type, lstr in zip(
        beta_types_key_list, labeled_beta_types_list, beta_lstr_list):
        samples = list(test_queries[key])

        lformula = parse_lstr_to_lformula(lstr)
        folf = fof.FirstOrderFormula(lformula)
        print(folf.formula.lstr())

        lstr_xy_dict[lstr] = []
        for sample in tqdm(samples, desc='test query answer processing'):
            d = align_entities_relations(labeled_type, sample)
            # possible answers into the key-value form
            easy_answer = {'f': list(test_easy_answers[sample])}
            hard_answer = {'f': list(test_hard_answers[sample])}
            folf.append_relation_and_symbols(d)
            lstr_xy_dict[lstr].append(
                (d, easy_answer, hard_answer)
            )

    with open(osp.join(output_folder, 'test-qaa.json'), 'wt') as f:
        json.dump(lstr_xy_dict, f)


if __name__ == "__main__":
    beta_folder = "./data/betae-dataset/{}"
    output_folder = "./data/{}"

    for dataset in [
        "FB15k-237-betae",
        "FB15k-betae",
        "NELL-betae"]:
        print(dataset)
        convert_beta_folder(beta_folder.format(dataset),
                            output_folder.format(dataset))
