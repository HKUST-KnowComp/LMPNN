from collections import defaultdict
from random import random, sample

import networkx as nx
from networkx.algorithms.components import connected_components

from .fof import *

def sample_efo_dnf_query(p, q, r,
                         per,
                         l, k, pn):
    """
    args:
        term level:
            p: number of existential variables
            q: number of free variables
            r: number of literals
        atom level:
            per: probability of sampling the edge from er graph
                 each edge is an atom with relation to be filled
        conjuction level
            l: maximum number of conjunctive queries
            k: number of atoms per conjunctive queries
            pn: number of prossibility of the negation,
                since we don't have any negation before,
                we won't suffer from the conflict
    """
    exist_vars = {f"ev:{i}": Variable(f"ev:{i}", Variable.EXIST)
                  for i in range(p)}

    free_vars = {f"fv:{i}": Variable(f"fv:{i}", Variable.FREE)
                 for i in range(q)}

    literals = {f"ltr:{i}": Literal(-1) for i in range(r)}

    terms = {}
    terms.update(exist_vars)
    terms.update(free_vars)
    terms.update(literals)

    print(terms)

    predicates = {}
    pred2termpair = {}
    term2pred = defaultdict(list)
    for k1 in terms:
        for k2 in terms:
            if k1 == k2:
                continue

            if random() > per:
                continue

            pid = f"pred:{len(pred2termpair)}"

            termpair = (k1, k2)
            pred2termpair[pid] = termpair

            _pred = BinaryPredicate(
                relation_id=-1, head=terms[k1], tail=terms[k2])
            predicates[pid] = _pred

            term2pred[k1].append(pid)
            term2pred[k2].append(pid)

    pkeys = list(predicates.keys())
    conjunctions = {}
    conj2terms = defaultdict(set)
    for i_conj in range(l):
        conj_key = f"conj:{i_conj}"
        conj_atoms = []
        conj_pred_keys = sample(pkeys, k=k)
        for pred_key in conj_pred_keys:
            if random() < pn:
                atom = Negation(predicates[pred_key])
            else:
                atom = predicates[pred_key]
            conj_atoms.append(atom)
            for t in pred2termpair[pred_key]:
                if 'v' in t:
                    conj2terms[conj_key].add(t)
        conjunctions[conj_key] = Conjunction(atoms=conj_atoms)

    print(conj2terms)

    conj_graph = nx.Graph()
    for conj_k1 in conj2terms:
        for conj_k2 in conj2terms:
            if conj_k1 == conj_k2:
                continue
            if len(conj2terms[conj_k1].intersection(conj2terms[conj_k2])) > 0:
                conj_graph.add_edge(conj_k1, conj_k2)

    largest_conn_conj_comp = sorted(
        connected_components(conj_graph), key=len, reverse=True)[0]
    dnf = DNF(conjunctions=largest_conn_conj_comp)
    DNFFormula()
