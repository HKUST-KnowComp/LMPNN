"""
A base class for existential first order formulas
It supports the verification and query answering tasks given the formula
For the verification, there is no free_vars
For the query answering, there should be at least one free_vars
Several assumptions about the formula
- In DNF
- Only with existential quantifier
The query is generated by the following steps:

>>>> Construct the conjunctive query skeleton:
    1. Define the number of terms, including
        - p existential vars
        - q free vars
        - r literals
        There are p + q + r terms
    2. Construct the edges between terms by random sampling, and then construct
        a connective graph, the edge is constructed by the certain ratio $P_e$
    3. Randomly corrupt the edges with negation $P_n$
<<<< In this way, the conjunctive query skeleton is constructed

>>>> Construct the DNF formula
    The key of DNF construction is that there must be at least one variable be
    shared in two conjunctive query
    1. determin the variables to be shared (more than one)
    2. determin the objects to be shared (not necessary more than one)
<<<<

>>>> Sampling accross the graph
    1. For question answering task
        1. sample the full graph,
            so that the predicates and objects are determined
        2. search over partial graph and full graph,
            and find out the answer tuples
    2. Instantiation of the query graph
<<<<
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import json
from typing import Dict, List
from random import sample

import torch

from src.language.tnorm import Tnorm
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
# from src.utils.data import RaggexxdBatch


def check_ldict(ldict):
    """
    Ldict is a nested dict that stores the GROUNDED information
    """
    assert 'op' in ldict
    op = ldict['op']
    assert 'args' in ldict
    args = ldict['args']

    if op == Term.op:
        assert 'name' in args
        assert 'state' in args
        assert 'entity_id_list' in args
    if op == BinaryPredicate.op:
        assert 'name' in args
        assert 'relation_id_list' in args
        check_ldict(args['term1'])
        check_ldict(args['term2'])
    if op == Negation.op:
        assert 'formula' in args
        check_ldict(args['formula'])
    if op == Conjunction.op or op == Disjunction.op:
        assert 'formulas' in args
        for f in args['formulas']:
            check_ldict(f)


def get_ldict(op, **args):
    ans = {'op': op, 'args': args}
    check_ldict(ans)
    return ans


class Lobject:
    op = "default"

    @abstractmethod
    def to_ldict(self) -> Dict:
        pass

    @abstractmethod
    def lstr(self) -> str:
        pass

    def __repr__(self):
        check_ldict(self.to_ldict())
        return json.dumps(self.to_ldict(), indent=1)

    @abstractmethod
    def get_predicates(self) -> Dict[str, 'BinaryPredicate']:
        pass


class Term(Lobject):
    EXISTENTIAL = 1
    FREE = 2
    UNIVERSAL = 3
    SYMBOL = 4
    # GROUNDED = 5

    op = "term"

    def __init__(self, state, name):
        self.state = state
        self.name = name
        self.parent_predicate = None
        self.entity_id_list = []

    @classmethod
    def parse(cls, ldict):
        op = ldict['op']
        assert op == cls.op
        args = ldict['args']
        name = args['name']
        state = args['state']
        object = cls(name=name, state=state)
        object.entity_id_list = args['entity_id_list']
        return object

    def to_ldict(self):
        ldict = {'op': self.op,
                 'args': {
                     'state': self.state,
                     'name': self.name,
                     'entity_id_list': self.entity_id_list}}
        return ldict

    def lstr(self) -> str:
        return self.name

    @property
    def is_free(self):
        return self.state == self.FREE

    @property
    def is_existential(self):
        return self.state == self.EXISTENTIAL

    @property
    def is_universal(self):
        return self.state == self.UNIVERSAL

    @property
    def is_symbol(self):
        return self.state == self.SYMBOL

    @property
    def is_grounded(self):
        return self.is_symbol or (self.state == Term.GROUNDED)


class Formula(Lobject):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parse(ldict):
        op = ldict['op']
        if op == BinaryPredicate.op:
            return BinaryPredicate.parse(ldict)
        elif op == Negation.op:
            return Negation.parse(ldict)
        elif op == Conjunction.op:
            return Conjunction.parse(ldict)
        elif op == Disjunction.op:
            return Disjunction.parse(ldict)
        else:
            raise NotImplementedError("Unsupported Operator")

    @property
    def num_predicates(self):
        pass


class BinaryPredicate(Formula):
    op = 'pred'

    def __init__(self,
                 name: str,
                 head: Term,
                 tail: Term) -> None:
        self.name = name
        self.head = head
        self.tail = tail
        self.relation_id_list = []
        self.skolem_negation = False

    @classmethod
    def parse(cls, ldict):
        op = ldict['op']
        assert op == cls.op
        args = ldict['args']

        name = args['name']
        head = Term.parse(args['head'])
        tail = Term.parse(args['tail'])
        object = cls(name=name, head=head, tail=tail)
        object.relation_id_list = args['relation_id_list']
        head.parent_predicate = object
        tail.parent_predicate = object
        return object

    def to_ldict(self):
        obj = {
            'op': self.op,
            'args': {
                'name': self.name,
                'relation_id_list': self.relation_id_list,
                'head': self.head.to_ldict(),
                'tail': self.tail.to_ldict()
            }
        }
        return obj

    def lstr(self):
        lstr = f"{self.name}({self.head.name},{self.tail.name})"
        return lstr

    def get_predicates(self) -> Dict[str, 'BinaryPredicate']:
        ans = {self.name: self}
        return ans

    def get_terms(self):
        return [self.head, self.tail]

    @property
    def num_predicates(self):
        return 1


class Connective(Formula):
    pass


class Negation(Connective):
    op = 'neg'

    def __init__(self, formula: Formula) -> None:
        self.formula = formula

    @classmethod
    def parse(cls, ldict):
        op = ldict['op']
        assert op == cls.op
        args = ldict['args']
        formula = Formula.parse(args['formula'])
        if formula.op == 'pred':
            formula.skolem_negation = True
        return cls(formula)

    def to_ldict(self):
        obj = {
            'op': self.op,
            'args': {'formula': self.formula.to_ldict()}
        }
        return obj

    def lstr(self) -> str:
        lstr = f"!({self.formula.lstr()})"
        return lstr

    def get_predicates(self) -> Dict[str, 'BinaryPredicate']:
        ans = {}
        ans.update(self.formula.get_predicates())
        return ans

    @property
    def num_predicates(self):
        return self.formula.num_predicates


class Conjunction(Connective):
    op = 'conj'

    def __init__(self, formulas: List[Formula]) -> None:
        self.formulas = formulas

    @classmethod
    def parse(cls, ldict):
        op = ldict['op']
        assert op == cls.op
        args = ldict['args']
        formula_dict_list = args['formulas']
        formulas = [Formula.parse(formula_dict)
                    for formula_dict in formula_dict_list]
        return cls(formulas)

    def to_ldict(self):
        obj = {
            'op': self.op,
            'args': {'formulas': [f.to_ldict() for f in self.formulas]}
        }
        return obj

    def lstr(self):
        lstr = "&".join(f"({f.lstr()})" for f in self.formulas)
        return lstr

    def get_predicates(self) -> Dict[str, 'BinaryPredicate']:
        ans = {}
        for f in self.formulas:
            ans.update(f.get_predicates())
        return ans

    @property
    def num_predicates(self):
        return sum([formula.num_predicates for formula in self.formulas])


class Disjunction(Connective):
    op = 'disj'

    def __init__(self, formulas: List[Formula]) -> None:
        self.formulas = formulas

    @classmethod
    def parse(cls, ldict):
        op = ldict['op']
        assert op == cls.op
        args = ldict['args']
        formula_dict_list = args['formulas']
        formulas = [Formula.parse(formula_dict)
                    for formula_dict in formula_dict_list]
        return cls(formulas)

    def to_ldict(self):
        obj = {
            'op': self.op,
            'args': {'formulas': [f.to_ldict() for f in self.formulas]}
        }
        return obj

    def lstr(self):
        lstr = "|".join(f"({f.lstr()})" for f in self.formulas)
        return lstr

    def get_predicates(self) -> Dict[str, 'BinaryPredicate']:
        ans = {}
        for f in self.formulas:
            ans.update(f.get_predicates())
        return ans

    @property
    def num_predicates(self):
        return sum([formula.num_predicates for formula in self.formulas])


class FirstOrderFormula:
    """
    The first order formula
    it also includes information about the quantifiers

    self.formula is parsed from the formula and provide the operator tree for
        evaluation
    self.predicate_dict stores each predicates by its name, which are edges
    self.symbol_dict stores each symbol by its name
    self.variable_dict stores each variable by its name

    self.easy_answer_list list for easy answers
    self.hard_answer_list list for hard answers
    self.noisy_answer_list list for noisy answers

    each answer is a dict whose keys are the variable and values are the list of possible answers
    """

    def __init__(self,
                 formula: Formula) -> None:
        self.formula: Formula = formula
        self.easy_answer_list = []
        self.hard_answer_list = []
        self.noisy_answer_list = []
        self.grounding_dict_list = []

        # update internal storage
        self.predicate_dict: Dict[str, BinaryPredicate] = {}
        self.pred_grounded_relation_id_dict: Dict[str, List] = {}

        self.term_dict: Dict[str, Term] = {}
        self.term_grounded_entity_id_dict: Dict[str, List] = {}

        self.term_name2predicate_name_dict: Dict[str, str] = defaultdict(list)
        # run initialization
        self._init_query()

    def _init_query(self):
        self.predicate_dict = self.formula.get_predicates()
        self.pred_grounded_relation_id_dict = {
            name: predicate.relation_id_list
            for name, predicate in self.predicate_dict.items()
        }

        self.term_dict = {}
        for _, pred in self.predicate_dict.items():
            for t in pred.get_terms():
                self.term_dict[t.name] = t

        # self.term_local_embedding_dict = {name: None
                #   for name in self.term_dict}
        self.term_grounded_entity_id_dict = {name: term.entity_id_list
                                             for name, term in self.term_dict.items()}

        for pred_name, predicate in self.predicate_dict.items():
            head, tail = predicate.get_terms()
            self.term_name2predicate_name_dict[head.name].append(pred_name)
            self.term_name2predicate_name_dict[tail.name].append(pred_name)


    def append_relation_and_symbols(self, append_dict):
        for k, v in append_dict.items():
            if k in self.term_dict:
                self.term_grounded_entity_id_dict[k].append(v)
            else:
                self.pred_grounded_relation_id_dict[k].append(v)

    def append_qa_instances(self,
                            append_dict,
                            easy_answers=[],
                            hard_answers=[],
                            noisy_answer=[]):
        self.append_relation_and_symbols(append_dict)
        self.easy_answer_list.append(easy_answers)
        self.hard_answer_list.append(hard_answers)
        self.noisy_answer_list.append(noisy_answer)

    def has_term_grounded_entity_id_list(self, key):
        return len(self.term_grounded_entity_id_dict[key]) > 0

    def get_term_grounded_entity_id_list(self, key):
        return self.term_grounded_entity_id_dict[key]

    def has_pred_grounded_relation_id_list(self, key):
        return len(self.pred_grounded_relation_id_dict[key]) > 0

    def get_pred_grounded_relation_id_list(self, key):
        return self.pred_grounded_relation_id_dict[key]

    @property
    def free_variable_dict(self):
        return {k: v
                for k, v in self.term_dict.items()
                if v.state == Term.FREE}

    @property
    def universal_variable_dict(self):
        return {k: v
                for k, v in self.term_dict.items()
                if v.state == Term.UNIVERSAL}

    @property
    def existential_variable_dict(self):
        return {k: v
                for k, v in self.term_dict.items()
                if v.state == Term.EXISTENTIAL}

    @property
    def symbol_dict(self):
        return {k: v
                for k, v in self.term_dict.items()
                if v.state == Term.SYMBOL}

    @property
    def is_sentence(self):
        """
        Determine the state of the formula
        A formula is sentence when all variables are quantified
        """
        return len({k: v for k, v in self.term_dict.items()
                    if v.state == Term.FREE}) == 0

    @property
    def lstr(self):
        return self.formula.lstr()

    @property
    def num_instances(self):
        num_instances = len(self.easy_answer_list)
        assert num_instances == len(self.hard_answer_list)
        for k in self.symbol_dict:
            assert num_instances == len(
                self.get_term_grounded_entity_id_list(k))

        for k in self.predicate_dict:
            assert num_instances == len(
                self.get_pred_grounded_relation_id_list(k))

        return len(self.easy_answer_list)

    @property
    def num_predicates(self):
        return self.formula.num_predicates

    @property
    def quantifier_rank(self):
        return len(self.existential_variable_dict) \
               + len(self.universal_variable_dict) \
               + len(self.free_variable_dict)

    def get_all_gounded_ids(self):
        entity_ids = []
        for term_name in self.term_grounded_entity_id_dict:
            entity_ids += self.term_grounded_entity_id_dict[term_name]
        relation_ids = []
        for pred_name in self.pred_grounded_relation_id_dict:
            relation_ids += self.pred_grounded_relation_id_dict[pred_name]
        return entity_ids, relation_ids