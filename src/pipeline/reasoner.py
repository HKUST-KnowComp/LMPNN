"""
A file maintains various reasoners
"""
from abc import abstractmethod
from collections import defaultdict

import torch
from torch import nn
from src.language.foq import (Atomic, Conjunction, Disjunction,
                              EFO1Query, Negation)

class Reasoner:
    formula: EFO1Query = None
    term_local_emb_dict = {}

    @abstractmethod
    def initialize_with_query(self, formula:EFO1Query):
        pass

    @abstractmethod
    def estimate_variable_embeddings(self):
        pass

    @abstractmethod
    def initialize_variable_embeddings(self):
        pass

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def get_ent_emb(self,
                    term_name,
                    begin_index=None,
                    end_index=None):
        if self.formula.has_term_grounded_entity_id_list(term_name):
            emb = self.nbp.get_entity_emb(
                self.formula.get_term_grounded_entity_id_list(term_name))
        elif self.term_local_emb_dict[term_name] is not None:
            emb = self.term_local_emb_dict[term_name]
        else:
            raise KeyError("Embedding does not found")
        # when it is not free variable, we consider the batch
        if begin_index is not None and end_index is not None:
            emb = emb[begin_index: end_index]
        return emb

    def get_rel_emb(self,
                    rel_name,
                    begin_index=None,
                    end_index=None,
                    inv=False):
        emb = self.nbp.get_relation_emb(
            self.formula.get_pred_grounded_relation_id_list(rel_name), inv)
        if begin_index is not None and end_index is not None:
            emb = emb[begin_index: end_index]
        return emb



    def evaluate_truth_values(self, free_var_emb_dict={}, batch_size_eval=None):
        """
        Input args:

        Return args:
        """
        def run_in_batch(batch_size):
            begin_idx = 0
            end_idx = begin_idx + batch_size
            collect = []
            while begin_idx < self.formula.num_instances:
                ret = self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    self.formula.formula,
                    begin_idx, end_idx)
                collect.append(ret)

                begin_idx = end_idx
                end_idx = begin_idx + batch_size
                end_idx = min(self.formula.num_instances, end_idx)

            return torch.cat(collect, dim=-1)

        if batch_size_eval:
            return run_in_batch(batch_size=batch_size_eval)
        else:
            return run_in_batch(batch_size=self.formula.num_instances)

    def batch_evaluate_truth_values(self,
                                    var_emb_assignment,
                                    formula,
                                    begin_index,
                                    end_index):
        """
        Recursive evaluation of the formula functions
        Input args:
            formula: the formula at this time
        Return args:
            - truth values in shape either:
                 [num candidate answers, batch size]
                 or
                 [batch size]
        """

        if isinstance(formula, Conjunction):
            return self.tnorm.conjunction(
                self.batch_evaluate_truth_values(
                    var_emb_assignment,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    var_emb_assignment,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Disjunction):
            return self.tnorm.disjunction(
                self.batch_evaluate_truth_values(
                    var_emb_assignment,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    var_emb_assignment,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Negation):
            return self.tnorm.negation(
                self.batch_evaluate_truth_values(
                    var_emb_assignment,
                    formula.formula,
                    begin_index,
                    end_index))

        elif isinstance(formula, Atomic):
            head_name = formula.head.name
            if head_name in var_emb_assignment:
                head_emb = var_emb_assignment[head_name]
            else:
                head_emb = self.get_ent_emb(head_name, begin_index, end_index)

            tail_name = formula.tail.name
            if tail_name in var_emb_assignment:
                tail_emb = var_emb_assignment[tail_name]
            else:
                tail_emb = self.get_ent_emb(tail_name, begin_index, end_index)

            pred_name = formula.name
            rel_emb = self.get_rel_emb(pred_name, begin_index, end_index)

            batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
            batch_truth_value = self.nbp.score2truth_value(batch_score)
#             batch_truth_value = batch_score  # CQD's trick for 2i, 3i
            return batch_truth_value
