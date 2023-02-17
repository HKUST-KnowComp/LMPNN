"""
A file maintains various reasoners
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from curses import termname
from imp import is_frozen
import math
from typing import Dict, List
from random import sample

import torch
from torch import nn
from src.language.fof import (BinaryPredicate, Conjunction, Disjunction,
                              FirstOrderFormula, Negation, Term)
from src.language.tnorm import Tnorm
from src.structure.neural_binary_predicate import NeuralBinaryPredicate

class Reasoner:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_with_formula(self, formula:FirstOrderFormula):
        pass

    @abstractmethod
    def get_embedding(self, term_name):
        pass

    @abstractmethod
    def estimate_lifted_embeddings(self):
        pass


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
                                    free_var_emb_dict,
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
                    free_var_emb_dict,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Disjunction):
            return self.tnorm.disjunction(
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Negation):
            return self.tnorm.negation(
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formula,
                    begin_index,
                    end_index))

        elif isinstance(formula, BinaryPredicate):
            head_name = formula.head.name
            if free_var_emb_dict and formula.head.is_free:
                head_emb = free_var_emb_dict[head_name]
            else:
                head_emb = self.get_embedding(
                    head_name, begin_index, end_index)

            tail_name = formula.tail.name
            if free_var_emb_dict and formula.tail.is_free:
                tail_emb = free_var_emb_dict[tail_name]
            else:
                tail_emb = self.get_embedding(
                    tail_name, begin_index, end_index)

            rel_emb = self.nbp.get_relation_emb(
                formula.relation_id_list[begin_index: end_index])

            batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
            batch_truth_value = self.nbp.score2truth_value(batch_score)
#             batch_truth_value = batch_score  # CQD's trick for 2i, 3i
            return batch_truth_value


class GradientEFOReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer):
        self.reasoning_rate = reasoning_rate
        self.reasoning_steps = reasoning_steps
        self.reasoinng_optimizer = reasoning_optimizer
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm

        # determined during the optimization
        self.formula: FirstOrderFormula = None
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}

    @classmethod
    def create(cls,
               nbp: NeuralBinaryPredicate,
               tnorm: Tnorm,
               reasoning_rate,
               reasoning_steps,
               reasoning_optimizer,
               sigma=1):
        rm = cls(nbp,
                 tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 sigma)
        return rm

    def initialize_with_formula(self, formula: FirstOrderFormula):
        self.formula = formula
        self.term_local_emb_dict = {
            term_name: None
            for term_name in self.formula.term_dict}

        self._last_ground_free_var_emb = {}

        # self.estimate_lifted_embeddings()

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor.detach().clone()
        self.term_local_emb_dict[key].requires_grad = True

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def initialize_variable_embeddings(self):
        """
        Input args:
        Return args:
            evars: list of existential variables
            uvars: list of universal variables
            fvars: list of free variables
        """

        def check_all_var_initialized():
            return all(self.term_initialized(term_name)
                       for term_name in self.formula.term_dict)

        while not check_all_var_initialized():
            for rel_name, pred in self.formula.predicate_dict.items():
                head_name, tail_name = pred.head.name, pred.tail.name

                if self.term_initialized(head_name) and not self.term_initialized(tail_name):
                    head_emb = self.get_embedding(
                        head_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        self.formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    self.set_local_embedding(tail_name, tail_emb)

                elif not self.term_initialized(head_name) and self.term_initialized(tail_name):
                    tail_emb = self.get_embedding(
                        head_name
                    )
                    rel_emb = self.nbp.get_relation_emb(
                        self.formula.pred_grounded_relation_id_dict[rel_name]
                    )

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)
                    self.set_local_embedding(head_name, head_emb)

                else:
                    continue
        return


    def initialize_variable_embeddings_v2(self):
        # normal initialization
        for symb_name in self.formula.symbol_dict:
            symb_emb = self.get_embedding(symb_name)

        for term_name in self.formula.existential_variable_dict:
            init_vec = torch.normal(0, 1e-3, symb_emb.shape, device=symb_emb.device)
            self.set_local_embedding(term_name, init_vec)

        for term_name in self.formula.free_variable_dict:
            init_vec = torch.normal(0, 1e-3, symb_emb.shape, device=symb_emb.device)
            self.set_local_embedding(term_name, init_vec)

    # ? to check the free_var treatment
    def get_embedding(self,
                      term_name,
                      begin_index=None,
                      end_index=None):
        """
            free_var_treatment:
                (implemented)
                - all: evaluate across all candidates
                - lift: lift the free variable as the existential variable
                - groundans:{k}: ground the free variable into k random answers
                - groundnoisy:{k}: ground the free variable into k noisy variables
                (to implement)
                - groundansfull: ground the answers to a full answer set
                - groundbarycenterfull: ground the barycenter of the answer set
                - groundbarycenter: ground the barycenter of the answer set
                - groundbarycenter: ground the barycenter of the answer set
        """
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

    # def evaluate_truth_values(self, free_var_treatment, batch_size_eval=None):
    #     """
    #     Input args:

    #     Return args:
    #     """
    #     def run_in_batch(batch_size):
    #         begin_idx = 0
    #         end_idx = begin_idx + batch_size
    #         end_idx = min(self.formula.num_instances, end_idx)
    #         collect = []
    #         while begin_idx < self.formula.num_instances:
    #             ret = self.batch_evaluate_truth_values(
    #                 self.formula.formula,
    #                 begin_idx, end_idx, free_var_treatment)
    #             collect.append(ret)

    #             begin_idx = end_idx
    #             end_idx = begin_idx + batch_size
    #             end_idx = min(self.formula.num_instances, end_idx)

    #         return torch.cat(collect, dim=-1)

    #     if batch_size_eval:
    #         return run_in_batch(batch_size=batch_size_eval)
    #     else:
    #         return run_in_batch(batch_size=self.formula.num_instances)

    # def batch_evaluate_truth_values(self,
    #                                 formula,
    #                                 begin_index,
    #                                 end_index,
    #                                 free_var_treatment):
    #     """
    #     Recursive evaluation of the formula functions
    #     Input args:
    #         formula: the formula at this time
    #     Return args:
    #         - truth values in shape either:
    #              [num candidate answers, batch size]
    #              or
    #              [batch size]

    #     """
    #     if isinstance(formula, Conjunction):
    #         return self.tnorm.conjunction(
    #             self.batch_evaluate_truth_values(
    #                 formula.formulas[0], begin_index, end_index, free_var_treatment),
    #             self.batch_evaluate_truth_values(
    #                 formula.formulas[1], begin_index, end_index, free_var_treatment)
    #         )

    #     elif isinstance(formula, Disjunction):
    #         return self.tnorm.disjunction(
    #             self.batch_evaluate_truth_values(
    #                 formula.formulas[0], begin_index, end_index, free_var_treatment),
    #             self.batch_evaluate_truth_values(
    #                 formula.formulas[1], begin_index, end_index, free_var_treatment)
    #         )

    #     elif isinstance(formula, Negation):
    #         return self.tnorm.negation(
    #             self.batch_evaluate_truth_values(
    #                 formula.formula, begin_index, end_index, free_var_treatment)
    #         )

    #     elif isinstance(formula, BinaryPredicate):
    #         head_name = formula.head.name
    #         tail_name = formula.tail.name
    #         head_emb = self.get_embedding(
    #             head_name, begin_index, end_index, free_var_treatment)
    #         tail_emb = self.get_embedding(
    #             tail_name, begin_index, end_index, free_var_treatment)

    #         rel_emb = self.nbp.get_relation_emb(
    #             formula.relation_id_list[begin_index: end_index])
    #         batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
    #         batch_truth_value = self.nbp.score2truth_value(batch_score)
    #         # batch_truth_value = batch_score  # CQD's trick for 2i, 3i
    #         return batch_truth_value

    def estimate_lifted_embeddings(self):
        self.initialize_variable_embeddings()
        evar_local_emb = [
            self.get_embedding(term_name)
            for term_name in self.formula.existential_variable_dict
            if self.term_local_emb_dict[term_name] is not None]

        for term_name in self.formula.free_variable_dict:
            assert self.term_local_emb_dict[term_name] is not None
            emb = self.get_embedding(term_name)
            evar_local_emb.append(emb)


        OptimizerClass = getattr(torch.optim, self.reasoinng_optimizer)
        optim: torch.optim.Optimizer = OptimizerClass(
            evar_local_emb, self.reasoning_rate)

        traj = [(-1, -1, -1)]

        for i in range(self.reasoning_steps):
            tv = self.evaluate_truth_values()
                # conjunction when equality
                # for term_name in self.formula.free_variable_dict:
                #     free_var_local_emb = self.get_embedding(
                #         term_name, free_var_treatment='lift')
                #     free_var_ground_emb = self.get_embedding(
                #         term_name, free_var_treatment=free_var_treatment)
                #     free_var_dist = torch.sum(
                #         (free_var_local_emb - free_var_ground_emb) ** 2, dim=-1)
                #     dist_tv = torch.exp(
                #          free_var_dist / self.sigma ** 2
                #     )
                #     tv = self.tnorm.conjunction(tv, dist_tv)
            # else:
                # tv = self.evaluate_truth_values(free_var_treatment=free_var_treatment)

            ntv = -tv.mean()
            efvar_local_emb_mat = torch.stack(evar_local_emb)
            reg = self.nbp.regularization(efvar_local_emb_mat).mean()

            loss = ntv + reg * 0.05
            traj.append((ntv.item(), reg.item(), loss.item()))
            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.fabs(traj[-1][-1] - traj[-2][-1]) < 1e-9:
                break

        return traj


class RelationalGNNLayer(nn.Module):
    def __init__(self, input_dim, rel_dim, output_dim):
        super(RelationalGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.rel_dim = rel_dim
        self.hidden_dim = output_dim
        self.h2t_linear = nn.Linear(input_dim + rel_dim, output_dim)
        # self.t2h_linear = nn.Linear(input_dim + rel_dim, output_dim)
        self.negation_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, ent_rel_list):
        encoded_entity_list = []
        for ent, rel, rel_order, neg in ent_rel_list:
            # if rel_order == +1, we estimate the tail by head and rel
            # if rel_order == -1, we estimate the head by rel and tail
            entrel = torch.cat([ent, rel], dim=-1)
            if rel_order > 0:
                enc_entrel = self.h2t_linear(entrel)
            else:
                assert NotImplementedError
                enc_entrel = self.t2h_linear(entrel)

            if neg: # if predicate has negation
                enc_entrel = self.negation_layer(enc_entrel)

            encoded_entity_list.append([enc_entrel, rel, rel_order, neg])
        return encoded_entity_list


class RelationalDeepSet(nn.Module):
    def __init__(self, ent_dim, rel_dim, num_layers=1) -> None:
        super(RelationalDeepSet, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.hidden_dim = ent_dim
        self.num_layers = num_layers

        self.init_rlinear = RelationalGNNLayer(self.ent_dim, self.rel_dim, self.hidden_dim)
        for i in range(self.num_layers-1):
            # encode the inputs into the hidden dim.
            setattr(self,
                f'rlinear-{i}',
                RelationalGNNLayer(self.hidden_dim, self.rel_dim, self.hidden_dim))

        self.clf = nn.Sequential(
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.ent_dim)
            )


    def forward(self, ent_rel_list):
        # element wise entity transformation

        def apply_relu(ent_rel_list):
            ent_rel_list = [(torch.relu(e), r, o, n)
                            for e, r, o, n in ent_rel_list]
            return ent_rel_list

        def add(ent_rel_list1, ent_rel_list2):
            ent_rel_list = [
                (e1+e2, r, o, n)
                for (e1, r, o, n), (e2, *_)
                in zip(ent_rel_list1, ent_rel_list2)]
            return ent_rel_list

        hidden_ent_rel_list = self.init_rlinear(ent_rel_list)
        for i in range(self.num_layers-1):
            apply_relu(hidden_ent_rel_list)
            hidden_ent_rel_list = getattr(self,
                                          f'rlinear-{i}')(hidden_ent_rel_list)
            hidden_ent_rel_list = add(hidden_ent_rel_list, ent_rel_list)

        agg = 0
        for e, *_ in hidden_ent_rel_list:
            agg += e
        # out = self.clf(agg)
        return agg


class DeepsetEFOReasoner(Reasoner):
    """
    In this class, we estimate the lifted embeddings of existential variables
    by GNN.
    ? how to handle negation query ?
    The computation order is a unrolling DFS order
    """
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 relational_deepset: RelationalDeepSet):
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm
        self.relational_deepset = relational_deepset

        # formula dependent
        self.formula: FirstOrderFormula = None
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    @classmethod
    def create(cls,
               formula: FirstOrderFormula,
               nbp: NeuralBinaryPredicate,
               tnorm: Tnorm,
               reasoning_rate,
               reasoning_steps,
               reasoning_optimizer,
               sigma=1):
        rm = cls(formula,
                 nbp,
                 tnorm,
                 reasoning_rate,
                 reasoning_steps,
                 reasoning_optimizer,
                 sigma)
        return rm

    def initialize_with_formula(self, formula):
        self.formula: FirstOrderFormula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def estimate_lifted_embeddings(self):
        self.get_embedding('f')

    def get_embedding(self,
                      term_name,
                      begin_index=None,
                      end_index=None):
        """
            If the embedding is known, then return the original ones
            Otherwise, return the lifted embedding estimated by DeepSet
        """
        self.visited_set.add(term_name)

        if self.term_local_emb_dict[term_name] is not None:
            return self.term_local_emb_dict[term_name][begin_index: end_index]

        if self.formula.has_term_grounded_entity_id_list(term_name):
            entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
            if begin_index is not None and end_index is not None:
                entity_id = entity_id[begin_index: end_index]
            emb = self.nbp.get_entity_emb(entity_id)
            self.set_local_embedding(term_name, emb)
        else:
            related_predicate_list = self.formula.term_name2predicate_name_dict[
                term_name]
            ent_rel_ord = []
            for pred_name in related_predicate_list:
                head, tail = self.formula.predicate_dict[pred_name].get_terms()
                neg = self.formula.predicate_dict[pred_name].skolem_negation
                rel_id = self.formula.get_pred_grounded_relation_id_list(pred_name)
                rel_id = rel_id[begin_index: end_index]
                rel = self.nbp.get_relation_emb(rel_id)
                if head.name == term_name:
                    if tail.name in self.visited_set:
                        continue
                    ord = -1
                    ent = self.get_embedding(tail.name, begin_index, end_index)
                elif tail.name == term_name:
                    if head.name in self.visited_set:
                        continue
                    ord = 1
                    ent = self.get_embedding(head.name, begin_index, end_index)
                else:
                    raise ValueError()
                ent_rel_ord.append([ent, rel, ord, neg])
            emb = self.relational_deepset(ent_rel_ord)
            self.set_local_embedding(term_name, emb)
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
                                    free_var_emb_dict,
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
                    free_var_emb_dict,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Disjunction):
            return self.tnorm.disjunction(
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[0],
                    begin_index,
                    end_index),
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formulas[1],
                    begin_index,
                    end_index))

        elif isinstance(formula, Negation):
            return self.tnorm.negation(
                self.batch_evaluate_truth_values(
                    free_var_emb_dict,
                    formula.formula,
                    begin_index,
                    end_index))

        elif isinstance(formula, BinaryPredicate):
            head_name = formula.head.name
            if free_var_emb_dict and formula.head.is_free:
                head_emb = free_var_emb_dict[head_name]
            else:
                head_emb = self.get_embedding(
                    head_name, begin_index, end_index)

            tail_name = formula.tail.name
            if free_var_emb_dict and formula.tail.is_free:
                tail_emb = free_var_emb_dict[tail_name]
            else:
                tail_emb = self.get_embedding(
                    tail_name, begin_index, end_index)

            rel_emb = self.nbp.get_relation_emb(
                formula.relation_id_list[begin_index: end_index])

            batch_score = self.nbp.embedding_score(head_emb, rel_emb, tail_emb)
#             batch_truth_value = self.nbp.score2truth_value(batch_score)
            batch_truth_value = batch_score  # CQD's trick for 2i, 3i
            return batch_truth_value

def complex_vector_multiplication(cva0,cva1,cvb0,cvb1):
    assert (cva0.size(-1) == cva1.size(-1) == cvb0.size(-1) == cvb1.size(-1))
    return torch.cat([
        cva0 * cvb0 - cva1 * cvb1,
        cva0 * cvb1 + cva1 * cvb0
    ], 1)


class LogicalGNNLayerComplEx(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, emb_dim, hidden_dim, num_entities, layers=1, eps=0.1, agg_func='sum'):
        super(LogicalGNNLayerComplEx, self).__init__()
        self.emb_dim = emb_dim
        self.feature_dim = 2 * emb_dim # for complex

        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.agg_func = agg_func

        self.eps = eps

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.layer_to_terms_embs_dict = {}
        if layers == 0:
            self.mlp = lambda x: x
        elif layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 3:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 4:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )


    def message_passing(self, term_emb_dict, predicates, pred_emb_dict):

        term_collect_embs_dict = defaultdict(list)
        for pred in predicates:
            head_name, tail_name = pred.head.name, pred.tail.name
            head_emb = term_emb_dict[head_name]
            head_embs = head_emb[..., :self.emb_dim], head_emb[..., self.emb_dim:]
            tail_emb = term_emb_dict[tail_name]
            tail_embs = tail_emb[..., :self.emb_dim], tail_emb[..., self.emb_dim:]
            pred_emb = pred_emb_dict[pred.name]
            pred_embs = pred_emb[..., :self.emb_dim], pred_emb[..., self.emb_dim:]
            sign = -1 if pred.skolem_negation else 1
            term_collect_embs_dict[head_name].append(
                sign * complex_vector_multiplication(
                    tail_embs[0], tail_embs[1], pred_embs[0], -pred_embs[1])
            )
            term_collect_embs_dict[tail_name].append(
                sign * complex_vector_multiplication(
                    head_embs[0], head_embs[1], pred_embs[0], pred_embs[1])
            )
        return term_collect_embs_dict

    def forward(self, init_term_emb_dict, predicates, pred_emb_dict):
        term_collect_embs_dict = self.message_passing(
            init_term_emb_dict, predicates, pred_emb_dict
        )
        if self.agg_func == 'sum':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) + init_term_emb_dict[t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        elif self.agg_func == 'mean':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) / len(collect_emb_list) + init_term_emb_dict[t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        else:
            raise NotImplementedError
        out_term_emb_dict = {
            t: self.mlp(aggemb)
            for t, aggemb in term_agg_emb_dict.items()
        }
        return out_term_emb_dict


class VanillaGNNLayerComplEx(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, emb_dim, hidden_dim, num_entities, layers=1, eps=0.1, agg_func='sum'):
        super(VanillaGNNLayerComplEx, self).__init__()
        self.emb_dim = emb_dim
        self.feature_dim = 2 * emb_dim # for complex
        self.message_dim = 4 * emb_dim + 2

        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.agg_func = agg_func

        self.eps = eps

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.layer_to_terms_embs_dict = {}
        self.encoder = nn.Sequential(
            nn.Linear(self.message_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )
        if layers == 0:
            self.mlp = lambda x: x
        elif layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.message_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )

    def message_passing(self, term_emb_dict, predicates, pred_emb_dict):
        term_collect_embs_dict = defaultdict(list)
        for pred in predicates:
            head_name, tail_name = pred.head.name, pred.tail.name
            head_emb = term_emb_dict[head_name]
            tail_emb = term_emb_dict[tail_name]
            pred_emb = pred_emb_dict[pred.name]
            sign = -1 if pred.skolem_negation else 1

            head_emb = head_emb.expand(pred_emb.shape)
            tail_emb = tail_emb.expand(pred_emb.shape)
            ones = torch.ones(tail_emb.shape[:-1]).unsqueeze(-1).to(head_emb.device)

            # from head to tail
            term_collect_embs_dict[head_name].append(
                torch.cat([head_emb, pred_emb, ones, sign * ones], -1)
            )
            term_collect_embs_dict[tail_name].append(
                torch.cat([tail_emb, pred_emb, - ones, sign * ones], -1)
            )
        return term_collect_embs_dict

    def forward(self, init_term_emb_dict, predicates, pred_emb_dict):
        term_collect_embs_dict = self.message_passing(
            init_term_emb_dict, predicates, pred_emb_dict
        )
        term_agg_emb_dict = {
            t: self.encoder(sum(collect_emb_list)) + init_term_emb_dict[t] * (self.eps)
            for t, collect_emb_list in term_collect_embs_dict.items()
        }
        out_term_emb_dict = {
            t: self.mlp(aggemb)
            for t, aggemb in term_agg_emb_dict.items()
        }
        return out_term_emb_dict


class GNNEFOReasonerComplEx(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 lgnn_layer: LogicalGNNLayerComplEx,
                 depth_shift=0):
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.formula: FirstOrderFormula = None
        self.depth_shift = depth_shift
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}

    def initialize_with_formula(self, formula):
        self.formula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def initialize_local_embedding(self):
        for term_name in self.formula.term_dict:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
                emb = self.nbp.get_entity_emb(entity_id)
            elif self.formula.term_dict[term_name].is_existential:
                emb = self.lgnn_layer.existential_embedding
            elif self.formula.term_dict[term_name].is_free:
                emb = self.lgnn_layer.free_embedding
            elif self.formula.term_dict[term_name].is_universal:
                emb = self.lgnn_layer.universal_embedding
            else:
                raise KeyError(f"term name {term_name} cannot be initialized")
            self.set_local_embedding(term_name, emb)

    def estimate_lifted_embeddings(self):
        self.initialize_local_embedding()

        predicates = self.formula.predicate_dict.values()
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        for pred_name in self.formula.predicate_dict:
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                relation_id = self.formula.get_pred_grounded_relation_id_list(pred_name)
                emb = self.nbp.get_relation_emb(relation_id)
                pred_emb_dict[pred_name] = emb

        for _ in range(
            max(1, self.formula.quantifier_rank + self.depth_shift)
        ):
            term_emb_dict = self.lgnn_layer(term_emb_dict, predicates, pred_emb_dict)

        for term_name in term_emb_dict:
            # if not self.formula.term_dict[term_name].is_symbol:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]

    def get_embedding(self, term_name, begin_index=None, end_index=None):
        assert self.term_local_emb_dict[term_name] is not None
        emb = self.term_local_emb_dict[term_name]
        if begin_index is not None and end_index is not None:
            emb = emb[begin_index: end_index]
        return emb


class LogicalGNNLayer(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, hidden_dim, nbp: NeuralBinaryPredicate, layers=1, eps=0.1, agg_func='sum'):
        super(LogicalGNNLayer, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_entities = nbp.num_entities
        self.agg_func = agg_func

        self.eps = eps

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.layer_to_terms_embs_dict = {}
        if layers == 0:
            self.mlp = lambda x: x
        elif layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 3:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )
        elif layers == 4:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim),
            )


    def message_passing(self, term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict):

        term_collect_embs_dict = defaultdict(list)
        for pred in predicates:
            head_name, tail_name = pred.head.name, pred.tail.name
            head_emb = term_emb_dict[head_name]
            tail_emb = term_emb_dict[tail_name]
            sign = -1 if pred.skolem_negation else 1

            pred_emb = pred_emb_dict[pred.name]

            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)

            term_collect_embs_dict[tail_name].append(
                sign * self.nbp.estimate_tail_emb(head_emb, pred_emb)
            )

            # term_collect_embs_dict[head_name].append(
            #     sign * self.nbp.estimate_head_emb(tail_emb, pred_emb)
            # )
            pred_emb = inv_pred_emb_dict[pred.name]
            term_collect_embs_dict[head_name].append(
                sign * self.nbp.estimate_tail_emb(tail_emb, pred_emb)
            )
        return term_collect_embs_dict

    def forward(self, init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict):
        term_collect_embs_dict = self.message_passing(
            init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict
        )
        if self.agg_func == 'sum':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) + init_term_emb_dict[t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        elif self.agg_func == 'mean':
            term_agg_emb_dict = {
                t: sum(collect_emb_list) / len(collect_emb_list) + init_term_emb_dict[t] * self.eps
                for t, collect_emb_list in term_collect_embs_dict.items()
            }
        else:
            raise NotImplementedError
        out_term_emb_dict = {
            t: self.mlp(aggemb)
            for t, aggemb in term_agg_emb_dict.items()
        }
        return out_term_emb_dict



class GNNEFOReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 tnorm: Tnorm,
                 lgnn_layer: LogicalGNNLayer,
                 depth_shift=0):
        self.nbp = nbp
        self.tnorm: Tnorm = tnorm
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.formula: FirstOrderFormula = None
        self.depth_shift = depth_shift
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}

    def initialize_with_formula(self, formula):
        self.formula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}
        self._last_ground_free_var_emb = {}
        self.visited_set = set()

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor

    def term_initialized(self, term_name):
        return self.formula.has_term_grounded_entity_id_list(term_name) \
                    or self.term_local_emb_dict[term_name] is not None

    def initialize_local_embedding(self):
        for term_name in self.formula.term_dict:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
                emb = self.nbp.get_entity_emb(entity_id)
            elif self.formula.term_dict[term_name].is_existential:
                emb = self.lgnn_layer.existential_embedding
            elif self.formula.term_dict[term_name].is_free:
                emb = self.lgnn_layer.free_embedding
            elif self.formula.term_dict[term_name].is_universal:
                emb = self.lgnn_layer.universal_embedding
            else:
                raise KeyError(f"term name {term_name} cannot be initialized")
            self.set_local_embedding(term_name, emb)

    def estimate_lifted_embeddings(self):
        self.initialize_local_embedding()

        predicates = self.formula.predicate_dict.values()
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        for pred_name in self.formula.predicate_dict:
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                relation_id = self.formula.get_pred_grounded_relation_id_list(pred_name)
                emb = self.nbp.get_relation_emb(relation_id, inv=False)
                pred_emb_dict[pred_name] = emb
                emb = self.nbp.get_relation_emb(relation_id, inv=True)
                inv_pred_emb_dict[pred_name] = emb

        for _ in range(
            max(1, self.formula.quantifier_rank + self.depth_shift)
        ):
            term_emb_dict = self.lgnn_layer(term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict)

        for term_name in term_emb_dict:
            # if not self.formula.term_dict[term_name].is_symbol:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]

    def get_embedding(self, term_name, begin_index=None, end_index=None):
        assert self.term_local_emb_dict[term_name] is not None
        emb = self.term_local_emb_dict[term_name]
        if begin_index is not None and end_index is not None:
            emb = emb[begin_index: end_index]
        return emb
