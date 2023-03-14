"""
A file maintains various reasoners
"""
from abc import abstractmethod
from collections import defaultdict
import math

import torch
from torch import nn
from src.language.foq import (Atomic, Conjunction, Disjunction,
                              EFO1Query, Negation, Term)
from src.language.tnorm import Tnorm
from src.structure.neural_binary_predicate import NeuralBinaryPredicate

class Reasoner:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_with_query(self, formula:EFO1Query):
        pass

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
                    term_name,
                    begin_index=None,
                    end_index=None,
                    inv=False):
        emb = self.nbp.get_relation_emb(
            self.formula.get_pred_grounded_relation_id_list(term_name), inv)
        if begin_index is not None and end_index is not None:
            emb = emb[begin_index: end_index]
        return emb


    @abstractmethod
    def estimate_variable_embeddings(self):
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


class GradientEFOReasoner(Reasoner):
    """
    Gradient based Reasoner (CQD-CO) for Existential First Order (EFO) formulas.
    """
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
        self.formula: EFO1Query = None
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

    def initialize_with_query(self, formula: EFO1Query):
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
            for atomic_name, pred in self.formula.atomic_dict.items():
                pred_name = pred.name
                head_name, tail_name = pred.head.name, pred.tail.name

                if self.term_initialized(head_name) and not self.term_initialized(tail_name):
                    head_emb = self.get_ent_emb(
                        head_name
                    )
                    rel_emb = self.get_rel_emb(pred_name)

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    self.set_local_embedding(tail_name, tail_emb)

                elif not self.term_initialized(head_name) and self.term_initialized(tail_name):
                    tail_emb = self.get_ent_emb(
                        head_name
                    )
                    rel_emb = self.get_rel_emb(pred_name)

                    head_emb = self.nbp.estimate_head_emb(tail_emb, rel_emb)
                    # formula.set_var_local_embedding(head_name, head_emb)
                    self.set_local_embedding(head_name, head_emb)

                else:
                    continue
        return

    def estimate_variable_embeddings(self):
        self.initialize_variable_embeddings()
        evar_local_emb = [
            self.get_ent_emb(term_name)
            for term_name in self.formula.existential_variable_dict
            if self.term_local_emb_dict[term_name] is not None]

        for term_name in self.formula.free_variable_dict:
            assert self.term_local_emb_dict[term_name] is not None
            emb = self.get_ent_emb(term_name)
            evar_local_emb.append(emb)


        OptimizerClass = getattr(torch.optim, self.reasoinng_optimizer)
        optim: torch.optim.Optimizer = OptimizerClass(
            evar_local_emb, self.reasoning_rate)

        traj = [(-1, -1, -1)]

        for i in range(self.reasoning_steps):
            tv = self.evaluate_truth_values()

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
            sign = -1 if pred.negated else 1

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
                 lgnn_layer: LogicalGNNLayer,
                 depth_shift=0):
        self.nbp = nbp
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.formula: EFO1Query = None
        self.depth_shift = depth_shift
        self.term_local_emb_dict = {}
        self._last_ground_free_var_emb = {}

    def initialize_with_query(self, formula):
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

    def estimate_variable_embeddings(self):
        self.initialize_local_embedding()

        predicates = self.formula.atomic_dict.values()
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        for atomic_name in self.formula.atomic_dict:
            pred_name = self.formula.atomic_dict[atomic_name].name
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                pred_emb_dict[pred_name] = self.get_rel_emb(pred_name)
                inv_pred_emb_dict[pred_name] = self.get_rel_emb(pred_name, inv=True)

        for _ in range(
            max(1, self.formula.quantifier_rank + self.depth_shift)
        ):
            term_emb_dict = self.lgnn_layer(term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict)

        for term_name in term_emb_dict:
            # if not self.formula.term_dict[term_name].is_symbol:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]
