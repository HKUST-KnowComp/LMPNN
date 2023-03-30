from collections import defaultdict

import torch
from torch import nn
from src.language.foq import EFO1Query
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.pipeline.reasoner import Reasoner

class LogicalMPLayer(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, hidden_dim, nbp: NeuralBinaryPredicate, layers=1, eps=0.1, agg_func='sum'):
        super(LogicalMPLayer, self).__init__()
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


    def message_passing(self, term_emb_dict, atomic_dict, pred_emb_dict, inv_pred_emb_dict):

        term_collect_embs_dict = defaultdict(list)
        for predicate, atomic in atomic_dict.items():
            head_name, tail_name = atomic.head.name, atomic.tail.name
            head_emb = term_emb_dict[head_name]
            tail_emb = term_emb_dict[tail_name]
            sign = -1 if atomic.negated else 1

            pred_emb = pred_emb_dict[atomic.relation]
            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)
            term_collect_embs_dict[tail_name].append(
                sign * self.nbp.estimate_tail_emb(head_emb, pred_emb)
            )

            # inv_pred_emb = inv_pred_emb_dict[atomic.relation]
            # term_collect_embs_dict[head_name].append(
            #     sign * self.nbp.estimate_tail_emb(tail_emb, inv_pred_emb)
            # )
            term_collect_embs_dict[head_name].append(
                sign * self.nbp.estimate_head_emb(tail_emb, pred_emb)
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



class LMPNNReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 lgnn_layer: LogicalMPLayer,
                 depth_shift=0):
        self.nbp = nbp
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.depth_shift = depth_shift

        self.formula: EFO1Query = None
        self.term_local_emb_dict = {}

    def initialize_with_query(self, formula):
        self.formula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}

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
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        for atomic_name in self.formula.atomic_dict:
            pred_name = self.formula.atomic_dict[atomic_name].relation
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                pred_emb_dict[pred_name] = self.get_rel_emb(pred_name)
                inv_pred_emb_dict[pred_name] = self.get_rel_emb(pred_name, inv=True)

        for _ in range(
            max(1, self.formula.quantifier_rank + self.depth_shift)
        ):
            term_emb_dict = self.lgnn_layer(
                term_emb_dict,
                self.formula.atomic_dict,
                pred_emb_dict,
                inv_pred_emb_dict)

        for term_name in term_emb_dict:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]