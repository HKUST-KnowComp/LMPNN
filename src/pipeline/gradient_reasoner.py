import torch
import math

from src.language.foq import EFO1Query
from src.language.tnorm import Tnorm
from src.pipeline.reasoner import Reasoner
from src.structure.neural_binary_predicate import NeuralBinaryPredicate

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

    def initialize_with_query(self, formula: EFO1Query):
        self.formula = formula
        self.term_local_emb_dict = {
            term_name: None
            for term_name in self.formula.term_dict}

    def set_local_embedding(self, key, tensor):
        self.term_local_emb_dict[key] = tensor.detach().clone()
        self.term_local_emb_dict[key].requires_grad = True

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
            for alstr, atomic in self.formula.atomic_dict.items():
                pred_name = atomic.relation
                head_name, tail_name = atomic.head.name, atomic.tail.name

                if self.term_initialized(head_name) and not self.term_initialized(tail_name):
                    head_emb = self.get_ent_emb(
                        head_name
                    )
                    rel_emb = self.get_rel_emb(pred_name)

                    tail_emb = self.nbp.estimate_tail_emb(head_emb, rel_emb)
                    self.set_local_embedding(tail_name, tail_emb)

                elif not self.term_initialized(head_name) and self.term_initialized(tail_name):
                    tail_emb = self.get_ent_emb(
                        tail_name
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
