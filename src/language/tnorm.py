from abc import abstractmethod
import torch


class Tnorm:
    def negation(self, a):
        return 1-a

    @staticmethod
    def get_tnorm(name):
        if name == 'product':
            return ProductTNorm
        elif name == 'godel':
            return GodelTNorm
        else:
            raise ValueError('Unknown t-norm: {}'.format(name))


class ProductTNorm(Tnorm):
    @classmethod
    def conjunction(self, a, b):
        return a * b

    @classmethod
    def disjunction(self, a, b):
        return a + b - a * b

class GodelTNorm(Tnorm):
    @classmethod
    def conjunction(self, a, b):
        return torch.min(a, b)

    @classmethod
    def disjunction(self, a, b):
        return torch.max(a, b)
