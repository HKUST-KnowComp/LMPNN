import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from src.language.tnorm import GodelTNorm, ProductTNorm, Tnorm
from src.pipeline.reasoning_machine import (DeepsetEFOReasoner,
                                            GradientEFOReasoner, Reasoner,
                                            GNNEFOReasonerComplEx,
                                            RelationalDeepSet,
                                            VanillaGNNLayerComplEx,
                                            LogicalGNNLayerComplEx,
                                            LogicalGNNLayer, GNNEFOReasoner)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader

torch.autograd.set_detect_anomaly(True)

lstr2name = {
    'r1(s1,f)': '1p',
    '(r1(s1,e1))&(r2(e1,f))': '2p',
    '((r1(s1,e1))&(r2(e1,e2)))&(r3(e2,f))': '3p',
    '(r1(s1,f))&(r2(s2,f))': '2i',
    '((r1(s1,f))&(r2(s2,f)))&(r3(s3,f))': '3i',
    '((r1(s1,e1))&(r2(s2,e1)))&(r3(e1,f))': 'ip',
    '((r1(s1,e1))&(r2(e1,f)))&(r3(s2,f))': 'pi',
    '(r1(s1,f))&(!(r2(s2,f)))': '2in',
    '((r1(s1,f))&(r2(s2,f)))&(!(r3(s3,f)))': '3in',
    '((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f))': 'inp',
    '((r1(s1,e1))&(r2(e1,f)))&(!(r3(s2,f)))': 'pin',
    '((r1(s1,e1))&(!(r2(e1,f))))&(r3(s2,f))': 'pni',
    '(r1(s1,f))|(r2(s2,f))': '2u',
    '((r1(s1,e1))|(r2(s2,e1)))&(r3(e1,f))': 'up',
    '!((!(r1(s1,f)))&(!(r2(s2,f))))': '2u-dm',
    '!(((!(r1(s1,e1)))|(r2(s2,e1)))&(r3(e1,f)))': 'up-dm'
}

name2lstr = {
    "1p": "r1(s1,f)",
    "2p": "r1(s1,e1)&r2(e1,f)",  # 2p
    "3p": "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "2i": "r1(s1,f)&r2(s2,f)",  # 2i
    "3i": "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "ip": "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",  # ip
    "pi": "r1(s1,e1)&r2(e1,f)&r3(s2,f)",  # pi
    "2in": "r1(s1,f)&!r2(s2,f)",  # 2in
    "3in": "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "inp": "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "pin": "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "pni": "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
    "2u": "r1(s1,f)|r2(s2,f)",  # 2u
    "up": "(r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up
    "2u-dm": "!(!r1(s1,f)&!r2(s2,f))",  # 2u-dm
    "up-dm": "!(!r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up-dm
}


negation_query = [
    "r1(s1,f)&!r2(s2,f)",  # 2in
    "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
]


parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')
parser.add_argument("--checkpoint_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--train_queries", action='append')
parser.add_argument("--eval_queries", action='append')

parser.add_argument("--eval_cqd", action="store_true", default=False)
parser.add_argument("--finetune_kge", action="store_true", default=False)
parser.add_argument("--no_relational_inference", action="store_true", default=False)

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=10)
parser.add_argument("--scale", type=float, default=0.1)
parser.add_argument("--p", type=int, default=1)
parser.add_argument("--checkpoint_path")

# optimization for the entire process
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--pretrain_epoch", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--batch_size_eval", type=int, default=64)
parser.add_argument("--batch_size_eval_dataloader", type=int, default=5000)
# need justification
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--learning_rate_pretrain", type=float, default=1e-4)
# need justification
parser.add_argument("--weight_decay", type=float, default=1e-4)
# need justification
parser.add_argument("--noisy_sample_size", type=int, default=128)
# contrastive learning temperature
parser.add_argument("--temp", type=float, default=0.05)
# neg sampling distance margin
parser.add_argument("--dist_margin", type=float, default=10.0)

parser.add_argument("--objective", type=str, default='lift-contrastive_cosine')
parser.add_argument("--contrastive_coef", type=float, default=1.0)
parser.add_argument("--lift_coef", type=float, default=1.0)
parser.add_argument("--tv_coef", type=float, default=1.0)
parser.add_argument("--neg_sample_dist_coef", type=float, default=1.0)
# reasoning machine
parser.add_argument("--reasoner", type=str, default='gnn', choices=['gnn', 'deepset', 'gradient'])
parser.add_argument("--tnorm", type=str, default='product', choices=['product', 'godel'])
# reasoner = gradient
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_steps", type=int, default=1000)
parser.add_argument("--reasoning_optimizer", type=str, default='AdamW')
parser.add_argument("--reasoning_steps_eval", type=int, default=1000)
# reasoner = gnn
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--depth_shift", type=int, default=0)
parser.add_argument("--agg_func", type=str, default='sum')

parser.add_argument("--score", type=str, default='cos', choices=['cos', 'dist'])
parser.add_argument("--gamma", type=float, default=9)


def train_neural_binary_predicate(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    # for each batch
    for ifof, fof in t:
        if fof.lstr != 'r1(s1,f)':
            continue
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_formula(fof)
        reasoner.initialize_local_embedding()

        pos_1answer_list = []
        neg_answers_list = []

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        pos_tv = reasoner.evaluate_truth_values({'f': batch_pos_emb})
        pos_nll = - torch.log(pos_tv + 1e-10).mean()
        neg_tv = reasoner.evaluate_truth_values({'f': batch_neg_emb})
        neg_nll = - torch.log(1 - neg_tv + 1e-10).mean()
        metric_step['pos_tv'] = pos_tv.mean().item()
        metric_step['pos_nll'] = pos_nll.item()
        metric_step['neg_tv'] = neg_tv.mean().item()
        metric_step['neg_nll'] = neg_nll.item()
        loss += pos_nll + neg_nll

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train neural link predictor {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def train_lifted_estimator_v1(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    T = args.temp
    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    # for each batch
    for ifof, fof in t:
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_formula(fof)

        # this procedure is somewhat of low efficiency
        # ? can we change it to batch implementation ?

        reasoner.estimate_lifted_embeddings()
        batch_fvar_emb = reasoner.get_embedding('f')

        if args.lift_coef > 0 and 'lift' in args.objective:
            lifted_tv = reasoner.evaluate_truth_values()
            lifted_nll = - torch.log(lifted_tv + 1e-10).mean()
            metric_step['lifted_tv'] = lifted_tv.mean().item()
            metric_step['lifted_tv_nll'] = lifted_nll.mean().item()
            loss += lifted_nll * args.lift_coef

        pos_1answer_list = []
        neg_answers_list = []

        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        if args.tv_coef > 0:
            pos_tv = reasoner.evaluate_truth_values({'f': batch_pos_emb})
            pos_tv_nll = - torch.log(pos_tv + 1e-10).mean()
            neg_tv = reasoner.evaluate_truth_values({'f': batch_neg_emb})
            neg_tv_nll = - torch.log(1 - neg_tv + 1e-10).mean()
            metric_step['pos_tv'] = pos_tv.mean().item()
            metric_step['pos_tv_nll'] = pos_tv_nll.item()
            metric_step['neg_tv'] = neg_tv.mean().item()
            metric_step['neg_tv_nll'] = neg_tv_nll.item()
            loss += (pos_tv_nll + neg_tv_nll) * args.tv_coef

        if args.contrastive_coef > 0 and 'contrastive_cosine' in args.objective:
            contrastive_pos_score = torch.exp(torch.cosine_similarity(
                batch_pos_emb, batch_fvar_emb, dim=-1) / T)
            contrastive_neg_score = torch.exp(torch.cosine_similarity(
                batch_neg_emb, batch_fvar_emb, dim=-1) / T)

            contrastive_nll = - torch.log(
                contrastive_pos_score / (contrastive_pos_score + contrastive_neg_score.sum(0))
            ).mean()
            metric_step['contrastive_pos_score'] = contrastive_pos_score.mean().item()
            metric_step['contrastive_neg_score'] = contrastive_neg_score.mean().item()
            metric_step['contrastive_nll'] = contrastive_nll.item()
            loss += contrastive_nll * args.contrastive_coef

        if args.neg_sample_dist_coef > 0 and 'neg_sample_dist' in args.objective:
            pos_sample_score = torch.sigmoid(
                args.dist_margin - torch.sum((batch_pos_emb-batch_fvar_emb)**2, dim=-1) / T)
            neg_sample_score = torch.sigmoid(
                args.dist_margin - torch.sum((batch_neg_emb-batch_fvar_emb)**2, dim=-1) / T)

            neg_sample_nll = - torch.log(pos_sample_score + 1e-10).mean() \
                             - torch.log(1 - neg_sample_score + 1e-10).mean()

            metric_step['pos_sample_score'] = pos_sample_score.mean().item()
            metric_step['neg_sample_score'] = neg_sample_score.mean().item()
            metric_step['neg_sample_nll'] = neg_sample_nll.item()
            loss += neg_sample_nll * args.neg_sample_dist_coef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train lifted estimator {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def train_lifted_estimator_v2(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        args):

    T = args.temp
    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    nbp.eval()

    # for each batch
    for ifof, fof in t:
        ####################
        loss = 0
        metric_step = {}

        reasoner.initialize_with_formula(fof)

        # this procedure is somewhat of low efficiency
        # ? can we change it to batch implementation ?

        reasoner.estimate_lifted_embeddings()
        batch_fvar_emb = reasoner.get_embedding('f')
        pos_1answer_list = []
        neg_answers_list = []


        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        batch_neg_emb = nbp.get_entity_emb(
            torch.cat(neg_answers_list, dim=1))

        if args.score == 'cos':
            contrastive_pos_score = torch.exp(torch.cosine_similarity(
                batch_pos_emb, batch_fvar_emb, dim=-1) / T)
            contrastive_neg_score = torch.exp(torch.cosine_similarity(
                batch_neg_emb, batch_fvar_emb, dim=-1) / T)
            contrastive_nll = - torch.log(
                contrastive_pos_score / (contrastive_pos_score + contrastive_neg_score.sum(0))
            ).mean()

        else:
            contrastive_pos_score = -torch.sigmoid(
                args.gamma - torch.norm(batch_pos_emb - batch_fvar_emb, dim=-1))
            contrastive_neg_score = -torch.sigmoid(
                - args.gamma + torch.norm(batch_neg_emb - batch_fvar_emb, dim=-1))
            contrastive_nll = -torch.log(contrastive_pos_score+1e-10).mean() - torch.log(contrastive_neg_score+1e-10).mean()

        metric_step['contrastive_pos_score'] = contrastive_pos_score.mean().item()
        metric_step['contrastive_neg_score'] = contrastive_neg_score.mean().item()
        metric_step['contrastive_nll'] = contrastive_nll.item()
        loss += contrastive_nll


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ####################
        metric_step['loss'] = loss.item()

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train lifted estimator {desc}] {json.dumps(metric_step)}")

    t.close()

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f'
    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
        ranking = ranking.squeeze()
        if fof.hard_answer_list[i]:
            # [1, num_entities]
            hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                        device=nbp.device)
            hard_answer_rank = ranking[hard_answers]

            # remove better easy answers from its rankings
            if fof.easy_answer_list[i][k]:
                easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                            device=nbp.device)
                easy_answer_rank = ranking[easy_answers].view(-1, 1)

                num_skipped_answers = torch.sum(
                    hard_answer_rank > easy_answer_rank, dim=0)
                pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
            else:
                pure_hard_ans_rank = hard_answer_rank.squeeze()

        else:
            pure_hard_ans_rank = ranking[
                torch.tensor(fof.easy_answer_list[i][k], device=nbp.device)]

        # remove better hard answers from its ranking
        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
        num_skipped_answers = torch.sum(
            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
        )
        pure_hard_ans_rank -= num_skipped_answers.reshape(
            pure_hard_ans_rank.shape)

        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().float().numpy()
        hit1 = (pure_hard_ans_rank < 1).detach().cpu().float().numpy()
        hit3 = (pure_hard_ans_rank < 3).detach().cpu().float().numpy()
        hit10 = (pure_hard_ans_rank < 10).detach().cpu().float().numpy()
        metric['mrr'].append(rr.mean())
        metric['hit1'].append(hit1.mean())
        metric['hit3'].append(hit3.mean())
        metric['hit10'].append(hit10.mean())


def evaluate_by_search_emb_then_rank_truth_value(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner):
    """
    Evaluation used in CQD, two phase computation
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            reasoner.initialize_with_formula(fof)
            reasoner.estimate_lifted_embeddings()
            with torch.no_grad():
                truth_value_entity_batch = reasoner.evaluate_truth_values(
                    free_var_emb_dict={
                        'f': nbp.entity_embedding.unsqueeze(1)
                    },
                    batch_size_eval=args.batch_size_eval)  # [num_entities batch_size]
            ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
            ranked_entity_ids = torch.argsort(
                ranking_score, dim=-1, descending=True)
            batch_entity_rankings = torch.argsort(
                ranked_entity_ids, dim=-1, descending=False)
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

            postfix = {}
            # postfix['reasoning_steps'] = len(traj)
            postfix['lstr'] = fof.lstr
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)
            torch.cuda.empty_cache()

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")


def evaluate_by_nearest_search(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: GradientEFOReasoner):
    """
    Evaluation used by nearest neighbor
    1. continuous optimiation of embeddings quant. + free
    2. evaluate all sentences with intermediate optimized
    """
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                reasoner.initialize_with_formula(fof)
                reasoner.estimate_lifted_embeddings()
                batch_fvar_emb = reasoner.get_embedding('f')
                batch_entity_rankings = nbp.get_all_entity_rankings(
                    batch_fvar_emb, score=args.score)
            # [batch_size, num_entities]
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])
            t.set_postfix({'lstr': fof.lstr})

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))
            pprint(sum_metric)

        postfix = {}
        for name in ['1p', '2p', '3p', '2i', 'inp']:
            if name in sum_metric:
                postfix[name + '_hit3'] = sum_metric[name]['hit3']
        torch.cuda.empty_cache()


    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")


if __name__ == "__main__":
    # * parse argument
    args = parser.parse_args()

    # * prepare the logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    # * initialize the kgindex
    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    # * load neural binary predicate
    print(f"loading the nbp {args.model_name}")
    nbp = get_nbp_class(args.model_name)(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        scale=args.scale,
        device=args.device)


    if args.checkpoint_path:
        print("loading model from", args.checkpoint_path)
        nbp.load_state_dict(torch.load(args.checkpoint_path), strict=True)

    nbp.to(args.device)
    print(f"model loaded from {args.checkpoint_path}")

    # * initialize reasoning machine
    if args.tnorm == 'product':
        tnorm = ProductTNorm
    else:
        tnorm = GodelTNorm

    if args.reasoner == 'deepset':
        ent_dim = nbp.entity_embedding.size(1)
        rel_dim = nbp.relation_embedding.size(1)
        rds = RelationalDeepSet(
            ent_dim, rel_dim, num_layers=args.num_layers).to(nbp.device)
        reasoner = DeepsetEFOReasoner(nbp, tnorm, rds)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            # list(rds.parameters()) + list(nbp.parameters()),
            list(rds.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    elif args.reasoner == 'gnn':
        if args.no_relational_inference:
            lgnn_layer = VanillaGNNLayerComplEx(nbp.embedding_dim,
                                                hidden_dim=args.hidden_dim,
                                                num_entities=nbp.num_entities,
                                                layers=args.num_layers,
                                                eps=args.eps,
                                                agg_func=args.agg_func)
        else:
            lgnn_layer = LogicalGNNLayer(hidden_dim=args.hidden_dim,
                                         nbp=nbp,
                                         layers=args.num_layers,
                                         eps=args.eps,
                                         agg_func=args.agg_func)
        lgnn_layer.to(nbp.device)
        reasoner = GNNEFOReasoner(nbp, tnorm, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        if args.finetune_kge:
            optimizer_estimator = getattr(torch.optim, args.optimizer)(
                list(lgnn_layer.parameters()) + list(nbp.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
        else:
            optimizer_estimator = getattr(torch.optim, args.optimizer)(
                list(lgnn_layer.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

    elif args.reasoner == 'gradient':
        reasoner = GradientEFOReasoner(nbp, tnorm,
                                       reasoning_rate=args.reasoning_rate,
                                       reasoning_steps=args.reasoning_steps,
                                       reasoning_optimizer=args.reasoning_optimizer)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            nbp.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    # * prepare dataset``
    print("loading dataset")
    if args.train_queries:
        train_queries = [name2lstr[tq] for tq in args.train_queries]
    else:
        train_queries = list(name2lstr.values())
    print("train queries", train_queries)

    if args.eval_queries:
        eval_queries = [name2lstr[tq] for tq in args.eval_queries]
    else:
        eval_queries = list(name2lstr.values())
    print("eval queries", eval_queries)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        target_lstr=eval_queries,
        batch_size=args.batch_size_eval_dataloader,
        shuffle=False,
        num_workers=0)

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        target_lstr=eval_queries,
        batch_size=args.batch_size_eval_dataloader,
        shuffle=False,
        num_workers=0)

    print("dataset prepared")

    if args.eval_cqd:
        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"CQD evaluate validate set",
            valid_dataloader, nbp,
            reasoner=GradientEFOReasoner(nbp, tnorm,
                                        reasoning_rate=args.reasoning_rate,
                                        reasoning_steps=args.reasoning_steps,
                                        reasoning_optimizer=args.reasoning_optimizer)
            )
        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"CQD evaluate test set",
            test_dataloader, nbp,
            reasoner=GradientEFOReasoner(nbp, tnorm,
                                        reasoning_rate=args.reasoning_rate,
                                        reasoning_steps=args.reasoning_steps,
                                        reasoning_optimizer=args.reasoning_optimizer)
            )
        exit()


    if args.pretrain_epoch > 0:
        zero_lmpnn = LogicalGNNLayer(hidden_dim=0,
                                     nbp=nbp,
                                     layers=0)
        zero_lmpnn.to(nbp.device)
        zero_reasoner = GNNEFOReasonerComplEx(nbp, tnorm, zero_lmpnn)
        train_dataloader_1p = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=["r1(s1,f)"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        valid_dataloader_1p = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'valid-qaa.json'),
            target_lstr=["r1(s1,f)"],
            batch_size=args.batch_size_eval_dataloader,
            shuffle=False,
            num_workers=0)
        test_dataloader_1p = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'test-qaa.json'),
            target_lstr=["r1(s1,f)"],
            batch_size=args.batch_size_eval_dataloader,
            shuffle=False,
            num_workers=0)
        optimizer_nbp = getattr(torch.optim, args.optimizer)(
            nbp.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        for e in range(args.pretrain_epoch):
            train_lifted_estimator_v2(f"pretrain epoch {e}",
                                      train_dataloader_1p, nbp, zero_reasoner, optimizer_nbp, args)

            if (e+1) % 20 == 0:
                evaluate_by_nearest_search(e, f"NN evaluate validate set pretrain epoch {e+1}",
                                           valid_dataloader_1p, nbp, zero_reasoner)
                evaluate_by_nearest_search(e, f"NN evaluate test set pretrain epoch {e+1}",
                                           test_dataloader_1p, nbp, zero_reasoner)
                last_name = os.path.join(args.checkpoint_dir,
                                        f'pretrain-nbp-last.ckpt')
                torch.save(nbp.state_dict(), last_name)
            if (e+1) % 200 == 0:
                save_name = os.path.join(args.checkpoint_dir,
                                        f'pretrain-nbp-{e+1}.ckpt')
                torch.save(nbp.state_dict(), save_name)
                logging.info(f"pretrain nbp at epoch {e+1} is saved to {save_name}")


    if args.epoch > 0:
        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            # size_limit=args.batch_size * 1,
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)

        for e in range(args.epoch):
            train_lifted_estimator_v2(f"epoch {e}",
                                      train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if (e+1) % 5 == 0:
                evaluate_by_nearest_search(e, f"NN evaluate validate set epoch {e+1}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search(e, f"NN evaluate test set epoch {e+1}",
                                           test_dataloader, nbp, reasoner)
                last_name = os.path.join(args.checkpoint_dir,
                                        f'lmpnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)
            if (e+1) % 20 == 0:
                save_name = os.path.join(args.checkpoint_dir,
                                        f'lmpnn-{e+1}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)
                logging.info(f"lmpnn at epoch {e+1} is saved to {save_name}")
