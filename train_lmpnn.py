import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import torch
import tqdm

from src.language.grammar import parse_lstr_to_lformula
from src.language.tnorm import Tnorm
from src.pipeline import (LMPNNReasoner, GradientEFOReasoner, LogicalMPLayer,
                          Reasoner)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader

torch.autograd.set_detect_anomaly(True)

from convert_beta_dataset import beta_lstr2name

lstr2name = {parse_lstr_to_lformula(k).lstr: v for k, v in beta_lstr2name.items()}
name2lstr = {v: k for k, v in lstr2name.items()}

negation_query = [name for name in name2lstr if '!' in name]

parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--train_queries", action='append')
parser.add_argument("--eval_queries", action='append')
parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
parser.add_argument("--batch_size_eval_truth_value", type=int, default=32, help="batch size for evaluating the truth value")
parser.add_argument("--batch_size_eval_dataloader", type=int, default=5000, help="batch size for evaluation")

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--checkpoint_path", required=True, type=str, help="path to the KGE checkpoint")
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=10)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--p", type=int, default=1)

# optimization for the entire process
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--noisy_sample_size", type=int, default=128)
parser.add_argument("--temp", type=float, default=0.05)

# reasoning machine
parser.add_argument("--reasoner", type=str, default='lmpnn', choices=['lmpnn', 'gradient', 'beam'])
parser.add_argument("--tnorm", type=str, default='product', choices=['product', 'godel'])

# reasoner = gradient
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_steps", type=int, default=1000)
parser.add_argument("--reasoning_optimizer", type=str, default='AdamW')

# reasoner = gnn
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--depth_shift", type=int, default=0)
parser.add_argument("--agg_func", type=str, default='sum')
parser.add_argument("--checkpoint_path_lmpnn", type=str, default=None)


def train_LMPNN(
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

        reasoner.initialize_with_query(fof)

        # this procedure is somewhat of low efficiency
        reasoner.estimate_variable_embeddings()
        batch_fvar_emb = reasoner.get_ent_emb('f')
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

        logging.info(f"[train LMPNN {desc}] {json.dumps(metric_step)}")

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
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    foqs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(foqs, desc=desc) as t:
        for query in t:
            reasoner.initialize_with_query(query)
            reasoner.estimate_variable_embeddings()
            with torch.no_grad():
                truth_value_entity_batch = reasoner.evaluate_truth_values(
                    free_var_emb_dict={
                        'f': nbp.entity_embedding.unsqueeze(1)
                    },
                    batch_size_eval=args.batch_size_eval_truth_value)  # [num_entities batch_size]
            ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
            ranked_entity_ids = torch.argsort(
                ranking_score, dim=-1, descending=True)
            batch_entity_rankings = torch.argsort(
                ranked_entity_ids, dim=-1, descending=False)
            compute_evaluation_scores(
                query, batch_entity_rankings, metric[query.lstr])

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

            postfix = {}
            # postfix['reasoning_steps'] = len(traj)
            postfix['lstr'] = query.lstr
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
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                reasoner.initialize_with_query(fof)
                reasoner.estimate_variable_embeddings()
                batch_fvar_emb = reasoner.get_ent_emb('f')
                batch_entity_rankings = nbp.get_all_entity_rankings(
                    batch_fvar_emb, score="cos")
            # [batch_size, num_entities]
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])
            t.set_postfix({'lstr': fof.lstr})

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

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

    for param in nbp.parameters():
        param.requires_grad = False

    # * load the dataset, by default, we load the dataset to test
    print("loading dataset")
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

    if args.reasoner in ['gradient', 'beam']:
        # for those reasoners without training
        tnorm = Tnorm.get_tnorm(args.tnorm)

        # todo: add more reasoners
        reasoner = GradientEFOReasoner(
            nbp,
            tnorm,
            reasoning_rate=args.reasoning_rate,
            reasoning_steps=args.reasoning_steps,
            reasoning_optimizer=args.reasoning_optimizer)

        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"valuate validate set", valid_dataloader, nbp, reasoner)
        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"evaluate test set", test_dataloader, nbp, reasoner)

    elif args.reasoner == 'lmpnn':
        # for those reasoners with training
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = LogicalMPLayer(hidden_dim=args.hidden_dim,
                                     nbp=nbp,
                                     layers=args.num_layers,
                                     eps=args.eps,
                                     agg_func=args.agg_func)

        if args.checkpoint_path_lmpnn:
            print("loading lmpnn model from", args.checkpoint_path_lmpnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_lmpnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = LMPNNReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print("Is entity embedding optimized ?", lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)

        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        for e in range(1, 1+args.epoch):
            train_LMPNN(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if e % 5 == 0:
                evaluate_by_nearest_search(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                        f'lmpnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                        f'lmpnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            evaluate_by_nearest_search(e, f"NN evaluate validate set",
                                        valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search(e, f"NN evaluate test set ",
                                        test_dataloader, nbp, reasoner)