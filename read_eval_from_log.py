import argparse
from collections import defaultdict
import pandas as pd
pd.options.display.float_format = '{:>5.2f}'.format

import json


metrics = ['mrr', 'hit1', 'hit3', 'hit10']

Conjunction_queries = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip']
Negation_queries = ['2in', '3in', 'inp', 'pin', 'pni']
Disjunction_queries = ['2u', 'up']
queries = Conjunction_queries + Disjunction_queries + Negation_queries

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, default='log/output.log')

def read_log_lines(filename):
    lines = []
    with open(filename, 'rt') as f:
        for line in f.readlines():
            lines.append(line)
    return lines

def filter_lines(lines, key_str):
    rec = []
    for line in lines:
        if not key_str in line:
            continue
        json_content = json.loads(line.split(']')[-1])
        try:
            rec.append(
                    json_content
                    )
        except:
            print("Error in ", line)
    return rec

def evaluation_to_tables(line, collect_metrics=metrics, verbose=True):
    data = defaultdict(list)
    for m in collect_metrics:
        data['metric'].append(m)
        for k in queries:
            if k in line:
                data[k].append(line[k][m] * 100)
        data['epoch'] = line['epoch']
    df = pd.DataFrame(data)
    try:
        df['epfo mean'] = df[Conjunction_queries + Disjunction_queries].mean(axis=1)
    except:
        pass
    try:
        df['Neg mean'] = df[Negation_queries].mean(axis=1)
    except:
        pass
    return df

def aggregate_evaluations(lines, key_str, collect_metrics=metrics, out_dir="unnamed"):
    print(key_str)
    rec_lines = filter_lines(lines, key_str)
    df_list = []
    for e, line in enumerate(rec_lines):
        df = evaluation_to_tables(line, collect_metrics)
        df_list.append(df)
    if len(df_list) == 0:
        return
    final_df = pd.concat(df_list)
    final_df = final_df.set_index(['epoch', 'metric'])
    return final_df

if __name__ == "__main__":
    args = parser.parse_args()

    lines = read_log_lines(args.log_file)
    #df = aggregate_evaluations(lines, 'NN evaluate valid', collect_metrics=['mrr'])
    #print(df.to_string(col_space=5))
    df = aggregate_evaluations(lines, 'NN evaluate test', collect_metrics=['mrr'])
    print(df.to_markdown())
