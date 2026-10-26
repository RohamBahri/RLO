"""Preregistered Wave4 score from completed, immutable validation histories.

Primary score per seed: mean of the last FOUR completed validation checkpoints,
each score mean_{n=10,20,30}(greedy_cost_n / sqrt(n)). Then average seeds.
No best-checkpoint search, no use of held-out TEST data, no forced 50M-to-25M
normalization. Fresh/total moves and wall time remain separate comparison axes.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


METHODS=('az','az_rollout','vtrace_mcts','gvf_mean','gvf_blend',
         'self_imitation','offpolicy','recent_vtrace','dualstream',
         'az_scratch','curriculum')


def validation_score(row, sizes=(10,20,30)):
    v=row['validation']
    vals=[]
    for n in sizes:
        part=v[str(n)]
        if part.get('feasible',1.)<1-1.e-12:
            raise ValueError(f'feasibility failed on n={n}')
        vals.append(part['cost']/math.sqrt(n))
    return statistics.mean(vals)


def summarize_run(folder,*,n_final=4,require_complete=True):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    lines=[json.loads(z) for z in (folder/'history.jsonl').read_text().splitlines()]
    if not lines:raise ValueError('empty run')
    last=lines[-1]
    max_moves=config['max_moves']
    if require_complete and last['budget_moves']<max_moves:
        raise ValueError(f'incomplete run ({last["budget_moves"]} / {max_moves})')
    candidates=[(z['budget_moves'],validation_score(z)) for z in lines if 'validation' in z]
    if len(candidates)<n_final:
        raise ValueError(f'fewer than {n_final} validation points')
    selected=candidates[-n_final:]
    return dict(method=config['method'],seed=config['seed'],
                score=statistics.mean(v for _,v in selected),
                final_score=selected[-1][1],
                scored_move_points=[m for m,_ in selected],
                real_moves=last['real_moves'],simulated_moves=last['simulated_moves'],
                budget_moves=last['budget_moves'],
                optimizer_updates=last['optimizer_updates'],
                replay_samples=last.get('replay_samples',0),
                gpu_seconds=last.get('elapsed_seconds'),
                train_seconds=last.get('train_seconds'),
                config_file=str(folder/'config.json'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args(argv)
    runs=[]
    for method in METHODS:
        for seed in (0,1,2):
            folder=Path(a.root)/f'{method}_seed{seed}'
            runs.append(summarize_run(folder,require_complete=not a.allow_incomplete))
    grouped=[]
    for method in METHODS:
        subset=[r for r in runs if r['method']==method]
        if len(subset)!=3 or {r['seed'] for r in subset}!={0,1,2}:
            raise AssertionError('missing or duplicated seeds')
        scores=[r['score'] for r in subset]
        grouped.append(dict(method=method,mean_score=statistics.mean(scores),
                            seed_sd=statistics.stdev(scores),
                            min_score=min(scores),max_score=max(scores),
                            mean_final=statistics.mean(r['final_score'] for r in subset),
                            mean_real_moves=statistics.mean(r['real_moves'] for r in subset),
                            mean_simulated_moves=statistics.mean(r['simulated_moves'] for r in subset),
                            mean_elapsed_seconds=statistics.mean(r['gpu_seconds'] for r in subset)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'wave4_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    (out/'wave4_methods.json').write_text(json.dumps(grouped,indent=2)+'\n')
    with (out/'wave4_methods.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)
    for entry in sorted(grouped,key=lambda r:r['mean_score']):
        print(f"{entry['method']:20} {entry['mean_score']:.6f} ± {entry['seed_sd']:.6f}")


if __name__=='__main__':main()
