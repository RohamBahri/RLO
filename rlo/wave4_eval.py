"""Sealed-split inference evaluation (no training, no checkpoint selection)."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import evaluation_instances, fingerprint
from .wave3_model import from_saved
from .wave4_inference import checked,complete,best_of_k,multistart,beam_search,mcts,handoff


def load_policy(path,device):
    safe=torch.load(path,map_location='cpu',weights_only=True)
    if safe.get('wave4_version')==1:
        from .wave3_model import ResearchPolicy
        model=ResearchPolicy(**safe['architecture']).to(device)
        model.load_state_dict(safe['model'])
        return model.eval()
    return from_saved(path,device=device)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--unlock-test',action='store_true')
    p.add_argument('--sizes',nargs='+',type=int,default=[10,12,20,25,30,40,50])
    p.add_argument('--factors',nargs='+',type=float,default=[1.])
    p.add_argument('--count',type=int,default=256)
    p.add_argument('--method',choices=('greedy','multistart','sample','beam','perturbed_beam',
                                        'mcts','handoff'),required=True)
    p.add_argument('--k',type=int,default=32)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.split=='test' and not a.unlock_test:p.error('sealed test requires --unlock-test')
    if a.count<1 or a.k<1 or min(a.sizes)<2 or min(a.factors)<=0:
        p.error('invalid settings')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    model=load_policy(a.checkpoint,a.device)
    rng=np.random.default_rng(a.seed)
    result=[]
    for n in a.sizes:
        for factor in a.factors:
            instances=evaluation_instances(n,a.split,a.count,factor)
            for inst in instances:
                started=time.perf_counter()
                if a.method=='greedy':s,stats=complete(model,inst)
                elif a.method=='multistart':s,stats=multistart(model,inst)
                elif a.method=='sample':s,stats=best_of_k(model,inst,a.k,rng)
                elif a.method=='beam':s,stats=beam_search(model,inst,a.k)
                elif a.method=='perturbed_beam':
                    s,stats=beam_search(model,inst,a.k,stochastic=True,rng=rng)
                elif a.method=='mcts':s,stats=mcts(model,inst,sims=a.k,rng=rng)
                else:s,stats=handoff(model,inst,remaining=min(10,a.k))
                observed=checked(inst,s)
                result.append(dict(id=fingerprint(inst),n=n,factor=factor,cost=observed,
                                   method=a.method,k=a.k,split=a.split,
                                   elapsed_seconds=time.perf_counter()-started,**stats))
            vals=[r['cost'] for r in result if r['n']==n and r['factor']==factor]
            print(json.dumps(dict(n=n,factor=factor,mean_cost=float(np.mean(vals)),
                                  normalized=float(np.mean(vals)/math.sqrt(n)),
                                  count=len(vals))),flush=True)
    target=Path(a.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
