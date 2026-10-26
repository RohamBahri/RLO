import json
import math
from pathlib import Path
import pytest
from rlo.wave4_report import summarize_run,validation_score


def test_score_sizes():
    v={'validation':{str(n):{'cost':2*math.sqrt(n),'feasible':1.} for n in (10,20,30)}}
    assert abs(validation_score(v)-2)<1.e-12


def test_no_best_checkpoint_selection(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'method':'az','seed':0,'max_moves':100}))
    values=[1.,2.,3.,4.,5.]
    rows=[]
    for i,v in enumerate(values):
        rows.append(dict(budget_moves=20*(i+1),real_moves=10*(i+1),
                         simulated_moves=10*(i+1),optimizer_updates=i+1,
                         elapsed_seconds=i+1,validation={str(n):{'cost':v*math.sqrt(n),
                         'feasible':1.} for n in (10,20,30)}))
    (tmp_path/'history.jsonl').write_text(''.join(json.dumps(z)+'\n' for z in rows))
    score=summarize_run(tmp_path)
    assert score['score']==pytest.approx(3.5) # final four, not best checkpoint 1.0
    assert score['final_score']==pytest.approx(5.)
    assert score['scored_move_points']==[40,60,80,100]


def test_incomplete_rejected(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'method':'az','seed':0,'max_moves':100}))
    (tmp_path/'history.jsonl').write_text(json.dumps({'budget_moves':10})+'\n')
    with pytest.raises(ValueError,match='incomplete'):
        summarize_run(tmp_path)
