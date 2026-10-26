"""Real-repository CPU integration correctness checks. Never method selection."""
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('method',[
    'calibrate','az','az_rollout','vtrace_mcts','gvf_mean','gvf_blend',
    'self_imitation','offpolicy','recent_vtrace','dualstream',
    'az_scratch','curriculum'])
def test_one_collection_all_arms(tmp_path,monkeypatch,method):
    pytest.importorskip('rlo.wave3_model')
    from rlo.wave3_model import ResearchPolicy
    from rlo import wave4_train as train
    torch.set_num_threads(1)
    monkeypatch.setattr(train,'evaluate_greedy',lambda *args,**kwargs:{})
    model=ResearchPolicy(width=128,layers=2,heads=4)
    source=tmp_path/'initial.pt'
    if method=='calibrate':
        torch.save(dict(campaign_version=1,campaign_config={'method':'pomo','seed':0},
                        transitions_total=25_000_000,model=model.state_dict()),source)
    else:
        torch.save(dict(wave4_version=1,method='calibrate',seed=0,
                        model=model.state_dict()),source)
    args=['--method',method,'--seed','0','--sizes','4','--batch','2',
          '--max-moves','25','--validate-every','1000000',
          '--checkpoint-every','1000000','--validation-count','1',
          '--simulations','2','--rollouts','2','--k','2',
          '--device','cpu','--out',str(tmp_path/'run')]
    if method in train.WARM | {'calibrate'}:
        args+=['--initial-checkpoint',str(source)]
    train.main(args)
    ckpt=torch.load(tmp_path/'run/last.pt',weights_only=False)
    assert ckpt['counters']['real_moves']>0
    assert ckpt['counters']['optimizer_updates']>=0
    assert (tmp_path/'run/policy.pt').is_file()


def test_exact_resume_recent_vtrace(tmp_path,monkeypatch):
    pytest.importorskip('rlo.wave3_model')
    from rlo import wave4_train as train
    torch.set_num_threads(1)
    monkeypatch.setattr(train,'evaluate_greedy',lambda *args,**kwargs:{})
    common=['--method','recent_vtrace','--seed','0','--sizes','4','--batch','2',
            '--max-moves','75','--validate-every','1000000',
            '--checkpoint-every','1000000','--validation-count','1',
            '--device','cpu']
    train.main(common+['--out',str(tmp_path/'continuous')])
    train.main(common+['--out',str(tmp_path/'interrupted'),
                       '--stop-after-moves','25'])
    train.main(common+['--out',str(tmp_path/'interrupted'),'--resume'])
    left=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    right=torch.load(tmp_path/'interrupted/last.pt',weights_only=False)
    assert left['counters']['real_moves']==right['counters']['real_moves']
    assert left['counters']['optimizer_updates']==right['counters']['optimizer_updates']
    for key in left['model']:
        assert torch.equal(left['model'][key],right['model'][key]),key
