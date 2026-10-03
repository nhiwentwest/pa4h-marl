"""Exercise the real joint PPO path, including the new auxiliary loss."""
import numpy as np
import torch
import marl_gang_train as trainer


def test_joint_ppo_consumes_branch_labels(capsys):
    assert trainer.A4_BRANCH_RANK
    torch.manual_seed(17);rng=np.random.RandomState(17)
    ag=trainer.Agents(16,128)
    g=rng.normal(size=(4,128)).astype(np.float32)
    agents={}
    for name,(dim,actions) in ag.dims.items():
        obs=rng.normal(size=(4,dim)).astype(np.float32)
        masks=np.ones((4,actions),dtype=bool)
        acts=np.arange(4)%actions
        with torch.no_grad():
            logits=ag.actors[name](torch.tensor(obs),torch.tensor(masks))
            lp=torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(acts)).numpy()
        agents[name]=dict(obs=obs,masks=masks,acts=acts,old_lp=lp,team_adv=np.array([-1.,-.5,.5,1.],dtype=np.float32),
            local_adv=None,cf_utility=np.zeros((4,actions),dtype=np.float32))
    agents['a4'].update(global_state=g,behavior_probs=np.ones((4,17),dtype=np.float32)/17,
        realized_return=np.ones(4,dtype=np.float32),old_value=np.zeros(4,dtype=np.float32))
    returns=np.full(17,np.nan);returns[0]=1.;returns[13]=1.4
    labels=[dict(obs=agents['a4']['obs'][0].copy(),mask=agents['a4']['masks'][0].copy(),returns=returns)]
    before={n:{k:v.detach().clone() for k,v in a.state_dict().items()} for n,a in ag.actors.items()}
    trainer.ppo_update_batch(ag,[dict(G=g,RET=np.ones(4,dtype=np.float32),agents=agents,branch_labels=labels)])
    output=capsys.readouterr().out
    assert 'branch_rank: pairs=1' in output
    for n,a in ag.actors.items():
        assert any(not torch.equal(v,a.state_dict()[k]) for k,v in before[n].items())
        assert all(torch.isfinite(v).all() for v in a.state_dict().values())
