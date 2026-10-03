import numpy as np
from branch_label_cache import cache_key, load_label, store_label


def test_policy_recipe_prefix_and_dual_changes_invalidate_cache(tmp_path):
    case=dict(job_id='a',step=1,index=0,obs=np.ones(2),mask=np.ones(3,bool),
              plans={0:[0],1:[1]},candidates=[0,1])
    tape=[(('a4',1,'signature'),0)]
    key=cache_key(case,tape,'actor0','data-recipe0',{'sla':1},0)
    assert key!=cache_key(case,tape,'actor1','data-recipe0',{'sla':1},0)
    assert key!=cache_key(case,tape,'actor0','data-recipe1',{'sla':1},0)
    assert key!=cache_key(case,tape,'actor0','data-recipe0',{'sla':2},0)
    assert key!=cache_key(case,[(('a4',1,'signature'),1)],'actor0','data-recipe0',{'sla':1},0)
    store_label(tmp_path,key,{'qualified':True},[{'completed':10,'sla':0,'power':0}])
    assert load_label(tmp_path,key)['rows'][0]['completed']==10
    assert load_label(tmp_path,'different') is None
