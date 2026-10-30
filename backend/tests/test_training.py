from app.training.train import train
from app.agents.policies import PPOPolicy
from app.evaluation.evaluate import evaluate

def test_train_save_load_evaluate(tmp_path):
    result=train(1024,seed=29,output=str(tmp_path))
    path=tmp_path/'ppo_commitgym.zip'
    assert path.exists() and result['timesteps']>=1024
    policy=PPOPolicy(str(path))
    report=evaluate(policies=['ppo'],seeds=[1000101],checkpoint=str(path))
    assert report['policies']['ppo']['sample_count']==3
    assert report['policies']['ppo']['simulation_steps']>0
