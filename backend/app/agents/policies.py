from app.environment.env import ACTIONS

class Baseline:
    name = 'baseline'
    def choose(self, env):
        s = env.s.public()
        if s['uncertain'] or not s['inspected'] and s['step'] == 0: action = 'inspect'
        elif s['retry_eligible']: action = 'retry' if s['inspected'] else 'inspect'
        elif s['recovered']:
            action = 'health_check' if not s['health_fresh'] else 'verify' if not s['verified'] else 'complete'
        elif s['active_release'] == 1: action = 'stage' if not s['pending_release'] else 'deploy'
        elif not s['health_fresh']: action = 'health_check'
        elif not s['reported_healthy']: action = 'rollback' if s['inspected'] else 'inspect'
        elif s['schema'] != 2: action = 'shift_traffic' if s['traffic'] else 'migrate'
        elif not s['traffic']: action = 'shift_traffic'
        elif not s['verified']: action = 'verify'
        else: action = 'complete'
        return ACTIONS.index(action)

class PPOPolicy:
    name = 'ppo'
    def __init__(self, path='models/ppo_commitgym.zip'):
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device='cpu')
    def choose(self, env):
        action, _ = self.model.predict(env.observation(), deterministic=True)
        return int(action)
