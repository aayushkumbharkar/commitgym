import argparse
import json
import time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from app.training.curriculum import CurriculumEnv

class TrainingLog(BaseCallback):
    def __init__(self, output, requested, seed):
        super().__init__()
        self.output = output
        self.requested = requested
        self.seed = seed
        self.episodes = []
        self.curve = []
        self.started = time.time()
    def write(self, status):
        payload = {'status': status, 'timesteps': self.num_timesteps, 'requested_timesteps': self.requested,
                   'curriculum': 'reachable baseline prefixes, annealed to full starts by 50% of training', 'training_seed': self.seed, 'episodes': len(self.episodes), 'curve': self.curve,
                   'elapsed_seconds': round(time.time()-self.started, 2), 'checkpoint': str(self.output / 'ppo_commitgym.zip')}
        tmp = self.output / 'training.tmp'
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        tmp.replace(self.output / 'training.json')
    def _on_step(self):
        self.training_env.set_attr('full_start_probability', min(1.0, 0.1 + 1.8*self.num_timesteps/self.requested))
        for info in self.locals['infos']:
            if 'episode' in info: self.episodes.append(info['episode'])
        if self.num_timesteps % 2048 == 0:
            recent = self.episodes[-100:]
            if recent:
                self.curve.append({'timesteps': self.num_timesteps, 'mean_reward': sum(e['r'] for e in recent)/len(recent), 'sample_count': len(recent)})
            self.write('training')
        return True

def train(timesteps=100000, seed=7, output='models'):
    torch.set_num_threads(1)
    folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(CurriculumEnv()) for _ in range(4)])
    # All training reset seeds are generated within [0, 1_000_000); held-out seeds start above that range.
    model = PPO('MlpPolicy', env, seed=seed, n_steps=256, batch_size=128, n_epochs=6,
                learning_rate=0.0007, gamma=0.99, ent_coef=0.02, policy_kwargs={'net_arch': [64,64]}, device='cpu', verbose=0)
    log = TrainingLog(folder, timesteps, seed)
    model.learn(total_timesteps=timesteps, callback=log)
    model.save(folder / 'ppo_commitgym')
    log.write('completed')
    (folder / 'episodes.json').write_text(json.dumps(log.episodes), encoding='utf-8')
    env.close()
    return json.loads((folder / 'training.json').read_text())

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--timesteps',type=int,default=100000); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--output',default='models'); a=p.parse_args()
    print(json.dumps(train(a.timesteps,a.seed,a.output), indent=2))
