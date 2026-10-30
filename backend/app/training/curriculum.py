from app.environment.env import CommitGymEnv
from app.agents.policies import Baseline

class CurriculumEnv(CommitGymEnv):
    """Training-only resets to reachable states; no expert labels enter PPO updates."""
    full_start_probability = 0.1
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.np_random.random() < self.full_start_probability: return obs, info
        original_seed=self.seed_value
        prefix=int(self.np_random.integers(1,12))
        actions=[]
        policy=Baseline()
        while not self.done:
            action=policy.choose(self)
            if action==9: break
            actions.append(action)
            self.step(action)
        super().reset(seed=original_seed,options=options)
        for action in actions[:min(prefix,len(actions))]: self.step(action)
        return self.observation(),self.info()
