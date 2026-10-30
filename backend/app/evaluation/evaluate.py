import argparse
import json
import time
from pathlib import Path
from app.environment.env import CommitGymEnv, SCENARIOS
from app.agents.policies import Baseline, PPOPolicy

def episode_metrics(env):
    success = env.s.transaction == 'completed'
    unsafe = any(not c['passed'] for c in env.info()['invariants'] if c['critical'])
    return {'success': success, 'critical_violations': env.violations, 'unsafe_terminal': unsafe,
            'reward': env.total_reward, 'actions': env.s.step, 'retries': env.retries, 'rollbacks': env.rollbacks,
            'incident': env.s.incident, 'recovery_success': success and env.s.incident, 'failure': not success,
            'simulation_steps': env.s.step}

def aggregate(rows):
    n=len(rows)
    incident_count=sum(r['incident'] for r in rows)
    return {'sample_count':n, 'safe_completion_rate':sum(r['success'] for r in rows)/n,
            'critical_invariant_violations':sum(r['critical_violations'] for r in rows),
            'unsafe_terminal_states':sum(r['unsafe_terminal'] for r in rows),
            'average_reward':sum(r['reward'] for r in rows)/n, 'actions_per_episode':sum(r['actions'] for r in rows)/n,
            'retry_count':sum(r['retries'] for r in rows), 'rollback_count':sum(r['rollbacks'] for r in rows),
            'recovery_sample_count':incident_count,
            'recovery_success_rate':sum(r['recovery_success'] for r in rows)/incident_count if incident_count else None,
            'failure_rate':sum(r['failure'] for r in rows)/n, 'simulation_steps':sum(r['simulation_steps'] for r in rows)}

def evaluate(policies=('baseline','ppo'), seeds=None, checkpoint='models/ppo_commitgym.zip'):
    seeds=seeds if seeds is not None else list(range(1000000,1000020))
    result={'split':'held_out', 'seeds':seeds, 'scenarios':list(SCENARIOS), 'policies':{}, 'episodes':[]}
    for name in policies:
        policy=Baseline() if name=='baseline' else PPOPolicy(checkpoint) if name=='ppo' else None
        if policy is None: raise ValueError('Use the API for explicitly enabled, budgeted Nemotron evaluation')
        rows=[]; started=time.perf_counter()
        for scenario in SCENARIOS:
            for seed in seeds:
                env=CommitGymEnv(scenario); env.reset(seed=seed)
                while not env.done: env.step(policy.choose(env))
                row={'policy':name,'scenario':scenario,'seed':seed,**episode_metrics(env)}
                rows.append(row); result['episodes'].append(row)
        result['policies'][name]={**aggregate(rows),'elapsed_seconds':time.perf_counter()-started,'llm_requests':0,'tokens':0}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=20); a=p.parse_args()
    result=evaluate(seeds=list(range(1000000,1000000+a.samples)))
    Path('models/evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['policies'],indent=2))
