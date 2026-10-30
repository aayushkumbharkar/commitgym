"""Explicit live smoke test; never run by pytest. No credentials or raw responses printed."""
from dotenv import load_dotenv
from app.agents.nemotron import Nemotron, AgentUnavailable
from app.environment.env import CommitGymEnv, ACTIONS
from app.services.store import Store
load_dotenv()
agent=Nemotron(Store())
try:
    print('Verified catalog model:',agent.verify_model())
    env=CommitGymEnv('partial_success');env.reset(seed=42)
    actions,usage=agent.decide(env,0)
    for action in actions:
        _,_,_,_,info=env.step(ACTIONS.index(action))
        print('Validated action:',action,'Outcome:',info['event']['outcome']['status'])
        if env.done or info['event']['outcome']['status']!='ok':break
    print('One live decision completed. Reported tokens:',usage['tokens'])
except AgentUnavailable as exc:
    print('Live inference unavailable:',str(exc))
    raise SystemExit(1)
