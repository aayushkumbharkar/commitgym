"""Explicit live smoke test for the configured provider (LLM_PROVIDER); never run by pytest.
Makes one catalog lookup and at most one chat request. Prints no credentials, raw responses or model text."""
from dotenv import load_dotenv
load_dotenv()
from app.agents.nemotron import Nemotron, AgentUnavailable
from app.environment.env import CommitGymEnv, ACTIONS
from app.services.store import Store
agent=Nemotron(Store())
print('Provider:',agent.provider,'| configured model:',agent.model)
try:
    print('Catalog lists model:',agent.verify_model())
    env=CommitGymEnv('partial_success');env.reset(seed=42)
    actions,decision=agent.decide(env,0)
    r=agent.last_request
    print('Request status:',r['status'],'| returned model:',r.get('response_model'),'| tokens:',r.get('tokens'),'| latency s:',r['latency_seconds'])
    print('Validated tool call: operate_simulator',decision['arguments'])
    for action in actions:
        _,reward,_,_,info=env.step(ACTIONS.index(action))
        print('  executed',action,'->',info['event']['outcome']['status'],'|',info['event']['outcome']['message'],'| reward',reward)
        if env.done or info['event']['outcome']['status']!='ok':break
    print('Invariants:',{c['name']:c['passed'] for c in env.info()['invariants']})
except AgentUnavailable as exc:
    print('Live inference unavailable:',str(exc))
    if agent.last_request: print('Request record:',{k:v for k,v in agent.last_request.items() if k!='at'})
    raise SystemExit(1)
