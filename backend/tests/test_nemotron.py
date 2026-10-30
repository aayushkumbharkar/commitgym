import json
from types import SimpleNamespace as NS
import pytest
from app.agents.nemotron import Nemotron, AgentUnavailable
from app.services.store import Store
from app.environment.env import CommitGymEnv

def response(arguments=None,name='operate_simulator'):
    return NS(usage=NS(total_tokens=100,prompt_tokens=80,completion_tokens=20),choices=[NS(message=NS(tool_calls=[NS(function=NS(name=name,arguments=json.dumps(arguments or {'actions':['inspect']})))]))])

@pytest.fixture
def agent(tmp_path,monkeypatch):
    for k,v in {'NEBIUS_ENABLED':'true','NEBIUS_INPUT_USD_PER_MILLION':'1','NEBIUS_OUTPUT_USD_PER_MILLION':'2','NEBIUS_BUDGET_USD':'1','NEBIUS_MAX_REQUESTS':'3'}.items(): monkeypatch.setenv(k,v)
    client=NS(models=NS(list=lambda:NS(data=[NS(id='nvidia/Nemotron-3_5-Lightning')])),chat=NS(completions=NS(create=lambda **kwargs:response())))
    return Nemotron(Store(str(tmp_path/'budget.sqlite3')),client)

def env():
    e=CommitGymEnv(); e.reset(seed=42); return e

def test_structured_calls_and_usage(agent):
    actions,decision=agent.decide(env(),0)
    assert actions==['inspect'] and decision['tokens']==100
    assert agent.store.budget()['requests']==1
    assert agent.store.get('nebius_connection','connection')['inference_verified']

@pytest.mark.parametrize('arguments',[{'actions':['shell']},{'actions':['inspect'],'command':'ls'},{'actions':[]},{'actions':['inspect']*4}])
def test_invalid_arguments_rejected(agent,arguments):
    agent.client.chat.completions.create=lambda **kwargs:response(arguments)
    with pytest.raises(AgentUnavailable,match='invalid operation'): agent.decide(env(),0)

def test_wrong_tool(agent):
    agent.client.chat.completions.create=lambda **kwargs:response(name='run_shell')
    with pytest.raises(AgentUnavailable,match='allowlisted'): agent.decide(env(),0)

def test_timeout_budget_retained(agent):
    def fail(**kwargs): raise TimeoutError('private credential must not appear')
    agent.client.chat.completions.create=fail
    with pytest.raises(AgentUnavailable,match='no automatic retry') as error: agent.decide(env(),0)
    assert 'private credential' not in str(error.value)
    assert agent.store.budget()['estimated_spend_usd']>0
    assert agent.store.budget()['requests']==1

def test_budget_and_episode_limit(agent):
    for _ in range(3): agent.decide(env(),0)
    with pytest.raises(AgentUnavailable,match='budget exhausted'): agent.decide(env(),0)
    with pytest.raises(AgentUnavailable,match='decision limit'): agent.decide(env(),5)

def test_missing_key_and_model(agent,monkeypatch):
    monkeypatch.delenv('NEBIUS_API_KEY',raising=False)
    with pytest.raises(AgentUnavailable,match='not configured'): Nemotron(agent.store).get_client()
    agent.client.models.list=lambda:NS(data=[])
    with pytest.raises(AgentUnavailable,match='unavailable'): agent.verify_model()

def test_authentication_failure_sanitized(agent):
    def fail(): raise RuntimeError('secret')
    agent.client.models.list=fail
    with pytest.raises(AgentUnavailable,match='lookup failed') as exc: agent.verify_model()
    assert 'secret' not in str(exc.value)

def test_zero_budget_disables_inference(agent):
    agent.cap=0
    with pytest.raises(AgentUnavailable,match='positive spending cap'): agent.decide(env(),0)
    assert agent.store.budget()['requests']==0
