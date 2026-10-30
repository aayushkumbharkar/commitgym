import json
from types import SimpleNamespace as NS
import pytest
from app.agents.nemotron import Nemotron, AgentUnavailable
from app.services.store import Store
from app.environment.env import CommitGymEnv

def response(arguments=None,name='operate_simulator'):
    return NS(usage=NS(total_tokens=100,prompt_tokens=80,completion_tokens=20),choices=[NS(message=NS(tool_calls=[NS(function=NS(name=name,arguments=json.dumps(arguments or {'actions':['inspect']})))]))])

@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    # A developer .env must never leak live settings into unit tests.
    for k in list(__import__('os').environ):
        if k.startswith(('LLM_','NEBIUS_','OPENROUTER_')): monkeypatch.delenv(k,raising=False)

@pytest.fixture
def agent(tmp_path,monkeypatch):
    for k,v in {'NEBIUS_API_KEY':'test-nebius-key','NEBIUS_ENABLED':'true','NEBIUS_INPUT_USD_PER_MILLION':'1','NEBIUS_OUTPUT_USD_PER_MILLION':'2','NEBIUS_BUDGET_USD':'1','NEBIUS_MAX_REQUESTS':'3'}.items(): monkeypatch.setenv(k,v)
    client=NS(models=NS(list=lambda:NS(data=[NS(id='nvidia/Nemotron-3_5-Lightning')])),chat=NS(completions=NS(create=lambda **kwargs:response())))
    return Nemotron(Store(str(tmp_path/'budget.sqlite3')),client)

def env():
    e=CommitGymEnv(); e.reset(seed=42); return e

def test_structured_calls_and_usage(agent):
    actions,decision=agent.decide(env(),0)
    assert actions==['inspect'] and decision['tokens']==100
    assert agent.store.budget()['requests']==1
    assert agent.store.get('nebius_connection','connection')['inference_verified']
    assert decision['provider']=='Nebius Token Factory' and agent.last_request['status']=='ok'

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

def test_offline_adapter_uses_validated_tool_batches():
    from app.agents.nemotron import OfflineToolAdapter
    e=env(); adapter=OfflineToolAdapter()
    actions,decision=adapter.decide(e,0)
    assert 1<=len(actions)<=3 and decision['mock'] and decision['tokens']==0 and decision['model'] is None
    assert e.s.step==0  # planning never mutates the real episode
    with pytest.raises(AgentUnavailable,match='decision limit'): adapter.decide(e,adapter.max_decisions)

def test_nebius_key_fallback_and_conflict(monkeypatch):
    from app.agents.nemotron import resolve_nebius_key
    assert resolve_nebius_key() is None
    monkeypatch.setenv('NEBIUS_TOKEN_FACTORY_KEY','fallback-value')
    assert resolve_nebius_key()=='fallback-value'
    monkeypatch.setenv('NEBIUS_API_KEY','fallback-value')
    assert resolve_nebius_key()=='fallback-value'
    monkeypatch.setenv('NEBIUS_API_KEY','primary-value')
    with pytest.raises(AgentUnavailable,match='different values') as error: resolve_nebius_key()
    assert 'primary-value' not in str(error.value) and 'fallback-value' not in str(error.value)

def test_openrouter_key_never_accepted_as_nebius_key(monkeypatch):
    from app.agents.nemotron import resolve_nebius_key
    monkeypatch.setenv('NEBIUS_API_KEY','sk-or-v1-misplaced')
    with pytest.raises(AgentUnavailable,match='OpenRouter key') as error: resolve_nebius_key()
    assert 'misplaced' not in str(error.value)

def openrouter(monkeypatch,key='sk-or-test',model='nvidia/nemotron-3.5-lightning:free'):
    monkeypatch.setenv('LLM_PROVIDER','openrouter'); monkeypatch.setenv('OPENROUTER_MODEL',model)
    if key: monkeypatch.setenv('OPENROUTER_API_KEY',key)

def router_client(create=None,models=('nvidia/nemotron-3.5-lightning:free',)):
    return NS(models=NS(list=lambda:NS(data=[NS(id=m) for m in models])),
              chat=NS(completions=NS(create=create or (lambda **kwargs:response()))))

def test_provider_status_without_key(monkeypatch):
    from app.agents.nemotron import provider_status
    openrouter(monkeypatch,key=None)
    status=provider_status()
    assert status['id']=='openrouter' and status['label']=='OpenRouter' and not status['configured']
    assert 'OPENROUTER_API_KEY is not configured' in status['config_error'] and status['free_model']
    nebius=provider_status('nebius')
    assert not nebius['configured'] and not nebius['enabled']

def test_openrouter_isolated_from_nebius_variables(monkeypatch,tmp_path):
    openrouter(monkeypatch,key=None); monkeypatch.setenv('NEBIUS_API_KEY','nebius-only-key')
    with pytest.raises(AgentUnavailable,match='OPENROUTER_API_KEY is not configured'): Nemotron(Store(str(tmp_path/'i.sqlite3'))).get_client()

def test_openrouter_free_model_decision(tmp_path,monkeypatch):
    openrouter(monkeypatch)
    sent={}
    def create(**kwargs):
        sent.update(kwargs); r=response(); r.model='nvidia/nemotron-3.5-lightning:free'; return r
    agent=Nemotron(Store(str(tmp_path/'b.sqlite3')),router_client(create))
    actions,decision=agent.decide(env(),0)  # no prices or budget needed for a ':free' model
    assert actions==['inspect'] and decision['provider']=='OpenRouter' and decision['response_model']=='nvidia/nemotron-3.5-lightning:free'
    assert sent['extra_body']['provider']=={'allow_fallbacks':False} and sent['tools'][0]['function']['name']=='operate_simulator'
    assert sent['max_tokens']<=1024 and 'models' not in sent
    assert agent.store.get('openrouter_connection','connection')['inference_verified']
    monkeypatch.setenv('LLM_MAX_REQUESTS','1')
    with pytest.raises(AgentUnavailable,match='budget exhausted'): Nemotron(agent.store,router_client()).decide(env(),0)

def test_openrouter_paid_model_refused(tmp_path,monkeypatch):
    openrouter(monkeypatch,model='nvidia/nemotron-3.5-lightning')
    with pytest.raises(AgentUnavailable,match="free OpenRouter"): Nemotron(Store(str(tmp_path/'c.sqlite3')),router_client()).decide(env(),0)

def test_untrusted_endpoint_rejected(tmp_path,monkeypatch):
    openrouter(monkeypatch); monkeypatch.setenv('OPENROUTER_BASE_URL','https://openrouter.ai.evil.example/v1')
    with pytest.raises(AgentUnavailable,match='must be https://openrouter.ai'): Nemotron(Store(str(tmp_path/'d.sqlite3'))).get_client()

def test_returned_model_mismatch_rejected(tmp_path,monkeypatch):
    openrouter(monkeypatch)
    def create(**kwargs): r=response(); r.model='openai/gpt-something'; return r
    agent=Nemotron(Store(str(tmp_path/'m.sqlite3')),router_client(create))
    with pytest.raises(AgentUnavailable,match='provider returned openai/gpt-something'): agent.decide(env(),0)
    assert agent.last_request['status']=='rejected'
    assert not agent.store.get('openrouter_connection','connection')['inference_verified']

def text_reply(content='I cannot help with that.',finish='stop'):
    return NS(model='nvidia/nemotron-3.5-lightning:free',usage=NS(total_tokens=10,prompt_tokens=8,completion_tokens=2),
              choices=[NS(finish_reason=finish,message=NS(content=content,tool_calls=None))])

@pytest.mark.parametrize('reply,match',[(text_reply(),'text or refusal'),(text_reply(finish='length'),'truncated'),
    (NS(model='nvidia/nemotron-3.5-lightning:free',usage=None,choices=[]),'no choices')])
def test_refusal_truncation_and_empty_responses_rejected(tmp_path,monkeypatch,reply,match):
    openrouter(monkeypatch)
    agent=Nemotron(Store(str(tmp_path/'r.sqlite3')),router_client(lambda **kwargs:reply))
    with pytest.raises(AgentUnavailable,match=match): agent.decide(env(),0)
    assert agent.last_request['status']=='rejected'

@pytest.mark.parametrize('arguments',['{"actions":["inspect"]','not json','{"actions":"inspect"}'])
def test_malformed_json_arguments_rejected(tmp_path,monkeypatch,arguments):
    openrouter(monkeypatch)
    reply=NS(model='nvidia/nemotron-3.5-lightning:free',usage=None,choices=[NS(finish_reason='tool_calls',
             message=NS(tool_calls=[NS(function=NS(name='operate_simulator',arguments=arguments))]))])
    with pytest.raises(AgentUnavailable,match='invalid operation arguments'):
        Nemotron(Store(str(tmp_path/'j.sqlite3')),router_client(lambda **kwargs:reply)).decide(env(),0)

def status_error(cls,code):
    import httpx
    request=httpx.Request('POST','https://openrouter.ai/api/v1/chat/completions')
    return cls('provider message with secret detail',response=httpx.Response(code,request=request),body=None)

def test_rate_limit_timeout_and_auth_failures_are_classified(tmp_path,monkeypatch):
    import openai, httpx
    openrouter(monkeypatch)
    cases=[(status_error(openai.RateLimitError,429),'rate_limited','rate limited'),
           (status_error(openai.AuthenticationError,401),'auth_failed','authentication failed'),
           (openai.APITimeoutError(request=httpx.Request('POST','https://openrouter.ai')),'timeout','timed out')]
    for i,(error,kind,text) in enumerate(cases):
        def create(**kwargs): raise error
        agent=Nemotron(Store(str(tmp_path/f'e{i}.sqlite3')),router_client(create))
        with pytest.raises(AgentUnavailable,match=text) as exc: agent.decide(env(),0)
        assert 'secret detail' not in str(exc.value) and agent.last_request['status']==kind
        assert agent.store.budget()['requests']==1  # the attempt still counts against the request cap
