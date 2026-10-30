import pytest
from fastapi.testclient import TestClient
import app.main as api
from app.services.store import Store

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'store',Store(str(tmp_path/'test.sqlite3')))
    import os
    for k in list(os.environ):
        if k.startswith(('LLM_','NEBIUS_','OPENROUTER_')): monkeypatch.delenv(k,raising=False)
    monkeypatch.delenv('COMMITGYM_ACCESS_TOKEN',raising=False)
    monkeypatch.delenv('COMMITGYM_PUBLIC',raising=False)
    return TestClient(api.app)

def test_api_persistence_replay(client):
    response=client.post('/api/episodes',json={'scenario':'partial_success','seed':42})
    assert response.status_code==201
    id=response.json()['id']
    result=client.post(f'/api/episodes/{id}/run').json()
    assert result['success'] and result['events']
    assert client.get(f'/api/episodes/{id}').json()==result
    assert client.get(f'/api/episodes/{id}/events').json()==result['events']
    assert client.post(f'/api/episodes/{id}/replay').json()['matched']
    assert client.post(f'/api/episodes/{id}/step',json={}).status_code==409
    # A fresh store instance simulates a backend restart.
    api.store=Store(api.store.path)
    assert client.get(f'/api/episodes/{id}').json()['success']

def test_validation_and_unavailable(client):
    assert client.post('/api/episodes',json={'scenario':'unknown'}).status_code==422
    assert client.get('/api/episodes/missing').status_code==404
    assert client.post('/api/episodes',json={'policy':'nemotron'}).status_code==503
    assert not client.get('/api/health').json()['nebius']['configured'] and not client.get('/api/health').json()['llm']['configured']
    id=client.post('/api/episodes',json={}).json()['id']
    assert client.post(f'/api/episodes/{id}/step',json={'action':'shell'}).status_code==422
    assert client.post('/api/evaluations',json={'seeds':[2]}).status_code==422

def test_evaluation_real_scenarios(client):
    response=client.post('/api/evaluations',json={'policies':['baseline'],'seeds':[1000000,1000001]})
    assert response.status_code==201
    data=response.json()
    assert data['policies']['baseline']['sample_count']==6
    assert data['policies']['baseline']['safe_completion_rate']==1
    assert len(data['episode_ids'])==6
    assert client.get('/api/evaluations/'+data['id']).json()==data

def test_public_access_guard(client,monkeypatch):
    monkeypatch.setenv('COMMITGYM_PUBLIC','true')
    assert client.post('/api/episodes',json={}).status_code==401
    monkeypatch.setenv('COMMITGYM_ACCESS_TOKEN','test-workspace-token')
    assert client.post('/api/episodes',json={},headers={'Authorization':'Bearer test-workspace-token'}).status_code==201
    assert client.get('/api/episodes').status_code==401 and client.get('/api/health').status_code==200
    # Non-API paths (the SPA shell) are not blocked by the guard; 404 here only because tests may lack frontend/dist.
    assert client.get('/index.html').status_code!=401

def test_mock_tool_adapter_episode_is_labelled(client):
    id=client.post('/api/episodes',json={'scenario':'stale_health','seed':7,'policy':'mock_tools'}).json()['id']
    response=client.post(f'/api/episodes/{id}/run')
    doc=client.get(f'/api/episodes/{id}').json()
    assert 1<=doc['model_calls']<=5 and doc['decisions']
    assert all(d['mock'] and d['tokens']==0 and 'no LLM' in d['provider'] for d in doc['decisions'])
    assert all(e['decision']['mock'] for e in doc['events'])
    assert client.get('/api/health').json()['llm']['usage']['requests']==0
    assert response.status_code==200 or 'decision limit' in response.json()['detail']
    assert client.post(f'/api/episodes/{id}/replay').json()['matched']

def test_health_reports_key_conflict_without_values(client,monkeypatch):
    monkeypatch.setenv('NEBIUS_API_KEY','one-secret'); monkeypatch.setenv('NEBIUS_TOKEN_FACTORY_KEY','two-secret')
    body=client.get('/api/health').text
    assert 'different values' in body and 'one-secret' not in body and 'two-secret' not in body

def test_health_separates_openrouter_and_nebius(client,monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','openrouter'); monkeypatch.setenv('OPENROUTER_API_KEY','sk-or-health-test')
    health=client.get('/api/health')
    data=health.json()
    assert data['llm']['label']=='OpenRouter' and data['llm']['configured'] and data['llm']['model'].endswith(':free')
    assert data['nebius']=={'configured':False,'enabled':False,'active':False,'config_error':'NEBIUS_API_KEY is not configured on the backend'}
    assert 'sk-or-health-test' not in health.text

def test_live_failure_is_recorded_without_mock_fallback(client,monkeypatch):
    import app.agents.nemotron as nemotron
    monkeypatch.setenv('LLM_PROVIDER','openrouter'); monkeypatch.setenv('OPENROUTER_API_KEY','sk-or-test')
    def failing(self, env, prior):
        self.last_request={'provider':'OpenRouter','model':self.model,'status':'rate_limited','error':'rate limited (HTTP 429)','latency_seconds':0.1}
        raise nemotron.AgentUnavailable('OpenRouter request failed: rate limited (HTTP 429); no automatic retry')
    monkeypatch.setattr(nemotron.Nemotron,'decide',failing)
    id=client.post('/api/episodes',json={'policy':'nemotron'}).json()['id']
    response=client.post(f'/api/episodes/{id}/run')
    assert response.status_code==503 and 'rate limited' in response.json()['detail']
    doc=client.get(f'/api/episodes/{id}').json()
    assert doc['model_requests'][0]['status']=='rate_limited' and doc['events']==[] and not doc['decisions']
    assert 'rate limited' in doc['blocked_reason']
