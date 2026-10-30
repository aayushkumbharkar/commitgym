import pytest
from fastapi.testclient import TestClient
import app.main as api
from app.services.store import Store

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'store',Store(str(tmp_path/'test.sqlite3')))
    monkeypatch.delenv('NEBIUS_API_KEY',raising=False)
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
    assert not client.get('/api/health').json()['nebius']['configured']
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
