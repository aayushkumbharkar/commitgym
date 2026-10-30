import json
import os
import secrets
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from app.environment.env import ACTIONS, SCENARIOS, CommitGymEnv
from app.agents.policies import Baseline, PPOPolicy
from app.agents.nemotron import Nemotron, OfflineToolAdapter, AgentUnavailable, ActionName, resolve_api_key
from app.evaluation.evaluate import episode_metrics, aggregate
from app.services.store import Store

load_dotenv()
app=FastAPI(title='CommitGym',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('COMMITGYM_CORS','http://localhost:5173,http://127.0.0.1:5173').split(','),allow_methods=['GET','POST'],allow_headers=['Content-Type','Authorization'])
lock=threading.RLock()
store=Store()

@app.middleware('http')
async def access_guard(request:Request,call_next):
    token=os.getenv('COMMITGYM_ACCESS_TOKEN','')
    public=os.getenv('COMMITGYM_PUBLIC','false').lower()=='true'
    # Only API routes are protected; the SPA shell must load so a viewer can enter the workspace token.
    path=request.url.path
    if request.method!='OPTIONS' and path.startswith('/api/') and path != '/api/health' and (token or public):
        supplied=request.headers.get('authorization','').removeprefix('Bearer ')
        if not token or not secrets.compare_digest(supplied,token): return JSONResponse({'detail':'Workspace access token required'},status_code=401)
    return await call_next(request)

@app.exception_handler(AgentUnavailable)
async def unavailable(_, exc): return JSONResponse({'detail':str(exc)},status_code=503)

class NewEpisode(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scenario: Literal['partial_success','stale_health','dependency']='partial_success'
    seed:int=Field(default=42,ge=0,le=2147483647)
    policy:Literal['baseline','ppo','nemotron','mock_tools']='baseline'
class StepBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:ActionName | None=None
class EvalBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    policies:list[Literal['baseline','ppo','nemotron']]=Field(default=['baseline','ppo'],min_length=1,max_length=3)
    seeds:list[int]=Field(default=[1000000,1000001,1000002],min_length=1,max_length=30)


@lru_cache(maxsize=2)
def cached_ppo(modified):
    return PPOPolicy()

def get_ppo():
    return cached_ppo(Path('models/ppo_commitgym.zip').stat().st_mtime_ns)

def require_policy(name):
    if name=='ppo' and not Path('models/ppo_commitgym.zip').exists(): raise HTTPException(409,'Train a PPO checkpoint first')
    if name=='nemotron' and not resolve_api_key(): raise AgentUnavailable('NEBIUS_API_KEY is not configured on the backend')


def reconstruct(doc):
    env=CommitGymEnv(doc['scenario']); env.reset(seed=doc['seed'])
    for event in doc['events']: env.step(ACTIONS.index(event['action']))
    return env


def persist(doc, env):
    doc.update(env.info())
    doc['events']=env.events
    doc['metrics']=episode_metrics(env)
    return store.save('episode',doc,doc['id'])


def get_doc(id):
    doc=store.get(id,'episode')
    if doc is None: raise HTTPException(404,'Episode not found')
    return doc


def advance(doc,env,action=None,policy=None):
    if env.done: raise HTTPException(409,'Episode already finished')
    if action is not None:
        env.step(ACTIONS.index(action))
        env.events[-1]['decision']={'provider':'Manual','summary':'User selected '+action,'arguments':{}}
    elif doc['policy'] in ('nemotron','mock_tools'):
        # Persist the attempt before a network call, including failed calls, so reloads cannot bypass limits.
        agent=Nemotron(store) if doc['policy']=='nemotron' else OfflineToolAdapter()
        prior=doc.get('model_calls',0)
        if prior >= agent.max_decisions: raise AgentUnavailable('Episode decision limit reached')
        doc['model_calls']=prior+1
        persist(doc,env)
        actions,decision=agent.decide(env,prior)
        doc.setdefault('decisions',[]).append(decision)
        for selected in actions:
            env.step(ACTIONS.index(selected)); env.events[-1]['decision']=decision
            if env.done or env.events[-1]['outcome']['status']!='ok': break
    else:
        selected=(policy or (Baseline() if doc['policy']=='baseline' else get_ppo())).choose(env)
        env.step(selected)
        env.events[-1]['decision']={'provider':doc['policy'],'summary':'Selected '+ACTIONS[selected], 'arguments':{}}
    return persist(doc,env)

@app.get('/api/health')
def health():
    connection=store.get('nebius_connection','connection')
    model=os.getenv('NEBIUS_MODEL','nvidia/Nemotron-3_5-Lightning')
    try: configured,config_error=resolve_api_key() is not None,None
    except AgentUnavailable as e: configured,config_error=False,str(e)
    return {'status':'ok','environment_version':'1.0.0','nebius':{'configured':configured,'config_error':config_error,
            'enabled':os.getenv('NEBIUS_ENABLED','false')=='true','model':model,
            'connection':connection if configured and connection and connection['model']==model else None,'usage':store.budget()},
            'ppo_available':Path('models/ppo_commitgym.zip').exists()}
@app.get('/api/scenarios')
def scenarios(): return [{'id':k,'description':v} for k,v in SCENARIOS.items()]
@app.post('/api/nebius/probe')
def probe(): return {'model':Nemotron(store).verify_model(),'catalog_verified':True,'inference_verified':False}
@app.post('/api/episodes',status_code=201)
def create(body:NewEpisode):
    require_policy(body.policy)
    env=CommitGymEnv(body.scenario); env.reset(seed=body.seed)
    doc=store.save('episode',{**body.model_dump(),'events':[],'decisions':[],'model_calls':0})
    return persist(doc,env)
@app.get('/api/episodes')
def episodes():
    return [{k:v for k,v in d.items() if k not in ('events','decisions')} for d in store.list('episode')]
@app.get('/api/episodes/{id}')
def episode(id:str): return get_doc(id)
@app.get('/api/episodes/{id}/events')
def events(id:str): return get_doc(id)['events']
@app.post('/api/episodes/{id}/step')
def step(id:str,body:StepBody):
    with lock:
        doc=get_doc(id); env=reconstruct(doc)
        # Preserve recorded policy metadata when reconstructing deterministic simulator events.
        env.events=doc['events'].copy()
        return advance(doc,env,body.action)
@app.post('/api/episodes/{id}/run')
def run(id:str):
    with lock:
        doc=get_doc(id); env=reconstruct(doc); env.events=doc['events'].copy()
        if env.done: raise HTTPException(409,'Episode already finished')
        policy=get_ppo() if doc['policy']=='ppo' else Baseline() if doc['policy']=='baseline' else None
        while not env.done:
            try: doc=advance(doc,env,policy=policy)
            except AgentUnavailable as e:
                doc['blocked_reason']=str(e); persist(doc,env); raise
        return doc
@app.post('/api/episodes/{id}/replay')
def replay(id:str):
    doc=get_doc(id); env=reconstruct(doc)
    actual=[{k:v for k,v in e.items() if k!='decision'} for e in doc['events']]
    return {'matched':actual==env.events,'steps':env.s.step,'state':env.s.public(),'seed':env.seed_value}
@app.post('/api/evaluations',status_code=201)
def evaluate(body:EvalBody):
    if len(set(body.seeds))!=len(body.seeds) or any(s<1000000 or s>2147483647 for s in body.seeds): raise HTTPException(422,'Use unique held-out seeds in [1000000,2147483647]')
    if len(set(body.policies))!=len(body.policies): raise HTTPException(422,'Duplicate policy')
    for name in body.policies: require_policy(name)
    result={'split':'held_out','seeds':body.seeds,'scenarios':list(SCENARIOS),'policies':{},'episode_ids':[]}
    with lock:
        for name in body.policies:
            rows=[]; before=store.budget()
            for scenario in SCENARIOS:
                for seed in body.seeds:
                    doc=create(NewEpisode(scenario=scenario,seed=seed,policy=name))
                    try: doc=run(doc['id'])
                    except AgentUnavailable: doc=get_doc(doc['id'])
                    rows.append(doc['metrics']); result['episode_ids'].append(doc['id'])
            after=store.budget()
            result['policies'][name]={**aggregate(rows),'llm_requests':after['requests']-before['requests'],
                'tokens':after['tokens']-before['tokens'],'estimated_cost_usd':after['estimated_spend_usd']-before['estimated_spend_usd']}
    return store.save('evaluation',result)
@app.get('/api/evaluations')
def evaluations(): return store.list('evaluation')
@app.get('/api/evaluations/{id}')
def evaluation(id:str):
    doc=store.get(id,'evaluation')
    if doc is None: raise HTTPException(404,'Evaluation not found')
    return doc
@app.get('/api/training/status')
def training():
    path=Path('models/training.json')
    data=json.loads(path.read_text()) if path.exists() else {'status':'not_started','timesteps':0,'curve':[]}
    eval_path=Path('models/evaluation.json')
    data['evaluation']=json.loads(eval_path.read_text())['policies'] if eval_path.exists() else None
    return data

# Docker serves the built SPA from the same origin; Vite handles development.
if Path('frontend/dist').is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount('/', StaticFiles(directory='frontend/dist', html=True), name='frontend')
