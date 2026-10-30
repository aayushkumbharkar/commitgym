"""Budgeted, allowlisted Token Factory tool calling. Ordinary tests inject a fake client."""
import json
import os
import time
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal
from app.environment.env import ACTIONS

ActionName = Literal['inspect','health_check','stage','deploy','migrate','shift_traffic','verify','retry','rollback','complete']
class Operations(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    actions: list[ActionName] = Field(min_length=1,max_length=3)

TOOL = {'type':'function','function':{'name':'operate_simulator',
        'description':'Execute 1 to 3 ordered simulator actions. Stop the batch on a rejected, stale, timeout or failed result and observe before deciding again. No external operations.',
        'parameters':Operations.model_json_schema()}}
SYSTEM = '''You operate CommitGym, a simulated blue-green deployment transaction. Use operate_simulator only.
Objective: deploy v2/schema2 with healthy compatible traffic, or recover an incident to verified v1.
Actions: inspect reconciles timeout actual commit state; stage stages v2; deploy deploys candidate while traffic stays v1;
health_check probes candidate (stale responses are NOT fresh evidence); shift_traffic drains if traffic is on, otherwise restores to candidate;
migrate updates schema to v2 (drain first); verify independently checks invariants; retry only an inspected uncommitted operation;
rollback atomically restores v1/schema1 snapshot after an inspected incident; complete asks the independent judge.
Never route traffic to an unhealthy or incompatible release. Obtain fresh health and verification before completion.
After a timeout inspect before retrying. After rollback recheck health then verify then complete.
Choose at most 3 sequential operations per call. Do not output private reasoning; the UI shows only actions and evidence.'''

class AgentUnavailable(Exception): pass

# Provider-neutral LLM_* settings take precedence; NEBIUS_* names remain as fallbacks.
KEY_VARS=('LLM_API_KEY','NEBIUS_API_KEY','NEBIUS_TOKEN_FACTORY_KEY')
DEFAULT_MODEL='nvidia/Nemotron-3_5-Lightning'
DEFAULT_BASE='https://api.tokenfactory.nebius.com/v1'
def setting(name, default=''):
    value=os.getenv('LLM_'+name,'').strip()
    return value if value else os.getenv('NEBIUS_'+name,default).strip()
def model_name(): return setting('MODEL',DEFAULT_MODEL)
def base_url(): return setting('BASE_URL',DEFAULT_BASE)
def enabled(): return setting('ENABLED','false').lower()=='true'
def provider_name(url=None):
    from urllib.parse import urlparse
    host=(urlparse(url or base_url()).hostname or '').lower()
    return 'OpenRouter' if host=='openrouter.ai' else 'Nebius Token Factory' if host.endswith('.nebius.com') else host or 'unknown'
def free_model(model=None):
    # OpenRouter ':free' variants are priced at zero; request and decision caps still apply.
    return provider_name()=='OpenRouter' and (model or model_name()).endswith(':free')
def resolve_api_key():
    """First configured of LLM_API_KEY, NEBIUS_API_KEY, NEBIUS_TOKEN_FACTORY_KEY. Values are never echoed."""
    values={k:os.getenv(k,'').strip() for k in KEY_VARS}
    distinct={v for v in values.values() if v}
    if len(distinct)>1:
        names=', '.join(k for k,v in values.items() if v)
        raise AgentUnavailable(f'Configuration error: {names} are set with different values; keep only one')
    return distinct.pop() if distinct else None
def key_configured():
    try: return resolve_api_key() is not None
    except AgentUnavailable: return False

def build_messages(env):
    context={'objective':'Verified safe deployment or incident recovery', 'state':env.s.public(),
             'vector_observation':env.observation().tolist(), 'available_actions':env.available(),
             'invariants':env.info()['invariants'], 'previous_tool_results':env.events[-3:]}
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(context)}]

def parse_tool_call(response):
    """The single validation boundary for model output: one allowlisted call with strict arguments."""
    calls=response.choices[0].message.tool_calls
    if not calls or len(calls)!=1 or calls[0].function.name != 'operate_simulator':
        raise AgentUnavailable('Model response rejected: exactly one allowlisted tool call required')
    try: return Operations.model_validate_json(calls[0].function.arguments)
    except ValidationError: raise AgentUnavailable('Model response rejected: invalid operation arguments') from None

class OfflineToolAdapter:
    """Deterministic stand-in for the model, for demos and tests without Nebius access.
    It is NOT an LLM: a local planner (the baseline, batched) emits an OpenAI-format tool call that passes
    through the same schema validation and batch execution as live Nemotron. No network, no tokens."""
    provider='Offline mock adapter (no LLM)'
    def __init__(self):
        self.max_decisions=min(5,max(1,int(setting('MAX_DECISIONS','5'))))
    def plan(self, env):
        import copy
        from app.agents.policies import Baseline
        sim=copy.deepcopy(env); actions=[]
        while len(actions)<3 and not sim.done:
            action=ACTIONS[Baseline().choose(sim)]; actions.append(action); sim.step(ACTIONS.index(action))
            if sim.events[-1]['outcome']['status']!='ok': break
        return actions
    def decide(self, env, prior_calls):
        if prior_calls >= self.max_decisions: raise AgentUnavailable('Episode decision limit reached; episode remains unresolved')
        build_messages(env)  # same context a live request would carry; nothing is sent anywhere
        arguments=json.dumps({'actions':self.plan(env)})
        from types import SimpleNamespace as NS
        response=NS(choices=[NS(message=NS(tool_calls=[NS(function=NS(name='operate_simulator',arguments=arguments))]))])
        operations=parse_tool_call(response)
        return operations.actions, {'provider':self.provider,'model':None,'mock':True,'tokens':0,'estimated_cost_usd':0,
            'latency_seconds':0,'summary':'Mock tool call (no LLM): '+', '.join(operations.actions),'arguments':operations.model_dump()}

class Nemotron:
    def __init__(self, store, client=None):
        self.store=store
        self.model=model_name()
        self.base=base_url()
        self.provider=provider_name(self.base)
        self.client=client
        self.max_decisions=min(5,max(1,int(setting('MAX_DECISIONS','5'))))
        self.output_tokens=min(1024,max(64,int(setting('MAX_OUTPUT_TOKENS','512'))))
        self.input_rate=float(setting('INPUT_USD_PER_MILLION','0'))
        self.output_rate=float(setting('OUTPUT_USD_PER_MILLION','0'))
        self.cap=float(setting('BUDGET_USD','0'))
        self.max_requests=min(100,int(setting('MAX_REQUESTS','20')))
    def get_client(self):
        if self.client is not None: return self.client
        key=resolve_api_key()
        if not key: raise AgentUnavailable('LLM_API_KEY is not configured on the backend')
        from urllib.parse import urlparse
        url=urlparse(self.base)
        host=(url.hostname or '').lower()
        if url.scheme != 'https' or not (host.endswith('.nebius.com') or host=='openrouter.ai'):
            raise AgentUnavailable('Use an official HTTPS Nebius Token Factory or OpenRouter API endpoint')
        from openai import OpenAI
        self.client=OpenAI(api_key=key,base_url=self.base,
                           timeout=min(60,float(setting('TIMEOUT_SECONDS','25'))),max_retries=0,
                           default_headers={'X-Title':'CommitGym'} if self.provider=='OpenRouter' else None)
        return self.client
    def verify_model(self):
        try:
            models=[m.id for m in self.get_client().models.list().data]
        except AgentUnavailable: raise
        except Exception: raise AgentUnavailable(f'{self.provider} model lookup failed; check authentication, endpoint and connectivity') from None
        candidates=[m for m in models if m.lower().startswith('nvidia/') and 'nemotron' in m.lower()]
        if self.model not in candidates:
            raise AgentUnavailable('Configured NVIDIA Nemotron model is unavailable. Available NVIDIA Nemotron IDs: '+', '.join(candidates))
        self.store.save('connection',{'model':self.model,'catalog_verified':True,'inference_verified':False,'checked_at':time.time()},'nebius_connection')
        return self.model
    def decide(self, env, prior_calls):
        if prior_calls >= self.max_decisions: raise AgentUnavailable('Episode decision limit reached; episode remains unresolved')
        if not enabled(): raise AgentUnavailable('Live inference disabled; set LLM_ENABLED=true after configuring limits')
        if not free_model(self.model) and (self.input_rate <= 0 or self.output_rate <= 0 or self.cap <= 0):
            raise AgentUnavailable('Set verified model token prices and a positive spending cap before live inference')
        connection=self.store.get('nebius_connection','connection')
        if not connection or connection['model'] != self.model or time.time()-connection['checked_at'] > 3600:
            self.verify_model()
        messages=build_messages(env)
        # UTF-8 bytes conservatively upper-bound token counts; unknown outcomes retain reservation.
        input_bound=len(json.dumps({'messages':messages,'tools':[TOOL]}).encode())+512
        if input_bound > 24000: raise AgentUnavailable('Input context exceeds configured request guard')
        reserve=(input_bound*self.input_rate+self.output_tokens*self.output_rate)/1_000_000
        try: reservation=self.store.reserve(reserve,self.cap,self.max_requests)
        except ValueError as e: raise AgentUnavailable(str(e)) from None
        start=time.perf_counter()
        try:
            response=self.get_client().chat.completions.create(model=self.model,messages=messages,tools=[TOOL],
                tool_choice={'type':'function','function':{'name':'operate_simulator'}},max_tokens=self.output_tokens,temperature=0)
        except Exception:
            # No automatic retry: a timeout may already have consumed credits.
            raise AgentUnavailable(f'{self.provider} request failed or timed out; reservation retained, no automatic retry') from None
        usage=response.usage
        tokens=usage.total_tokens if usage else 0
        actual=(usage.prompt_tokens*self.input_rate+usage.completion_tokens*self.output_rate)/1_000_000 if usage else None
        self.store.settle(reservation,actual,tokens,'received')
        self.store.save('connection',{'model':self.model,'catalog_verified':True,'inference_verified':True,'checked_at':time.time()},'nebius_connection')
        operations=parse_tool_call(response)
        return operations.actions, {'provider':self.provider,'model':self.model,'tokens':tokens,
            'estimated_cost_usd':actual,'latency_seconds':time.perf_counter()-start,
            'summary':'Selected operations: '+', '.join(operations.actions),'arguments':operations.model_dump()}
