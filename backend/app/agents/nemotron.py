"""Budgeted, allowlisted Nemotron tool calling through an isolated provider. Ordinary tests inject a fake client.

Providers are deliberately separate: OpenRouter reads only OPENROUTER_* variables and Nebius Token Factory reads only
NEBIUS_* variables, so a key for one can never be sent to the other. LLM_PROVIDER selects which one live episodes use.
"""
import json
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse
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
Draining traffic is temporary: complete is accepted only after traffic is restored to a healthy compatible release with fresh health and verification.
Never route traffic to an unhealthy or incompatible release. Obtain fresh health and verification before completion.
After a timeout inspect before retrying. After rollback recheck health then verify then complete.
Typical upgrade: stage, deploy, health_check, shift_traffic (drain), migrate, shift_traffic (restore to v2), verify, complete.
At episode start nothing is pending, so begin with stage. If deploy times out, inspect before anything else. If a fresh probe is unhealthy, inspect then rollback, then health_check, verify, complete.
You have at most 5 calls per episode: batch up to 3 consecutive operations whenever you expect each to succeed; the batch stops at the first unexpected result.
Once verification can pass, include verify and complete in the same call; an unfinished episode counts as a failure.
Do not output private reasoning; the UI shows only actions and evidence.'''

class AgentUnavailable(Exception): pass

NEBIUS_KEY_VARS=('NEBIUS_API_KEY','NEBIUS_TOKEN_FACTORY_KEY')
NEBIUS_DEFAULT_BASE='https://api.tokenfactory.nebius.com/v1'
NEBIUS_DEFAULT_MODEL='nvidia/Nemotron-3_5-Lightning'
OPENROUTER_DEFAULT_BASE='https://openrouter.ai/api/v1'
OPENROUTER_DEFAULT_MODEL='nvidia/nemotron-3.5-lightning:free'

def env(name, default=''): return os.getenv(name,'').strip() or default
def limit(name, default):
    # Shared request guards; LLM_* takes precedence, legacy NEBIUS_* names still apply.
    return env('LLM_'+name) or env('NEBIUS_'+name) or default

def resolve_nebius_key():
    """NEBIUS_API_KEY or NEBIUS_TOKEN_FACTORY_KEY; never an OpenRouter key. Values are never echoed."""
    values={k:env(k) for k in NEBIUS_KEY_VARS}
    if any(v.startswith('sk-or-') for v in values.values()):
        raise AgentUnavailable('Configuration error: a Nebius key variable contains an OpenRouter key; move it to OPENROUTER_API_KEY')
    distinct={v for v in values.values() if v}
    if len(distinct)>1:
        raise AgentUnavailable('Configuration error: NEBIUS_API_KEY and NEBIUS_TOKEN_FACTORY_KEY are set with different values; keep only one')
    return distinct.pop() if distinct else None

@dataclass
class Provider:
    id:str
    label:str
    base:str
    model:str
    key:str|None
    enabled:bool
    free:bool
    def check(self):
        """Validate configuration without network access; raises with a user-facing reason."""
        url=urlparse(self.base); host=(url.hostname or '').lower()
        if self.id=='openrouter':
            if url.scheme!='https' or host!='openrouter.ai': raise AgentUnavailable('OPENROUTER_BASE_URL must be https://openrouter.ai/api/v1')
            if not self.model.endswith(':free'): raise AgentUnavailable("Only free OpenRouter model variants (ending ':free') are allowed")
            if not self.key: raise AgentUnavailable('OPENROUTER_API_KEY is not configured on the backend')
        else:
            if url.scheme!='https' or not host.endswith('.nebius.com'): raise AgentUnavailable('Use an official HTTPS Nebius Token Factory endpoint')
            if not self.key: raise AgentUnavailable('NEBIUS_API_KEY is not configured on the backend')
            if not self.enabled: raise AgentUnavailable('Nebius inference disabled; set NEBIUS_ENABLED=true after configuring a budget')

def provider_config(selected=None):
    pid=(selected or env('LLM_PROVIDER','nebius')).lower()
    if pid=='openrouter':
        model=env('OPENROUTER_MODEL',OPENROUTER_DEFAULT_MODEL)
        return Provider('openrouter','OpenRouter',env('OPENROUTER_BASE_URL',OPENROUTER_DEFAULT_BASE),model,
                        env('OPENROUTER_API_KEY') or None,True,model.endswith(':free'))
    if pid=='nebius':
        return Provider('nebius','Nebius Token Factory',env('NEBIUS_BASE_URL',NEBIUS_DEFAULT_BASE),env('NEBIUS_MODEL',NEBIUS_DEFAULT_MODEL),
                        resolve_nebius_key(),env('NEBIUS_ENABLED','false').lower()=='true',False)
    raise AgentUnavailable('Configuration error: LLM_PROVIDER must be openrouter or nebius')

def provider_status(selected=None):
    """Configuration status for the UI; contains no secret values."""
    try:
        p=provider_config(selected)
    except AgentUnavailable as e:
        return {'id':selected or env('LLM_PROVIDER','nebius'),'label':None,'configured':False,'enabled':False,'model':None,'free_model':False,'config_error':str(e)}
    try: p.check(); error=None
    except AgentUnavailable as e: error=str(e)
    return {'id':p.id,'label':p.label,'configured':bool(p.key),'enabled':p.enabled,'model':p.model,'free_model':p.free,'config_error':error}

def build_messages(env_, calls_remaining=None):
    context={'objective':'Verified safe deployment or incident recovery', 'model_calls_remaining':calls_remaining, 'state':env_.s.public(),
             'vector_observation':env_.observation().tolist(), 'available_actions':env_.available(),
             'invariants':env_.info()['invariants'], 'previous_tool_results':env_.events[-3:]}
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(context)}]

def parse_tool_call(response):
    """The single validation boundary for model output: one allowlisted call with strict arguments.
    Valid JSON is never assumed; free endpoints do not enforce structured output."""
    choices=getattr(response,'choices',None)
    if not choices: raise AgentUnavailable('Model response rejected: no choices returned')
    message=getattr(choices[0],'message',None)
    calls=getattr(message,'tool_calls',None) if message is not None else None
    if not calls:
        if getattr(choices[0],'finish_reason',None)=='length': raise AgentUnavailable('Model response rejected: output truncated before a tool call')
        raise AgentUnavailable('Model response rejected: text or refusal instead of a tool call')
    if len(calls)!=1 or calls[0].function.name != 'operate_simulator':
        raise AgentUnavailable('Model response rejected: exactly one allowlisted tool call required')
    try: return Operations.model_validate_json(calls[0].function.arguments or '')
    except ValidationError: raise AgentUnavailable('Model response rejected: invalid operation arguments') from None

def same_model(configured, returned):
    norm=lambda m:(m or '').lower().removesuffix(':free')
    return norm(configured)==norm(returned)

def classify_error(exc):
    import openai
    if isinstance(exc,openai.APITimeoutError): return 'timeout','request timed out'
    if isinstance(exc,openai.RateLimitError): return 'rate_limited','rate limited (HTTP 429)'
    if isinstance(exc,openai.AuthenticationError): return 'auth_failed','authentication failed (HTTP 401); check the API key'
    if isinstance(exc,openai.PermissionDeniedError): return 'forbidden','access denied (HTTP 403)'
    if isinstance(exc,openai.NotFoundError): return 'not_found','model or endpoint not found (HTTP 404)'
    if isinstance(exc,openai.APIStatusError): return 'http_error',f'provider error (HTTP {exc.status_code})'
    if isinstance(exc,openai.APIConnectionError): return 'connection_error','could not connect to the provider'
    return 'error','unexpected request failure'

class OfflineToolAdapter:
    """Deterministic stand-in for the model, for demos and tests without provider access.
    It is NOT an LLM: a local planner (the baseline, batched) emits an OpenAI-format tool call that passes
    through the same schema validation and batch execution as live Nemotron. No network, no tokens."""
    provider='Offline mock adapter (no LLM)'
    last_request=None
    def __init__(self):
        self.max_decisions=min(5,max(1,int(limit('MAX_DECISIONS','5'))))
    def plan(self, env_):
        import copy
        from app.agents.policies import Baseline
        sim=copy.deepcopy(env_); actions=[]
        while len(actions)<3 and not sim.done:
            action=ACTIONS[Baseline().choose(sim)]; actions.append(action); sim.step(ACTIONS.index(action))
            if sim.events[-1]['outcome']['status']!='ok': break
        return actions
    def decide(self, env_, prior_calls):
        if prior_calls >= self.max_decisions: raise AgentUnavailable('Episode decision limit reached; episode remains unresolved')
        build_messages(env_)  # same context a live request would carry; nothing is sent anywhere
        arguments=json.dumps({'actions':self.plan(env_)})
        from types import SimpleNamespace as NS
        response=NS(choices=[NS(message=NS(tool_calls=[NS(function=NS(name='operate_simulator',arguments=arguments))]))])
        operations=parse_tool_call(response)
        return operations.actions, {'provider':self.provider,'model':None,'mock':True,'tokens':0,'estimated_cost_usd':0,
            'latency_seconds':0,'summary':'Mock tool call (no LLM): '+', '.join(operations.actions),'arguments':operations.model_dump()}

class Nemotron:
    def __init__(self, store, client=None, provider=None):
        self.store=store
        self.config=provider_config(provider)
        self.model=self.config.model
        self.provider=self.config.label
        self.client=client
        self.last_request=None
        self.max_decisions=min(5,max(1,int(limit('MAX_DECISIONS','5'))))
        self.output_tokens=min(1024,max(64,int(limit('MAX_OUTPUT_TOKENS','512'))))
        # Prices apply only to Nebius; OpenRouter is restricted to ':free' variants.
        self.input_rate=0.0 if self.config.free else float(env('NEBIUS_INPUT_USD_PER_MILLION','0'))
        self.output_rate=0.0 if self.config.free else float(env('NEBIUS_OUTPUT_USD_PER_MILLION','0'))
        self.cap=0.0 if self.config.free else float(env('NEBIUS_BUDGET_USD','0'))
        self.max_requests=min(100,int(limit('MAX_REQUESTS','20')))
        self.connection_key=f'{self.config.id}_connection'
    def get_client(self):
        if self.client is not None: return self.client
        self.config.check()
        from openai import OpenAI
        self.client=OpenAI(api_key=self.config.key,base_url=self.config.base,
                           timeout=min(60,float(limit('TIMEOUT_SECONDS','25'))),max_retries=0,
                           default_headers={'X-Title':'CommitGym'} if self.config.id=='openrouter' else None)
        return self.client
    def verify_model(self):
        try:
            models=[m.id for m in self.get_client().models.list().data]
        except AgentUnavailable: raise
        except Exception as e: raise AgentUnavailable(f'{self.provider} model lookup failed: {classify_error(e)[1]}') from None
        candidates=[m for m in models if m.lower().startswith('nvidia/') and 'nemotron' in m.lower()]
        if self.model not in candidates:
            raise AgentUnavailable('Configured NVIDIA Nemotron model is unavailable. Available NVIDIA Nemotron IDs: '+', '.join(candidates))
        self.store.save('connection',{'provider':self.config.id,'model':self.model,'catalog_verified':True,'inference_verified':False,'checked_at':time.time()},self.connection_key)
        return self.model
    def record(self, status, start, **extra):
        self.last_request={'provider':self.provider,'model':self.model,'status':status,
                           'latency_seconds':round(time.perf_counter()-start,3) if start else 0,'at':time.time(),**extra}
        return self.last_request
    def decide(self, env_, prior_calls):
        if prior_calls >= self.max_decisions: raise AgentUnavailable('Episode decision limit reached; episode remains unresolved')
        self.config.check()
        if not self.config.free and (self.input_rate <= 0 or self.output_rate <= 0 or self.cap <= 0):
            raise AgentUnavailable('Set verified model token prices and a positive spending cap before live inference')
        connection=self.store.get(self.connection_key,'connection')
        if not connection or connection['model'] != self.model or time.time()-connection['checked_at'] > 3600:
            self.verify_model()
        messages=build_messages(env_,self.max_decisions-prior_calls)
        # UTF-8 bytes conservatively upper-bound token counts; unknown outcomes retain reservation.
        input_bound=len(json.dumps({'messages':messages,'tools':[TOOL]}).encode())+512
        if input_bound > 24000: raise AgentUnavailable('Input context exceeds configured request guard')
        reserve=(input_bound*self.input_rate+self.output_tokens*self.output_rate)/1_000_000
        try: reservation=self.store.reserve(reserve,self.cap,self.max_requests)
        except ValueError as e: raise AgentUnavailable(str(e)) from None
        extra={}
        if self.config.id=='openrouter':
            # No provider fallbacks and no returned reasoning text; the request stays on the configured free model.
            # Hidden reasoning otherwise exhausts the output cap before the tool call (observed live); disable it.
            extra={'extra_body':{'provider':{'allow_fallbacks':False},'reasoning':{'enabled':False,'exclude':True}}}
        start=time.perf_counter()
        try:
            response=self.get_client().chat.completions.create(model=self.model,messages=messages,tools=[TOOL],
                tool_choice={'type':'function','function':{'name':'operate_simulator'}},max_tokens=self.output_tokens,temperature=0,**extra)
        except Exception as e:
            # No automatic retry: a timeout may already have consumed credits or rate-limit quota.
            kind,reason=classify_error(e)
            self.store.settle(reservation,None,0,kind)
            self.record(kind,start,error=reason)
            raise AgentUnavailable(f'{self.provider} request failed: {reason}; no automatic retry') from None
        usage=getattr(response,'usage',None)
        tokens=getattr(usage,'total_tokens',0) or 0
        usage_fields={'tokens':tokens,'prompt_tokens':getattr(usage,'prompt_tokens',None),'completion_tokens':getattr(usage,'completion_tokens',None)}
        actual=(usage.prompt_tokens*self.input_rate+usage.completion_tokens*self.output_rate)/1_000_000 if usage else None
        self.store.settle(reservation,actual,tokens,'received')
        returned=getattr(response,'model',None) or self.model
        if not same_model(self.model,returned):
            self.record('rejected',start,response_model=returned,error='provider returned a different model',**usage_fields)
            raise AgentUnavailable(f'Model response rejected: provider returned {returned}, not {self.model}')
        try: operations=parse_tool_call(response)
        except AgentUnavailable as e:
            self.record('rejected',start,response_model=returned,error=str(e),**usage_fields)
            raise
        # Inference counts as verified only after a valid, allowlisted tool call, not after any text reply.
        self.store.save('connection',{'provider':self.config.id,'model':self.model,'catalog_verified':True,'inference_verified':True,'checked_at':time.time()},self.connection_key)
        request=self.record('ok',start,response_model=returned,**usage_fields)
        return operations.actions, {'provider':self.provider,'model':self.model,'response_model':returned,'tokens':tokens,
            'estimated_cost_usd':actual,'latency_seconds':request['latency_seconds'],'request_status':'ok',
            'summary':'Selected operations: '+', '.join(operations.actions),'arguments':operations.model_dump()}
