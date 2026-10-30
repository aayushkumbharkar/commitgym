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

class Nemotron:
    def __init__(self, store, client=None):
        self.store=store
        self.model=os.getenv('NEBIUS_MODEL','nvidia/Nemotron-3_5-Lightning')
        self.base=os.getenv('NEBIUS_BASE_URL','https://api.tokenfactory.nebius.com/v1')
        self.client=client
        self.max_decisions=min(5,max(1,int(os.getenv('NEBIUS_MAX_DECISIONS','5'))))
        self.output_tokens=min(1024,max(64,int(os.getenv('NEBIUS_MAX_OUTPUT_TOKENS','512'))))
        self.input_rate=float(os.getenv('NEBIUS_INPUT_USD_PER_MILLION','0'))
        self.output_rate=float(os.getenv('NEBIUS_OUTPUT_USD_PER_MILLION','0'))
        self.cap=float(os.getenv('NEBIUS_BUDGET_USD','0'))
        self.max_requests=min(100,int(os.getenv('NEBIUS_MAX_REQUESTS','20')))
    def get_client(self):
        if self.client is not None: return self.client
        if not os.getenv('NEBIUS_API_KEY'): raise AgentUnavailable('NEBIUS_API_KEY is not configured on the backend')
        from urllib.parse import urlparse
        url=urlparse(self.base)
        if url.scheme != 'https' or not (url.hostname or '').endswith('.nebius.com'):
            raise AgentUnavailable('Use an official HTTPS Nebius API endpoint')
        from openai import OpenAI
        self.client=OpenAI(api_key=os.environ['NEBIUS_API_KEY'],base_url=self.base,
                           timeout=min(60,float(os.getenv('NEBIUS_TIMEOUT_SECONDS','25'))),max_retries=0)
        return self.client
    def verify_model(self):
        try:
            models=[m.id for m in self.get_client().models.list().data]
        except AgentUnavailable: raise
        except Exception: raise AgentUnavailable('Token Factory model lookup failed; check authentication, endpoint and connectivity') from None
        candidates=[m for m in models if m.lower().startswith('nvidia/') and 'nemotron' in m.lower()]
        if self.model not in candidates:
            raise AgentUnavailable('Configured NVIDIA Nemotron model is unavailable. Available NVIDIA Nemotron IDs: '+', '.join(candidates))
        self.store.save('connection',{'model':self.model,'catalog_verified':True,'inference_verified':False,'checked_at':time.time()},'nebius_connection')
        return self.model
    def decide(self, env, prior_calls):
        if prior_calls >= self.max_decisions: raise AgentUnavailable('Episode decision limit reached; episode remains unresolved')
        if os.getenv('NEBIUS_ENABLED','false').lower() != 'true': raise AgentUnavailable('Live inference disabled; enable it locally after configuring a budget')
        if self.input_rate <= 0 or self.output_rate <= 0 or self.cap <= 0:
            raise AgentUnavailable('Set verified model token prices and a positive spending cap before live inference')
        connection=self.store.get('nebius_connection','connection')
        if not connection or connection['model'] != self.model or time.time()-connection['checked_at'] > 3600:
            self.verify_model()
        context={'objective':'Verified safe deployment or incident recovery', 'state':env.s.public(),
                 'vector_observation':env.observation().tolist(), 'available_actions':env.available(),
                 'invariants':env.info()['invariants'], 'previous_tool_results':env.events[-3:]}
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(context)}]
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
            raise AgentUnavailable('Token Factory request failed or timed out; reservation retained, no automatic retry') from None
        usage=response.usage
        tokens=usage.total_tokens if usage else 0
        actual=(usage.prompt_tokens*self.input_rate+usage.completion_tokens*self.output_rate)/1_000_000 if usage else None
        self.store.settle(reservation,actual,tokens,'received')
        self.store.save('connection',{'model':self.model,'catalog_verified':True,'inference_verified':True,'checked_at':time.time()},'nebius_connection')
        calls=response.choices[0].message.tool_calls
        if not calls or len(calls)!=1 or calls[0].function.name != 'operate_simulator':
            raise AgentUnavailable('Model response rejected: exactly one allowlisted tool call required')
        try: operations=Operations.model_validate_json(calls[0].function.arguments)
        except ValidationError: raise AgentUnavailable('Model response rejected: invalid operation arguments') from None
        return operations.actions, {'provider':'Nebius Token Factory','model':self.model,'tokens':tokens,
            'estimated_cost_usd':actual,'latency_seconds':time.perf_counter()-start,
            'summary':'Selected operations: '+', '.join(operations.actions),'arguments':operations.model_dump()}
