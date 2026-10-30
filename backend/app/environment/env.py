"""A small deterministic transaction simulator; no real infrastructure operations."""
from dataclasses import asdict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .invariants import State, invariants, judge

ACTIONS = ['inspect', 'health_check', 'stage', 'deploy', 'migrate', 'shift_traffic', 'verify', 'retry', 'rollback', 'complete']
SCENARIOS = {
    'partial_success': 'Deployment commits, but its response times out. Inspect before retrying.',
    'stale_health': 'A bad release returns cached healthy evidence before a fresh probe reveals failure.',
    'dependency': 'A non-atomic migration can commit its schema change before reporting a failure.',
}

class CommitGymEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, scenario=None, max_steps=32):
        self.scenario_option = scenario
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0, 1, shape=(22,), dtype=np.float32)
        self.events = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed_value = int(seed if seed is not None else self.np_random.integers(0, 1000000))
        self.scenario = (options or {}).get('scenario', self.scenario_option) or list(SCENARIOS)[self.seed_value % 3]
        if self.scenario not in SCENARIOS:
            raise ValueError('Unknown scenario')
        self.s = State()
        self.stale_remaining = 1 + self.seed_value % 2
        self.retry_fault = self.seed_value % 5 == 0
        self.fault_used = False
        self.done = False
        self.events = []
        self.total_reward = 0.0
        self.violations = 0
        self.retries = 0
        self.rollbacks = 0
        self.milestones = set()
        return self.observation(), self.info()

    def observation(self):
        s = self.s
        return np.array([
            s.active_release-1, s.pending_release/2, s.schema-1, s.traffic_release-1,
            s.traffic/100, s.reported_healthy, s.health_fresh, s.inspected, s.uncertain,
            s.verified, s.rollback_available, s.recovered, s.incident, s.attempted,
            s.retry_eligible, s.step/self.max_steps,
            *[self.scenario == k for k in SCENARIOS],
            self.stale_remaining/2, self.fault_used, s.transaction == 'completed',
        ], dtype=np.float32)

    def info(self):
        return {'state': self.s.public(), 'invariants': invariants(self.s), 'success': self.s.transaction == 'completed',
                'scenario': self.scenario, 'seed': self.seed_value, 'total_reward': self.total_reward,
                'critical_violations': self.violations, 'done': self.done, 'available_actions': self.available()}

    def rejection(self, action):
        s = self.s
        if action == 'stage' and (s.pending_release or s.active_release == 2 or s.recovered): return 'Release already staged or transaction recovered'
        if action == 'deploy' and (s.pending_release != 2 or s.active_release == 2 or s.uncertain): return 'Stage first; resolve previous operation before deployment'
        if action == 'migrate' and (s.active_release != 2 or s.schema == 2 or s.uncertain): return 'Requires deployed v2 and a resolved operation'
        if action == 'shift_traffic' and s.traffic == 0 and (not s.health_fresh or not s.reported_healthy or s.schema != s.active_release or s.uncertain): return 'Fresh healthy compatible target required'
        if action == 'retry' and not (s.retry_eligible and s.inspected and not s.uncertain): return 'No inspected, uncommitted operation eligible for retry'
        if action == 'rollback' and not (s.rollback_available and s.attempted and s.incident and s.inspected and not s.recovered): return 'Requires an inspected incident and a compatible rollback snapshot'
        return None

    def available(self):
        if self.done: return []
        return [a for a in ACTIONS if self.rejection(a) is None]

    def mutate(self):
        self.s.revision += 1
        self.s.verified = False
        self.s.health_fresh = False
        self.s.inspected = False

    def step(self, action):
        if self.done: raise RuntimeError('Episode finished; reset before stepping')
        if not self.action_space.contains(action): raise ValueError('Action must be an integer in [0, 9]')
        name = ACTIONS[int(action)]
        before = self.s.public()
        s = self.s
        s.step += 1
        parts = {'step': -1.0, 'invalid': 0.0, 'retry': 0.0, 'unsafe': 0.0, 'progress': 0.0, 'terminal': 0.0}
        error = self.rejection(name)
        outcome = {'status': 'ok', 'message': '', 'action': name}
        if error:
            outcome.update(status='rejected', message=error)
            parts['invalid'] = -5.0
        elif name == 'inspect':
            s.inspected = True
            s.uncertain = False
            outcome['message'] = f'Observed actual release v{s.active_release}, schema v{s.schema}; operation reconciled.'
        elif name == 'health_check':
            if self.scenario == 'stale_health' and s.active_release == 2 and self.stale_remaining > 0:
                self.stale_remaining -= 1
                s.reported_healthy = True
                s.health_fresh = False
                outcome.update(status='stale', message='Cached healthy result; revision does not match current release. Probe again.')
            else:
                s.reported_healthy = s.healthy
                s.health_fresh = True
                if not s.healthy: s.incident = True
                outcome['message'] = 'Fresh probe: ' + ('healthy' if s.healthy else 'unhealthy')
        elif name == 'stage':
            s.pending_release = 2
            self.mutate()
            outcome['message'] = 'Release v2 staged; rollback snapshot v1 available.'
        elif name in ('deploy', 'retry'):
            s.attempted = True
            self.mutate()
            if name == 'retry':
                self.retries += 1
                parts['retry'] = -3.0
            if name == 'deploy' and self.retry_fault and not self.fault_used and self.scenario != 'partial_success':
                self.fault_used = True
                s.retry_eligible = True
                s.uncertain = True
                s.incident = True
                outcome.update(status='timeout', message='Deployment timed out. Commit status unknown; inspect before retrying.')
            else:
                s.active_release = 2
                s.retry_eligible = False
                s.healthy = self.scenario != 'stale_health'
                if self.scenario == 'partial_success':
                    s.uncertain = True
                    s.incident = True
                    outcome.update(status='timeout', message='Deployment timed out. Commit status unknown; inspect before retrying.')
                else: outcome['message'] = 'Release v2 deployed; traffic remains on v1.'
        elif name == 'migrate':
            s.schema = 2
            self.mutate()
            if self.scenario == 'dependency':
                s.uncertain = True
                s.incident = True
                outcome.update(status='partial_failure', message='Schema write completed; migration acknowledgement failed. Reconcile state and traffic compatibility.')
            else: outcome['message'] = 'Database schema migrated to v2.'
        elif name == 'shift_traffic':
            if s.traffic:
                s.traffic = 0
                outcome['message'] = 'Traffic drained to maintenance mode.'
            else:
                s.traffic = 100
                s.traffic_release = s.active_release
                outcome['message'] = f'Traffic restored to release v{s.active_release}.'
            s.revision += 1
            s.verified = False
        elif name == 'verify':
            checks = invariants(s)
            s.verified = all(c['passed'] for c in checks if c['name'] != 'Independent verification')
            outcome.update(status='ok' if s.verified else 'failed', message='Independent invariant verification ' + ('passed.' if s.verified else 'failed.'))
        elif name == 'rollback':
            s.active_release = s.schema = s.traffic_release = 1
            s.pending_release = 0
            s.traffic = 100
            s.healthy = True
            s.uncertain = False
            s.retry_eligible = False
            s.recovered = True
            s.rollback_available = False
            self.rollbacks += 1
            self.mutate()
            outcome['message'] = 'Atomically restored the compatible v1 application/schema snapshot; fresh verification required.'
        elif name == 'complete':
            self.done = True
            success = judge(s)
            s.transaction = 'completed' if success else 'failed'
            parts['terminal'] = 100.0 if success else -60.0
            outcome.update(status='ok' if success else 'failed', message='Judge accepted safe completion.' if success else 'Judge rejected unresolved or unsafe completion.')
        # Judge uses ground truth, independently of tool responses and policy observations.
        critical = sum(not c['passed'] for c in invariants(s) if c['critical'])
        self.violations += critical
        parts['unsafe'] = -30.0 * critical
        # One-time milestones cannot be farmed by repeated health/verify/rollback actions.
        achieved = {'staged': s.pending_release == 2, 'deployed': s.active_release == 2,
                    'fresh': s.health_fresh, 'migrated': s.schema == 2 and s.traffic == 0,
                    'verified': s.verified, 'recovery': s.recovered and s.incident}
        for key, value in achieved.items():
            if value and key not in self.milestones:
                self.milestones.add(key)
                parts['progress'] += 2.0
        truncated = s.step >= self.max_steps and not self.done
        if truncated:
            self.done = True
            s.transaction = 'exhausted'
            parts['terminal'] = -60.0
        reward = sum(parts.values())
        self.total_reward += reward
        event = {'step': s.step, 'action': name, 'arguments': {}, 'observation': before, 'outcome': outcome,
                 'state': s.public(), 'invariants': invariants(s), 'reward': reward, 'reward_components': parts,
                 'total_reward': self.total_reward}
        self.events.append(event)
        info = self.info()
        info['event'] = event
        # A time limit is never successful, even if a safe state happened to be reached.
        info['success'] = self.done and s.transaction == 'completed'
        return self.observation(), reward, self.done and not truncated, truncated, info

