import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from app.environment.env import CommitGymEnv, ACTIONS, SCENARIOS
from app.agents.policies import Baseline

def act(env, name): return env.step(ACTIONS.index(name))
def setup(scenario='partial_success', seed=42):
    env = CommitGymEnv(scenario)
    env.reset(seed=seed)
    return env

def test_gym_contract(): check_env(setup(), skip_render_check=True)

@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('seed', [0, 1, 42, 103])
def test_baseline_safe(scenario, seed):
    env = setup(scenario, seed)
    policy = Baseline()
    while not env.done: env.step(policy.choose(env))
    assert env.s.transaction == 'completed'
    assert env.violations == 0
    assert env.total_reward > 60


def test_determinism():
    a, b = setup(), setup()
    for action in [2,3,0,1,5,4,1,5,6,9]:
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]


def test_partial_commit_and_duplicate_rejected():
    env = setup()
    act(env, 'stage')
    _, _, _, _, info = act(env, 'deploy')
    assert info['event']['outcome']['status'] == 'timeout'
    assert env.s.active_release == 2 and env.s.uncertain
    act(env, 'inspect')
    assert act(env, 'retry')[4]['event']['outcome']['status'] == 'rejected'


def test_stale_health_is_not_evidence():
    env = setup('stale_health', 42)
    act(env, 'stage'); act(env, 'deploy')
    assert not env.s.healthy
    assert act(env, 'health_check')[4]['event']['outcome']['status'] == 'stale'
    assert env.s.reported_healthy and not env.s.health_fresh
    act(env, 'health_check')
    assert not env.s.reported_healthy and env.s.health_fresh


def test_schema_violation_reward():
    env = setup('dependency', 42)
    for name in ['stage','deploy','migrate']: result = act(env, name)
    event = result[4]['event']
    assert env.s.schema == 2 and env.s.traffic_release == 1
    assert event['outcome']['status'] == 'partial_failure'
    assert event['reward_components']['unsafe'] == -30
    assert result[1] == sum(event['reward_components'].values())


def test_completion_cannot_lie():
    env = setup()
    _, reward, terminated, _, info = act(env, 'complete')
    assert terminated and not info['success'] and reward == -61
    with pytest.raises(RuntimeError): act(env, 'inspect')


def test_invalid_action():
    env = setup()
    assert act(env, 'deploy')[4]['event']['outcome']['status'] == 'rejected'
    with pytest.raises(ValueError): env.step(100)


def test_progress_cannot_be_farmed():
    env = setup()
    act(env, 'health_check')
    for _ in range(5): assert act(env, 'health_check')[1] == -1
    assert act(env, 'rollback')[4]['event']['outcome']['status'] == 'rejected'


def test_retry_only_after_inspection():
    env = setup('dependency', 5)
    act(env, 'stage'); act(env, 'deploy')
    assert env.s.active_release == 1
    assert act(env, 'retry')[4]['event']['outcome']['status'] == 'rejected'
    act(env, 'inspect'); act(env, 'retry')
    assert env.s.active_release == 2


def test_rollback_requires_reverification():
    env = setup('stale_health')
    for name in ['stage','deploy','health_check','health_check','inspect','rollback','complete']: result = act(env,name)
    assert env.s.active_release == env.s.schema == 1
    assert not result[4]['success']

