from dataclasses import dataclass, asdict


@dataclass
class State:
    active_release: int = 1
    pending_release: int = 0
    schema: int = 1
    traffic_release: int = 1
    traffic: int = 100
    healthy: bool = True
    reported_healthy: bool = True
    health_fresh: bool = False
    inspected: bool = False
    uncertain: bool = False
    verified: bool = False
    rollback_available: bool = True
    recovered: bool = False
    incident: bool = False
    attempted: bool = False
    retry_eligible: bool = False
    transaction: str = 'open'
    revision: int = 0
    step: int = 0

    def public(self):
        result = asdict(self)
        # Ground truth health is available only to the independent judge.
        result.pop('healthy')
        return result


def invariants(s):
    compatible = s.traffic == 0 or s.traffic_release == s.schema
    health = s.traffic == 0 or s.traffic_release != s.active_release or s.healthy
    return [
        {'name': 'Traffic / schema compatibility', 'passed': compatible, 'critical': True},
        {'name': 'Healthy traffic target', 'passed': health, 'critical': True},
        {'name': 'Fresh health evidence', 'passed': s.health_fresh, 'critical': False},
        {'name': 'Operation outcome resolved', 'passed': not s.uncertain, 'critical': False},
        {'name': 'Independent verification', 'passed': s.verified, 'critical': False},
    ]


def judge(s):
    safe = all(x['passed'] for x in invariants(s))
    objective = (s.active_release == 2 and s.schema == 2) or (s.recovered and s.incident and s.active_release == 1 and s.schema == 1)
    return bool(safe and objective and s.traffic == 100 and s.traffic_release == s.active_release and s.attempted)
