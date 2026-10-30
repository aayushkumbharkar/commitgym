import type { Metrics, Training } from '../lib/types'
import { nice } from '../lib/api'
export function LearningCurve({data}:{data:Training['curve']}){
  if(!data.length)return <div className="empty">No training measurements yet. Run the local training command to generate a real learning curve.</div>
  const lo=Math.min(-100,...data.map(d=>d.mean_reward)),hi=Math.max(110,...data.map(d=>d.mean_reward)),max=data.at(-1)!.timesteps
  const y=(v:number)=>180-(v-lo)/(hi-lo)*145
  const points=data.map(d=>`${55+d.timesteps/max*710},${y(d.mean_reward)}`).join(' ')
  return <svg viewBox="0 0 800 230" className="chart" role="img" aria-label={`Actual PPO training rewards across ${max} timesteps. Latest mean ${data.at(-1)!.mean_reward.toFixed(1)}`}>
    {[lo,0,hi].map(v=><g key={v}><line x1="55" x2="770" y1={y(v)} y2={y(v)} stroke="#30362e" strokeDasharray="4 5"/><text x="42" y={y(v)+4} textAnchor="end">{v.toFixed(0)}</text></g>)}
    <polyline points={points} fill="none" stroke="#b9ee6a" strokeWidth="2.5"/><text x="55" y="214">0</text><text x="765" y="214" textAnchor="end">{max.toLocaleString()} timesteps</text>
  </svg>
}
export function Comparison({metrics}:{metrics:Record<string,Metrics>}){
  return <><div className="comparison-bars">{Object.entries(metrics).map(([name,m])=><div key={name} className="bar-row"><span>{nice(name)}</span><div className="bar-track"><div style={{width:`${m.safe_completion_rate*100}%`}}/></div><strong>{(m.safe_completion_rate*100).toFixed(0)}%</strong><small>n = {m.sample_count}</small></div>)}</div>
  <div className="metric-charts">{([
    ['Invariant violations','critical_invariant_violations'],['Mean reward','average_reward'],['Actions per episode','actions_per_episode'],['Recovery rate','recovery_success_rate']
  ] as const).map(([label,key])=>{const max=Math.max(1,...Object.values(metrics).map(m=>Math.abs(m[key]??0)));return <div key={key}><h3>{label}</h3>{Object.entries(metrics).map(([name,m])=><div className="mini-bar" key={name}><span>{name}</span><div><i style={{width:`${Math.abs(m[key]??0)/max*100}%`,background:(m[key]??0)<0?'#f58a76':undefined}}/></div><strong>{m[key]===null?'N/A':key==='recovery_success_rate'?`${((m[key]??0)*100).toFixed(0)}%`:(m[key]??0).toFixed(1)}</strong></div>)}</div>})}</div><div className="table-scroll"><table><thead><tr><th>Policy</th><th>Safe completion</th><th>Deployed v2</th><th>Safe rollback</th><th>Unsafe / unresolved</th><th>Violations</th><th>Mean reward</th><th>Actions / run</th><th>Recovery</th><th>Tokens</th></tr></thead><tbody>{Object.entries(metrics).map(([name,m])=><tr key={name}><td className="policy-name">{nice(name)} <small>n={m.sample_count}</small></td><td>{(100*m.safe_completion_rate).toFixed(1)}%</td><td>{(100*(m.deployment_completion_rate??0)).toFixed(0)}%</td><td>{(100*(m.rollback_completion_rate??0)).toFixed(0)}%</td><td>{m.unsafe_terminal_states} / {m.unresolved_episodes??Math.round(m.failure_rate*m.sample_count)-m.unsafe_terminal_states}</td><td>{m.critical_invariant_violations}</td><td>{m.average_reward.toFixed(1)}</td><td>{m.actions_per_episode.toFixed(1)}</td><td>{m.recovery_success_rate===null?'N/A':`${(100*m.recovery_success_rate).toFixed(1)}%`} <small>n={m.recovery_sample_count}</small></td><td>{m.tokens}</td></tr>)}</tbody></table></div></>
}
