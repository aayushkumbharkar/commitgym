import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import type { SimState } from '../lib/types'
export function Topology({state}:{state:SimState}){
  const root=useRef<SVGSVGElement>(null)
  useEffect(()=>{
    const media=gsap.matchMedia()
    media.add('(prefers-reduced-motion: no-preference)',()=>{gsap.fromTo('.topology-node',{opacity:.5,y:4},{opacity:1,y:0,duration:.28,stagger:.04});if(state.uncertain)gsap.fromTo('.api-node',{opacity:.45},{opacity:1,repeat:2,yoyo:true,duration:.2})})
    return ()=>media.revert()
  },[state.revision,state.uncertain])
  const health=state.health_fresh?(state.reported_healthy?'Healthy':'Unhealthy'):'Unverified'
  const danger=state.health_fresh&&!state.reported_healthy
  const incompatible=state.traffic>0 && state.traffic_release!==state.schema
  return <div className="topology"><div className="topology-grid"/><svg ref={root} viewBox="0 0 900 270" role="img" aria-label={`Topology: web traffic ${state.traffic} percent to v${state.traffic_release}, API v${state.active_release} ${health}, database schema v${state.schema}`}>
    <defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#74836b"/></marker></defs>
    <path className="connection" d="M258 126H338M558 126H638" markerEnd="url(#arrow)" stroke={incompatible?'#f58a76':'#74836b'}/>
    <text x="298" y="107" textAnchor="middle" className="graph-label">{state.traffic ? `${state.traffic}%`:'DRAINED'}</text><text x="598" y="107" textAnchor="middle" className="graph-label">SQL</text>
    {[{x:40,name:'WEB FRONTEND',sub:`Traffic → v${state.traffic_release}`,foot:state.traffic?'Serving requests':'Maintenance mode',status:incompatible?'bad':'good',icon:'web'},
      {x:340,name:'API SERVICE',sub:`Active release v${state.active_release}`,foot:state.uncertain?'Commit status uncertain':health,status:danger?'bad':state.health_fresh?'good':'pending',icon:'api'},
      {x:640,name:'DATABASE',sub:`Schema v${state.schema}`,foot:incompatible?'Dependency mismatch':'Snapshot available',status:incompatible?'bad':'good',icon:'db'}].map(n=><g key={n.name} className={`topology-node ${n.icon}-node`}>
      <rect x={n.x} y="53" width="218" height="146" rx="8" fill="#191d19" stroke={n.status==='bad'?'#c36c5d':'#3b4439'}/>
      <rect x={n.x+16} y="70" width="26" height="26" rx="5" fill="#293024"/>
      <path d={`M${n.x+23} 78h12v10h-12zM${n.x+26} 91h6`} fill="none" stroke="#b9ee6a" strokeWidth="1.5"/>
      <circle cx={n.x+198} cy="82" r="4" fill={n.status==='bad'?'#f58a76':n.status==='good'?'#b9ee6a':'#ebc879'}/>
      <text x={n.x+16} y="120" className="graph-title">{n.name}</text><text x={n.x+16} y="146" className="graph-value">{n.sub}</text><text x={n.x+16} y="178" className={`graph-foot ${n.status}`}>{n.foot}</text>
    </g>)}
    <text x="450" y="240" textAnchor="middle" className="graph-label">ISOLATED SIMULATOR · REVISION {state.revision} · NO EXTERNAL INFRASTRUCTURE</text>
  </svg></div>
}
