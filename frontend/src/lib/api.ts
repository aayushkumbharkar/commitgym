const base = import.meta.env.VITE_API_URL || ''
export async function api<T>(path:string,body?:unknown):Promise<T> {
  const token=sessionStorage.getItem('commitgym-access-token')
  const response=await fetch(`${base}/api${path}`,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:`Bearer ${token}`}:{})},body:body===undefined?undefined:JSON.stringify(body)})
  if(!response.ok){const payload=await response.json().catch(()=>({detail:'Backend request failed'}));throw new Error(typeof payload.detail==='string'?payload.detail:`Request validation failed (${response.status})`)}
  return response.json()
}
export const nice=(text:string)=>text==='mock_tools'?'mock tool adapter (no LLM)':text==='ppo'?'PPO':text.replaceAll('_',' ')
// A safe rollback is not a deployment: every outcome label names which terminal state the judge accepted.
export function outcome(state:{transaction:string;recovered:boolean;active_release:number}|undefined,done=true,unsafe=false):{ok:boolean|null;label:string}{
  if(!state||(!done&&state.transaction==='open'))return {ok:null,label:'In progress'}
  if(state.transaction==='completed')return state.recovered?{ok:true,label:'Safe rollback · v1 restored'}:{ok:true,label:'Deployed v2 · verified'}
  if(unsafe)return {ok:false,label:'Unsafe outcome'}
  return state.transaction==='open'?{ok:null,label:'In progress'}:{ok:false,label:'Unresolved'}
}
