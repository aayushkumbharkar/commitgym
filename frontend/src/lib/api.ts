const base = import.meta.env.VITE_API_URL || ''
export async function api<T>(path:string,body?:unknown):Promise<T> {
  const token=sessionStorage.getItem('commitgym-access-token')
  const response=await fetch(`${base}/api${path}`,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:`Bearer ${token}`}:{})},body:body===undefined?undefined:JSON.stringify(body)})
  if(!response.ok){const payload=await response.json().catch(()=>({detail:'Backend request failed'}));throw new Error(typeof payload.detail==='string'?payload.detail:`Request validation failed (${response.status})`)}
  return response.json()
}
export const nice=(text:string)=>text.replaceAll('_',' ')
