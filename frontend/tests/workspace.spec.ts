import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('real baseline episode, partial commit, inspector and replay',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message))
  await page.goto('/')
  await expect(page.getByText('Backend connected',{exact:true})).toBeVisible()
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  for(let i=0;i<3;i++)await page.getByRole('button',{name:'Step',exact:true}).click()
  await expect(page.getByText('Deployment timed out.',{exact:false}).first()).toBeVisible()
  await page.getByRole('button',{name:'Run episode',exact:true}).click()
  await expect(page.locator('.run-metrics').getByText('Deployed v2 · verified',{exact:true})).toBeVisible()
  await expect(page.getByText('Judge accepted safe completion.',{exact:true}).first()).toBeVisible()
  await page.getByRole('button',{name:'Verify replay',exact:true}).click()
  await expect(page.getByRole('status')).toContainText('Replay verified')
  await page.getByLabel('Replay step').fill('2')
  await expect(page.getByText('STEP 3',{exact:true})).toBeVisible()
  await page.evaluate(()=>window.scrollTo(0,0))
  await page.screenshot({path:'../docs/screenshots/environment-lab.png',fullPage:true})
  expect(errors).toEqual([])
})

test('stale health and rollback execute actual transitions',async({page})=>{
  await page.goto('/')
  await page.getByLabel('FAILURE SCENARIO').selectOption('stale_health')
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  await page.getByRole('button',{name:'Run episode',exact:true}).click()
  await expect(page.locator('.run-metrics').getByText('Safe rollback · v1 restored',{exact:true})).toBeVisible()
  await expect(page.locator('.event').filter({hasText:'Cached healthy result'})).toHaveCount(1)
  await expect(page.locator('.event').filter({hasText:'Atomically restored'})).toHaveCount(1)
})

test('PPO checkpoint executes through the UI',async({page})=>{
  await page.goto('/')
  await page.getByLabel('AGENT POLICY').selectOption('ppo')
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  await page.getByRole('button',{name:'Run episode',exact:true}).click()
  // The first PPO request on a fresh backend loads torch and the checkpoint.
  await expect(page.locator('.run-metrics').getByText('Safe rollback · v1 restored',{exact:true})).toBeVisible({timeout:45000})
})

test('all navigation views and actual evaluation load',async({page})=>{
  await page.goto('/')
  for(const name of ['Overview','Agent Runs','Training','Evaluations','History','Settings']){
    await page.getByRole('navigation').getByRole('button',{name,exact:true}).click()
    await expect(page.getByRole('heading',{name,exact:true,level:1})).toBeVisible()
  }
  await page.getByRole('navigation').getByRole('button',{name:'Evaluations',exact:true}).click()
  await page.getByRole('button',{name:'Run local comparison',exact:true}).click()
  await expect(page.getByRole('status')).toContainText('15 episodes per policy')
  await expect(page.locator('tbody tr')).toHaveCount(2)
  await page.evaluate(()=>window.scrollTo(0,0))
  await page.screenshot({path:'../docs/screenshots/evaluations.png',fullPage:true})
})

test('model unavailable state is explicit',async({page})=>{
  await page.goto('/')
  await page.getByLabel('AGENT POLICY').selectOption('nemotron')
  await expect(page.getByText('Nemotron live inference is unverified.',{exact:false})).toBeVisible()
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  await expect(page.getByRole('alert')).toContainText('LLM_API_KEY is not configured')
})

test('mock tool adapter is labelled as offline and not an LLM',async({page})=>{
  await page.goto('/')
  await page.getByLabel('AGENT POLICY').selectOption('mock_tools')
  await expect(page.getByRole('note')).toContainText('not an LLM')
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  await page.getByRole('button',{name:'Run episode',exact:true}).click()
  await expect(page.locator('.run-metrics').getByText('Deployed v2 · verified',{exact:true})).toBeVisible()
  await expect(page.getByText('Mock tool call (no LLM)',{exact:false}).first()).toBeVisible()
  await expect(page.getByText('Token Factory',{exact:false})).toHaveCount(0)
})

test('offline backend displays a recovery action',async({page})=>{
  await page.route('**/api/**',route=>route.abort())
  await page.goto('/')
  await expect(page.getByText('Backend offline',{exact:true})).toBeVisible()
  await expect(page.getByRole('button',{name:'Retry connection'})).toBeVisible()
  await expect(page.getByRole('button',{name:'Start episode',exact:true})).toBeDisabled()
})

for(const width of [375,768,1440])test(`responsive workspace ${width}px and automated accessibility`,async({page})=>{
  await page.setViewportSize({width,height:1000})
  await page.goto('/')
  await page.getByRole('button',{name:'Start episode',exact:true}).click()
  await page.getByRole('button',{name:'Run episode',exact:true}).click()
  await expect(page.locator('.run-metrics').getByText('Deployed v2 · verified',{exact:true})).toBeVisible()
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy()
  const audit=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze()
  expect(audit.violations).toEqual([])
  await page.evaluate(()=>window.scrollTo(0,0))
  await page.screenshot({path:`../docs/screenshots/lab-${width}.png`,fullPage:true})
})
