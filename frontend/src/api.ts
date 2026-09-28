import type { DamageAnalysis, DamageMode, Job, JobList, Point, SampleList, ZoneSeed } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const data = await response.json()
  if (!response.ok) {
    const detail = data?.detail
    throw new Error(typeof detail === 'string' ? detail : '요청을 처리하지 못했습니다. 잠시 후 다시 시도하세요.')
  }
  return data as T
}

export const api = {
  jobs: () => request<JobList>('/api/jobs?limit=100'),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  samples: () => request<SampleList>('/api/samples'),
  updateLinks: (id: string, total_links: number | null) => request<Job>(`/api/jobs/${id}/links`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ total_links }),
  }),
  deleteJob: (id: string) => request<{ deleted: boolean; pending: boolean }>(`/api/jobs/${id}`, { method: 'DELETE' }),
  damage: (id: string) => request<DamageAnalysis>(`/api/jobs/${id}/damage`),
  analyzeDamage: (id: string, zone_seeds?: ZoneSeed[]) => request<DamageAnalysis>(`/api/jobs/${id}/damage`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ zone_seeds }),
  }),
  decideDamage: (id: string, candidateId: string, included: boolean) => request<DamageAnalysis>(`/api/jobs/${id}/damage/candidates/${candidateId}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ included }),
  }),
  damageModes: (id: string, active_modes: DamageMode[]) => request<DamageAnalysis>(`/api/jobs/${id}/damage/modes`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ active_modes }),
  }),
  manualDamage: (id: string, mode: DamageMode, polygon: Point[]) => request<DamageAnalysis>(`/api/jobs/${id}/damage/candidates`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode, polygon }),
  }),
}

export function fileBase(jobId: string) { return `/api/jobs/${jobId}/files` }
export function sampleBase(caseName: string) { return `/samples/${caseName}` }
export function damageFileBase(jobId: string) { return `/api/jobs/${jobId}/damage/files` }

export function formatNumber(value: number) {
  return new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(value)
}

export function formatDate(value?: string) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}
