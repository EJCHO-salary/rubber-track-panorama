import type { HalfTurnResult, Job, JobList, Point, SampleList, ZoneAnalysis, ZoneAssistResult, ZoneGroup, ZoneInstances, ZoneShape } from './types'

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
  zones: (id: string) => request<ZoneAnalysis>(`/api/jobs/${id}/zones`),
  zoneInstances: (id: string, groupId: string, sectionId: string) => request<ZoneInstances>(
    `/api/jobs/${id}/zones/instances?group_id=${encodeURIComponent(groupId)}&section_id=${encodeURIComponent(sectionId)}`),
  saveZones: (id: string, groups: ZoneGroup[], shapes: ZoneShape[]) => request<ZoneAnalysis>(`/api/jobs/${id}/zones`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ groups, shapes }),
  }),
  assistZone: (id: string, polygon: Point[]) => request<ZoneAssistResult>(`/api/jobs/${id}/zones/assist`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ polygon }),
  }),
  suggestHalfTurn: (id: string, polygon: Point[], repeat_pitches: number) => request<HalfTurnResult>(`/api/jobs/${id}/zones/half-turn`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ polygon, repeat_pitches }),
  }),
}

export function fileBase(jobId: string) { return `/api/jobs/${jobId}/files` }
export function sampleBase(caseName: string) { return `/samples/${caseName}` }
export function zoneFileBase(jobId: string, groupId: string) { return `/api/jobs/${jobId}/zones/groups/${groupId}` }

export function formatNumber(value: number) {
  return new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(value)
}

export function formatDate(value?: string) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}
