import type { CrackReview, CrackTracePreview, CrackTraceRequest, HalfTurnResult, Job, JobList, Point, SampleList, ZoneAnalysis, ZoneAssistResult, ZoneGroup, ZoneInstances, ZoneShape } from './types'

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
  cracks: (id: string) => request<CrackReview>(`/api/jobs/${id}/cracks`),
  clearCracks: (id: string) => request<CrackReview>(`/api/jobs/${id}/cracks`, { method: 'DELETE' }),
  proposeCracks: (id: string, group_id: string, sensitivity: 'low' | 'normal' | 'high',
                  tear_min_length_mm: number, tear_max_length_mm: number | null,
                  chunk_min_area_mm2: number) => request<CrackReview>(`/api/jobs/${id}/cracks/propose`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_id, sensitivity, tear_min_length_mm, tear_max_length_mm, chunk_min_area_mm2 }),
  }),
  previewCrackTrace: (id: string, trace: CrackTraceRequest) => request<CrackTracePreview>(`/api/jobs/${id}/cracks/trace`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trace),
  }),
  applyCrackTrace: (id: string, trace: CrackTraceRequest) => request<CrackReview>(`/api/jobs/${id}/cracks/trace/apply`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trace),
  }),
  reviewCrack: (id: string, candidateId: string, status: 'pending' | 'accepted' | 'excluded') => request<CrackReview>(`/api/jobs/${id}/cracks/${candidateId}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }),
  }),
  reviewCracks: (id: string, candidate_ids: string[], status: 'pending' | 'accepted' | 'excluded', damage_type?: 'chunk' | 'tear', auto_larger = false) => request<CrackReview>(`/api/jobs/${id}/cracks/review-batch`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_ids, status, damage_type, auto_larger }),
  }),
  addManualCrack: (id: string, points: Point[], closed = false, candidateId: string | null = null) => request<CrackReview>(`/api/jobs/${id}/cracks/manual`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points, closed, candidate_id: candidateId }),
  }),
  deleteCrack: (id: string, candidateId: string) => request<CrackReview>(`/api/jobs/${id}/cracks/${candidateId}`, { method: 'DELETE' }),
  deleteCracks: (id: string, candidate_ids: string[]) => request<CrackReview>(`/api/jobs/${id}/cracks/delete-batch`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate_ids }),
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
