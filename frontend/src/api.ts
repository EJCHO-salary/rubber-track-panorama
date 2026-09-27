import type { Job, JobList, SampleList } from './types'

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
}

export function fileBase(jobId: string) { return `/api/jobs/${jobId}/files` }
export function sampleBase(caseName: string) { return `/samples/${caseName}` }

export function formatNumber(value: number) {
  return new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(value)
}

export function formatDate(value?: string) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}
