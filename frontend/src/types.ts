export type JobStatus = 'queued' | 'processing' | 'complete' | 'needs_review'

export interface Settings {
  width_mm: number
  pitch_mm: number
  total_links: number | null
  pixels_per_mm?: number
}

export interface Links {
  suggested_links: number | null
  entered_links: number | null
  effective_links: number | null
  source: 'user' | 'image' | 'unknown'
  total_length_mm: number | null
  total_length_m: number | null
  disagrees_with_image: boolean
}

export interface JobSummary {
  id: string
  status: JobStatus
  message: string
  created_at: string
  updated_at: string
  settings: Settings
  input_count: number
  used_count?: number
  full_loop_verified?: boolean
  links: Links
}

export interface Job extends JobSummary {
  input_names?: string[]
  queue_position: number | null
  result?: {
    output_size_wh: [number, number]
    input_count: number
    used_count: number
    skipped_exact_duplicates: string[]
    observed_span_pitches: number
    full_loop_verified: boolean
    verified_loop_links: number | null
    coverage_status: string
    has_deepzoom?: boolean
  }
}

export interface Sample {
  id: string
  settings: Settings
  input_count: number
  used_count: number
  output_size_wh: [number, number]
  full_loop_verified: boolean
  verified_loop_links: number | null
  has_deepzoom: boolean
}

export type JobList = { items: JobSummary[]; total: number }
export type SampleList = { items: Sample[] }
