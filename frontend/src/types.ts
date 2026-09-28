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

export type DamageZone = 'tread' | 'groove' | 'sprocket_hole' | 'embedded_core'
export type DamageMode = 'chunk' | 'tear'
export type Point = [number, number]

export interface ZoneSeed {
  zone: DamageZone
  polygon: Point[]
}

export interface DamageCandidate {
  id: string
  mode: DamageMode
  zone: DamageZone
  source: 'auto' | 'manual'
  included: boolean
  reviewed: boolean
  auto_reason: string | null
  area_mm2: number
  length_mm: number
  width_mm: number
  bbox_xywh: [number, number, number, number]
  polygon: Point[]
  zone_area_mm2: Partial<Record<DamageZone, number>>
}

export interface DamageMetric {
  visible_area_mm2: number
  damaged_area_mm2: number
  damage_percent: number
  chunk_count: number
  tear_count: number
}

export interface DamageAnalysis {
  version: number
  created_at: string
  image_size_wh: [number, number]
  nominal_pixels_per_mm: number
  pitch_px: number
  zone_seeds: ZoneSeed[]
  zone_detection: { confidence: 'low' | 'medium' | 'high'; hole_components_per_pitch: number; central_band_fraction: [number, number] }
  candidates: DamageCandidate[]
  active_modes: DamageMode[]
  summary: { zones: Record<DamageZone, DamageMetric>; total: DamageMetric; excluded_count: number; pending_review_count: number }
  review_status: 'unverified_auto_candidates' | 'manually_reviewed'
}
