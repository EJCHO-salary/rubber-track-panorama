import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, ArrowLeft, Check, ChevronRight, CircleDot, Eraser, Eye, Layers3, MousePointer2, PenLine, RotateCcw, ScanSearch, Undo2, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, formatNumber } from '../api'
import type { DamageAnalysis, DamageCandidate, DamageMode, DamageZone, Point, ZoneSeed } from '../types'
import DamageCanvas, { type CanvasLayer } from '../components/DamageCanvas'
import styles from './DamageWorkspace.module.css'

const zoneNames: Record<DamageZone, string> = {
  tread: '트레드', groove: '골부', sprocket_hole: '스프라켓 홀', embedded_core: '심금 매설 중앙부',
}
const zoneOrder: DamageZone[] = ['tread', 'groove', 'sprocket_hole', 'embedded_core']
const zoneHelp: Record<DamageZone, string> = {
  tread: '노면과 닿는 돌출 형상', groove: '트레드 사이의 낮은 골',
  sprocket_hole: '네모난 구멍의 내부만', embedded_core: '홀을 제외한 중앙 고무',
}
const modeNames: Record<DamageMode, string> = { chunk: '청크', tear: '티어' }
type Tool = 'zone' | 'damage' | null
type CandidateFilter = 'all' | 'pending' | 'included' | 'excluded'

function prettyArea(value: number) { return `${formatNumber(value)} mm²` }

export default function DamageWorkspace() {
  const { jobId } = useParams()
  const id = jobId!
  const queryClient = useQueryClient()
  const job = useQuery({ queryKey: ['job', id], queryFn: () => api.job(id) })
  const damage = useQuery({ queryKey: ['damage', id], queryFn: () => api.damage(id), retry: false,
    enabled: job.data?.status === 'complete' })
  const [layer, setLayer] = useState<CanvasLayer>('zones')
  const [tool, setTool] = useState<Tool>(null)
  const [zone, setZone] = useState<DamageZone>('sprocket_hole')
  const [damageMode, setDamageMode] = useState<DamageMode>('chunk')
  const [draft, setDraft] = useState<Point[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [focusTarget, setFocusTarget] = useState<{ candidate: DamageCandidate; nonce: number } | null>(null)
  const [filter, setFilter] = useState<CandidateFilter>('all')
  const [zoneFilter, setZoneFilter] = useState<DamageZone | 'all'>('all')
  const [error, setError] = useState('')

  function accept(data: DamageAnalysis) {
    queryClient.setQueryData(['damage', id], data)
    setError('')
  }
  function fail(cause: unknown) { setError(cause instanceof Error ? cause.message : '변경사항을 저장하지 못했습니다.') }

  const reanalyze = useMutation({ mutationFn: (seeds: ZoneSeed[] | undefined) => api.analyzeDamage(id, seeds), onSuccess: accept, onError: fail })
  const decide = useMutation({ mutationFn: ({ candidateId, included }: { candidateId: string; included: boolean }) => api.decideDamage(id, candidateId, included), onSuccess: accept, onError: fail })
  const modes = useMutation({ mutationFn: (active: DamageMode[]) => api.damageModes(id, active), onSuccess: accept, onError: fail })
  const addCandidate = useMutation({ mutationFn: ({ mode, polygon }: { mode: DamageMode; polygon: Point[] }) => api.manualDamage(id, mode, polygon), onSuccess: data => {
    const previousIds = new Set(damage.data?.candidates.map(candidate => candidate.id) ?? [])
    accept(data)
    setSelectedId(data.candidates.find(candidate => candidate.source === 'manual' && !previousIds.has(candidate.id))?.id ?? null)
    setDraft([]); setTool(null); setLayer('candidates')
  }, onError: fail })
  const busy = reanalyze.isPending || decide.isPending || modes.isPending || addCandidate.isPending
  const analysis = damage.data
  const selected = analysis?.candidates.find(candidate => candidate.id === selectedId)
  const filtered = useMemo(() => (analysis?.candidates ?? []).filter(candidate => {
    if (zoneFilter !== 'all' && candidate.zone !== zoneFilter) return false
    if (filter === 'pending' && candidate.reviewed) return false
    if (filter === 'included' && !candidate.included) return false
    if (filter === 'excluded' && candidate.included) return false
    return true
  }), [analysis, filter, zoneFilter])

  function startTool(next: Tool) { setTool(next); setDraft([]); setLayer(next === 'zone' ? 'zones' : 'candidates'); setError('') }
  function selectCandidate(candidate: DamageCandidate, focus = false) {
    setSelectedId(candidate.id)
    setLayer('candidates')
    if (focus) setFocusTarget({ candidate, nonce: Date.now() })
  }
  function savePolygon() {
    if (!analysis || draft.length < 3 || busy) return
    if (tool === 'zone') {
      reanalyze.mutate([...analysis.zone_seeds, { zone, polygon: draft }], {
        onSuccess: () => { setDraft([]); setTool(null) },
      })
    } else if (tool === 'damage') addCandidate.mutate({ mode: damageMode, polygon: draft })
  }
  function changeMode(mode: DamageMode) {
    if (!analysis || busy) return
    const active = analysis.active_modes.includes(mode)
      ? analysis.active_modes.filter(item => item !== mode)
      : [...analysis.active_modes, mode]
    if (active.length) modes.mutate(active)
  }

  if (job.isLoading || (job.data?.status === 'complete' && damage.isLoading)) return <div className={styles.loading}>손상 검토 화면을 준비하고 있습니다…</div>
  if (!job.data) return <div className={styles.loading}>작업을 찾지 못했습니다.</div>
  if (job.data.status !== 'complete') return <div className={styles.loading}>전개 작업이 끝난 뒤 심층 분석을 시작할 수 있습니다. <Link to={`/jobs/${id}`}>작업으로 돌아가기</Link></div>
  if (!analysis) return <div className={styles.empty}>
    <div className={styles.emptyIcon}><ScanSearch size={30} /></div><span className={styles.eyebrow}>DEEP INSPECTION</span>
    <h2>이 작업의 손상 분석을 시작합니다</h2><p>전개 사진에서 반복 영역과 손상 후보를 찾습니다. 결과는 이후 직접 수정할 수 있습니다.</p>
    <button className={styles.primary} onClick={() => reanalyze.mutate(undefined)} disabled={busy}>자동 분석 실행 <ChevronRight size={17} /></button>
    {error && <p className={styles.error} role="alert">{error}</p>}
  </div>

  return <div className={styles.page}>
    <div className={styles.crumb}><Link to={`/jobs/${id}`}><ArrowLeft size={16} /> 작업 결과</Link><ChevronRight size={14} /><span>심층 분석</span></div>
    <div className={styles.heading}>
      <div><span className={styles.eyebrow}>SURFACE INTELLIGENCE / REVIEW</span><h2>손상 분석 작업대</h2>
        <p>{formatNumber(job.data.settings.width_mm)} mm 폭 · {formatNumber(job.data.settings.pitch_mm)} mm 피치 · 사람의 검토로 영역과 후보를 고칩니다.</p></div>
      <div className={styles.headingActions}><span className={styles.reviewBadge}><CircleDot size={14} /> {analysis.review_status === 'manually_reviewed' ? '후보 검토 완료' : `검토 대기 ${analysis.summary.pending_review_count}건`}</span>
        <button className={styles.subtleButton} onClick={() => reanalyze.mutate(undefined)} disabled={busy}><RotateCcw size={15} /> 자동 재분석</button></div>
    </div>
    <div className={styles.warning}><AlertCircle size={18} /><span>현재 손상률은 영상의 이상 면적 추정치입니다. 중앙 홀·그림자·골 경계의 오탐을 원본과 비교하며 검토해 주세요.</span></div>
    {error && <div className={styles.error} role="alert"><AlertCircle size={16} />{error}<button onClick={() => setError('')} aria-label="오류 닫기"><X size={15} /></button></div>}
    <div className={styles.workspace}>
      <section className={styles.mainPanel}>
        <div className={styles.canvasHeader}><div><span className={styles.eyebrow}>PANORAMA CANVAS</span><h3>전개 사진과 검토 레이어</h3></div>
          <div className={styles.layers} role="group" aria-label="표시 레이어">
            <button className={layer === 'source' ? styles.layerActive : ''} onClick={() => setLayer('source')}><Eye size={15} /> 원본</button>
            <button className={layer === 'zones' ? styles.layerActive : ''} onClick={() => setLayer('zones')}><Layers3 size={15} /> 영역</button>
            <button className={layer === 'candidates' ? styles.layerActive : ''} onClick={() => setLayer('candidates')}><ScanSearch size={15} /> 손상 후보</button>
          </div>
        </div>
        <DamageCanvas jobId={id} analysis={analysis} layer={layer} drawing={tool !== null} draft={draft} selectedId={selectedId}
          focusTarget={focusTarget} onAddPoint={point => setDraft(points => [...points, point])}
          onSelectCandidate={candidateId => setSelectedId(candidateId)} />
        <div className={styles.canvasFooter}>
          <span><i className={styles.legendTread} />트레드</span><span><i className={styles.legendGroove} />골부</span>
          <span><i className={styles.legendHole} />스프라켓 홀</span><span><i className={styles.legendCore} />심금 매설 중앙부</span>
        </div>
        <div className={styles.metrics}>
          <div className={styles.totalMetric}><span>선택된 후보의 영상 면적 비율</span><strong>{formatNumber(analysis.summary.total.damage_percent)}<small>%</small></strong>
            <p>{prettyArea(analysis.summary.total.damaged_area_mm2)} / {prettyArea(analysis.summary.total.visible_area_mm2)}</p></div>
          {zoneOrder.map(item => <div key={item} className={styles.zoneMetric}><span><i className={styles[`legend_${item}`]} />{zoneNames[item]}</span>
            <strong>{formatNumber(analysis.summary.zones[item].damage_percent)}%</strong><small>{prettyArea(analysis.summary.zones[item].damaged_area_mm2)}</small></div>)}
        </div>
      </section>
      <aside className={styles.inspector}>
        <div className={styles.inspectorTabs}><button className={tool === 'zone' || layer === 'zones' ? styles.inspectorTabActive : ''} onClick={() => { setLayer('zones'); setTool(null); setDraft([]) }}>영역 보정</button>
          <button className={tool === 'damage' || layer === 'candidates' ? styles.inspectorTabActive : ''} onClick={() => { setLayer('candidates'); setTool(null); setDraft([]) }}>손상 후보</button></div>
        {(tool === 'zone' || (tool === null && layer === 'zones')) ? <div className={styles.inspectorBody}>
          <div className={styles.sectionIntro}><span className={styles.eyebrow}>01 / ZONE CALIBRATION</span><h3>반복 영역 지정</h3>
            <p>한 피치의 실제 경계를 따라 표시하면 같은 위치의 패턴을 전체 사진에 적용합니다. 중앙 고무를 먼저 지정하고, 네모난 홀 내부를 따로 지정하세요. 홀의 측벽과 테두리는 중앙 고무에 포함합니다.</p></div>
          <div className={styles.zoneChoices}>{zoneOrder.map(item => <button key={item} className={zone === item ? styles.zoneChoiceActive : styles.zoneChoice}
            onClick={() => setZone(item)}><i className={styles[`legend_${item}`]} /><span><strong>{zoneNames[item]}</strong><small>{zoneHelp[item]}</small></span><Check size={15} /></button>)}</div>
          <div className={styles.drawBlock}><strong>{tool === 'zone' ? `${zoneNames[zone]} 다각형을 그리고 있습니다` : '영역 경계를 직접 고치기'}</strong>
            <p>{tool === 'zone' ? `${draft.length}개 점을 찍었습니다. 3개 이상이면 저장할 수 있습니다.` : '원본을 확대하고 한 피치의 경계를 따라 점을 찍으세요.'}</p>
            {tool === 'zone' ? <div className={styles.drawActions}><button onClick={() => setDraft(points => points.slice(0, -1))} disabled={!draft.length || busy}><Undo2 size={15} /> 이전 점</button>
              <button onClick={() => { setDraft([]); setTool(null) }} disabled={busy}><X size={15} /> 취소</button>
              <button className={styles.primary} onClick={savePolygon} disabled={draft.length < 3 || busy}>{busy ? '적용 중…' : '전체 피치에 적용'} <Check size={15} /></button></div>
              : <button className={styles.drawStart} onClick={() => startTool('zone')}><PenLine size={17} /> {zoneNames[zone]} 경계 그리기</button>}</div>
          <div className={styles.savedSeeds}><div className={styles.miniHead}><strong>적용한 영역 보정</strong><span>{analysis.zone_seeds.length}개</span></div>
            {analysis.zone_seeds.length ? analysis.zone_seeds.map((seed, index) => <div className={styles.seedRow} key={`${index}-${seed.zone}`}>
              <i className={styles[`legend_${seed.zone}`]} /><span>{zoneNames[seed.zone]}<small>{seed.polygon.length}개 꼭짓점 · 전체 피치 반복</small></span>
              <button aria-label={`${zoneNames[seed.zone]} 보정 삭제`} title="이 영역 보정 삭제" disabled={busy}
                onClick={() => reanalyze.mutate(analysis.zone_seeds.filter((_, i) => i !== index))}><Eraser size={16} /></button></div>)
              : <p>아직 직접 지정한 영역이 없습니다. 현재 색상은 자동 추정입니다.</p>}
          </div>
        </div> : <div className={styles.inspectorBody}>
          <div className={styles.sectionIntro}><span className={styles.eyebrow}>02 / DAMAGE REVIEW</span><h3>손상 후보 검토</h3>
            <p>붉은 윤곽은 청크, 노란 윤곽은 티어입니다. 후보를 확인하고 실제 손상만 집계에 포함하세요.</p></div>
          <div className={styles.modeControls}><span>집계 모드</span><div>{(['chunk', 'tear'] as DamageMode[]).map(item => <button key={item}
            className={analysis.active_modes.includes(item) ? styles.modeActive : ''} onClick={() => changeMode(item)} disabled={busy}>
            <Check size={14} /> {modeNames[item]}</button>)}</div></div>
          <div className={styles.candidateFilters}><select aria-label="후보 상태 필터" value={filter} onChange={event => setFilter(event.target.value as CandidateFilter)}>
            <option value="all">전체 후보</option><option value="pending">미검토</option><option value="included">집계 포함</option><option value="excluded">집계 제외</option></select>
            <select aria-label="영역 필터" value={zoneFilter} onChange={event => setZoneFilter(event.target.value as DamageZone | 'all')}>
              <option value="all">모든 영역</option>{zoneOrder.map(item => <option key={item} value={item}>{zoneNames[item]}</option>)}</select></div>
          {selected && <div className={styles.selectedCard}><div className={styles.selectedTop}><span>{modeNames[selected.mode]} · {zoneNames[selected.zone]}</span><strong>{selected.reviewed ? '검토함' : '미검토'}</strong></div>
            <div className={styles.selectedNumber}>{prettyArea(selected.area_mm2)}</div><p>길이 {formatNumber(selected.length_mm)} mm · 폭 {formatNumber(selected.width_mm)} mm</p>
            {selected.auto_reason === 'repeated_design_or_registration_pattern' && <small>같은 피치 위치에 반복돼 자동 집계에서 제외했습니다.</small>}
            <div className={styles.selectedActions}><button onClick={() => decide.mutate({ candidateId: selected.id, included: true })} disabled={busy}
              className={selected.included ? styles.decisionActive : ''}><Check size={15} /> 손상 포함</button>
              <button onClick={() => decide.mutate({ candidateId: selected.id, included: false })} disabled={busy}
                className={!selected.included ? styles.decisionActive : ''}><X size={15} /> 제외</button></div></div>}
          <div className={styles.candidateHead}><strong>후보 목록</strong><span>{filtered.length} / {analysis.candidates.length}</span></div>
          <div className={styles.candidateList}>{filtered.map((candidate, index) => <button key={candidate.id}
            className={candidate.id === selectedId ? styles.candidateRowActive : styles.candidateRow} onClick={() => selectCandidate(candidate, true)}>
            <span className={styles.candidateIndex}>{String(index + 1).padStart(2, '0')}</span><span className={styles.candidateText}><strong>{modeNames[candidate.mode]} <i>·</i> {zoneNames[candidate.zone]}</strong>
              <small>{prettyArea(candidate.area_mm2)} · {candidate.reviewed ? '검토함' : '미검토'}</small></span>
            <span className={candidate.included ? styles.includedPill : styles.excludedPill}>{candidate.included ? '포함' : '제외'}</span></button>)}
            {!filtered.length && <p className={styles.noCandidates}>조건에 맞는 후보가 없습니다.</p>}</div>
          <div className={styles.manualBlock}><strong>빠진 손상 직접 추가</strong><div><button className={damageMode === 'chunk' ? styles.modeActive : ''} onClick={() => setDamageMode('chunk')}>청크</button>
            <button className={damageMode === 'tear' ? styles.modeActive : ''} onClick={() => setDamageMode('tear')}>티어</button></div>
            {tool === 'damage' ? <><p>{draft.length}개 점 · 사진에서 손상 외곽을 찍으세요.</p><div className={styles.drawActions}>
              <button onClick={() => setDraft(points => points.slice(0, -1))} disabled={!draft.length || busy}><Undo2 size={15} /> 이전 점</button>
              <button onClick={() => { setTool(null); setDraft([]) }} disabled={busy}>취소</button>
              <button className={styles.primary} onClick={savePolygon} disabled={draft.length < 3 || busy}>후보 추가 <Check size={15} /></button></div></>
              : <button className={styles.drawStart} onClick={() => startTool('damage')}><MousePointer2 size={16} /> 손상 영역 그리기</button>}</div>
        </div>}
      </aside>
    </div>
  </div>
}
