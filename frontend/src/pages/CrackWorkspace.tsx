import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Check, PenLine, ScanSearch, Settings2, Trash2, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, formatNumber } from '../api'
import type { CrackCandidate, CrackReview, CrackTraceRequest, Point } from '../types'
import ZoneCanvas from '../components/ZoneCanvas'
import { toggleSelection } from '../crackSelection'
import styles from './CrackWorkspace.module.css'

const noop = () => {}
const damageTypes = [
  { id: 'chunk', label: '청크' },
  { id: 'tear', label: '티어' },
] as const
type DamageType = NonNullable<CrackCandidate['damage_type']>
type Popup = { kind: 'candidate' | 'trace'; screen: Point; candidateId: string | null }

export default function CrackWorkspace() {
  const { jobId } = useParams(), id = jobId!
  const queryClient = useQueryClient()
  const zones = useQuery({ queryKey: ['zones', id], queryFn: () => api.zones(id) })
  const review = useQuery({ queryKey: ['cracks', id], queryFn: () => api.cracks(id) })
  const [groupId, setGroupId] = useState('geometry')
  const [sensitivity, setSensitivity] = useState<'low' | 'normal' | 'high'>('normal')
  const [tearMinInput, setTearMinInput] = useState('10')
  const [tearMaxInput, setTearMaxInput] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [showZones, setShowZones] = useState(false)
  const [popup, setPopup] = useState<Popup | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [selectionBounds, setSelectionBounds] = useState<[number, number, number, number] | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [drawingCandidateId, setDrawingCandidateId] = useState<string | null>(null)
  const [manualPoints, setManualPoints] = useState<Point[]>([])
  const [manualClosed, setManualClosed] = useState(false)
  const [tracePoint, setTracePoint] = useState<Point | null>(null)
  const [traceRegion, setTraceRegion] = useState<[number, number, number, number] | null>(null)
  const [traceType, setTraceType] = useState<DamageType>('tear')
  const [traceTolerance, setTraceTolerance] = useState(30)
  const [traceOffset, setTraceOffset] = useState(.4)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const candidate = review.data?.candidates?.find(item => item.id === popup?.candidateId)
  const traceRequest: CrackTraceRequest = {
    point: tracePoint, region: traceRegion, candidate_id: popup?.kind === 'trace' ? popup.candidateId : null,
    damage_type: traceType, offset_mm: traceOffset, tolerance: traceTolerance, seed_radius_px: 24, accept: true,
  }
  const [queuedTrace, setQueuedTrace] = useState<CrackTraceRequest | null>(null)
  useEffect(() => {
    if (popup?.kind !== 'trace') { setQueuedTrace(null); return }
    const timer = window.setTimeout(() => setQueuedTrace({ ...traceRequest }), 180)
    return () => window.clearTimeout(timer)
    // Only primitive controls and point coordinates should trigger a new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [popup?.kind, popup?.candidateId, tracePoint?.[0], tracePoint?.[1],
      traceRegion?.[0], traceRegion?.[1], traceRegion?.[2], traceRegion?.[3],
      traceType, traceTolerance, traceOffset])
  const preview = useQuery({
    queryKey: ['crack-trace', id, queuedTrace],
    queryFn: () => api.previewCrackTrace(id, queuedTrace!),
    enabled: Boolean(queuedTrace && review.data?.ready && !review.data?.stale),
    retry: false,
  })
  const currentPreview = Boolean(queuedTrace && popup?.kind === 'trace' &&
    queuedTrace.candidate_id === popup.candidateId &&
    queuedTrace.point?.[0] === tracePoint?.[0] && queuedTrace.point?.[1] === tracePoint?.[1] &&
    queuedTrace.region?.join(',') === traceRegion?.join(',') &&
    queuedTrace.damage_type === traceType && queuedTrace.tolerance === traceTolerance &&
    queuedTrace.offset_mm === traceOffset)

  function save(data: CrackReview) { queryClient.setQueryData(['cracks', id], data); setError('') }
  const proposal = useMutation({
    mutationFn: (range: { min: number; max: number | null }) =>
      api.proposeCracks(id, groupId, sensitivity, range.min, range.max),
    onSuccess: data => { save(data); setPopup(null); setShowSettings(false); setNotice('후보를 다시 찾았습니다.') },
    onError: (cause: Error) => setError(cause.message),
  })
  const decision = useMutation({
    mutationFn: (next: { candidateId: string; status: 'accepted' | 'excluded' | 'pending'; type?: DamageType }) =>
      api.reviewCracks(id, [next.candidateId], next.status, next.type, false),
    onSuccess: data => save(data),
    onError: (cause: Error) => setError(cause.message),
  })
  const remove = useMutation({
    mutationFn: (candidateId: string) => api.deleteCrack(id, candidateId),
    onSuccess: data => { save(data); setPopup(null); setNotice('후보를 삭제했습니다. 다시 찾기를 눌러도 같은 위치에 재생성되지 않습니다.') },
    onError: (cause: Error) => setError(cause.message),
  })
  const removeMany = useMutation({
    mutationFn: (candidateIds: string[]) => api.deleteCracks(id, candidateIds),
    onSuccess: data => {
      save(data); setSelectedIds([]); setSelectionBounds(null)
      setNotice('선택한 후보를 삭제했습니다. 다시 찾기를 눌러도 같은 위치에 재생성되지 않습니다.')
    },
    onError: (cause: Error) => setError(cause.message),
  })
  const applyTrace = useMutation({
    mutationFn: (request: CrackTraceRequest) => api.applyCrackTrace(id, request),
    onSuccess: data => {
      save(data)
      const result = popup?.candidateId ?? data.candidates?.[0]?.id ?? null
      setPopup(result ? { kind: 'candidate', screen: popup?.screen ?? [420, 300], candidateId: result } : null)
      setNotice('선택한 균열을 저장하고 집계에 반영했습니다.')
    },
    onError: (cause: Error) => setError(cause.message),
  })
  const manual = useMutation({
    mutationFn: () => api.addManualCrack(id, manualPoints, manualClosed, drawingCandidateId),
    onSuccess: data => {
      save(data); stopDrawing(); setPopup(null)
      setNotice('그린 손상을 저장하고 선 모양과 면적에 따라 분류·집계했습니다.')
    },
    onError: (cause: Error) => setError(cause.message),
  })

  function openCandidate(candidateId: string, screen: Point) {
    setSelectedIds([]); setSelectionBounds(null)
    setDrawing(false); setTraceRegion(null); setTracePoint(null)
    setPopup({ kind: 'candidate', candidateId, screen })
  }
  function openTrace(point: Point | null, region: [number, number, number, number] | null,
                     candidateId: string | null, screen: Point) {
    setSelectedIds([]); setSelectionBounds(null)
    const item = review.data?.candidates?.find(entry => entry.id === candidateId)
    const kind = item?.damage_type ?? item?.suggested_damage_type ?? 'tear'
    setTraceType(kind)
    setTraceOffset(kind === 'tear' ? .4 : 2.5)
    setTraceTolerance(30); setTracePoint(point); setTraceRegion(region)
    setDrawing(false)
    setPopup({ kind: 'trace', candidateId, screen })
  }
  function startDrawing(candidateId: string | null = null) {
    setSelectedIds([]); setSelectionBounds(null)
    setDrawing(true); setDrawingCandidateId(candidateId); setManualPoints([]); setManualClosed(false)
    setPopup(null); setError('')
  }
  function stopDrawing() {
    setDrawing(false); setDrawingCandidateId(null); setManualPoints([]); setManualClosed(false)
    setError('')
  }
  function clearAll() {
    if (!window.confirm('이 작업의 모든 후보와 판정 기록을 삭제할까요? 원본 사진과 영역 지도는 유지됩니다.')) return
    api.clearCracks(id).then(data => { save(data); setPopup(null); setSelectedIds([]); setSelectionBounds(null); setShowSettings(false) })
      .catch((cause: Error) => setError(cause.message))
  }

  function toggleSettings() {
    if (!showSettings) {
      setGroupId(review.data?.group_id ?? 'geometry')
      setSensitivity(review.data?.sensitivity ?? 'normal')
      setTearMinInput(String(review.data?.tear_min_length_mm ?? 10))
      setTearMaxInput(review.data?.tear_max_length_mm == null ? '' : String(review.data.tear_max_length_mm))
    }
    setShowSettings(value => !value)
  }

  function findCandidates() {
    const min = Number(tearMinInput)
    const max = tearMaxInput.trim() === '' ? null : Number(tearMaxInput)
    if (tearMinInput.trim() === '' || !Number.isFinite(min) || min < 0) {
      setError('티어 길이 하한은 0 mm 이상의 숫자로 입력해 주세요.')
      return
    }
    if (max !== null && (!Number.isFinite(max) || max < min)) {
      setError('티어 길이 상한은 하한 이상으로 입력하거나 비워 주세요.')
      return
    }
    setError('')
    proposal.mutate({ min, max })
  }

  function selectCandidates(ids: string[], shift: boolean, bounds: [number, number, number, number]) {
    setPopup(null)
    setSelectedIds(current => toggleSelection(current, ids, shift))
    setSelectionBounds(bounds)
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!selectedIds.length || event.altKey || event.ctrlKey || event.metaKey ||
          (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]'))) return
      if (event.key === 'Escape') { setSelectedIds([]); setSelectionBounds(null) }
      if (event.key === 'Delete' && !removeMany.isPending) {
        event.preventDefault()
        removeMany.mutate(selectedIds)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedIds, removeMany.isPending, removeMany.mutate])

  if (zones.isLoading || review.isLoading) return <div className={styles.loading}>검토 화면을 준비하고 있습니다…</div>
  if (!zones.data || !review.data) return <div className={styles.loading}>검토 데이터를 불러오지 못했습니다.</div>
  const data = review.data, candidates = data.candidates ?? []
  const currentGroup = zones.data.groups.find(item => item.id === data.group_id) ?? zones.data.groups[0]
  const highlightedIds = selectedIds.length ? selectedIds : popup?.candidateId ? [popup.candidateId] : []
  const bulkLeft = selectionBounds ? (selectionBounds[2] + 252 < window.innerWidth
    ? selectionBounds[2] + 12 : Math.max(12, selectionBounds[0] - 252)) : 0
  const bulkTop = selectionBounds ? Math.max(12, Math.min(selectionBounds[1], window.innerHeight - 125)) : 0
  const popupLeft = popup ? Math.max(16, Math.min(popup.screen[0] + 16, window.innerWidth - 328)) : 0
  const popupHeight = popup?.kind === 'trace' ? Math.min(window.innerHeight * .78, 540) : 400
  const popupTop = popup ? Math.max(16, Math.min(popup.screen[1] + 12, window.innerHeight - popupHeight - 16)) : 0
  const busy = decision.isPending || remove.isPending || removeMany.isPending || applyTrace.isPending || proposal.isPending
  const autoCount = candidates.filter(item => item.decision_source === 'auto' && item.status === 'accepted').length
  const editedCount = candidates.filter(item => item.decision_source === 'manual' && item.status === 'accepted').length
  const drawnArea = manualClosed && manualPoints.length >= 3
    ? Math.abs(manualPoints.reduce((sum, point, index) => {
        const next = manualPoints[(index + 1) % manualPoints.length]
        return sum + point[0] * next[1] - next[0] * point[1]
      }, 0)) / 2 / zones.data.nominal_pixels_per_mm ** 2 : 0
  return <div className={styles.page}>
    <Link className={styles.back} to={'/jobs/' + id + '/analysis'}><ArrowLeft size={16} /> 영역 편집으로</Link>
    <header className={styles.heading}>
      <div><span>CRACK REVIEW / IMAGE WORKSPACE</span><h2>사진에서 바로 손상 검토</h2>
        <p>검출 결과를 자동 분류했습니다. 사진에서 유형·윤곽을 수정하거나 잘못된 후보를 삭제하세요.</p></div>
      <div className={styles.summary}><strong>{formatNumber(data.totals?.accepted ?? 0)}</strong><small>집계된 손상</small></div>
    </header>
    {error && <div className={styles.error} role="alert">{error}<button onClick={() => setError('')} aria-label="오류 닫기"><X size={15} /></button></div>}
    {notice && <div className={styles.notice} role="status">{notice}<button onClick={() => setNotice('')} aria-label="알림 닫기"><X size={15} /></button></div>}
    <div className={styles.layout}>
      <section className={styles.canvasPanel}>
        <div className={styles.canvasHeading}>
          <div><strong>전개 사진</strong><small>후보 클릭: 수정 · 드래그: 복수 선택 · Shift+클릭/드래그: 추가·해제 · Del: 삭제 · 빈 곳 클릭: 균열 추출</small></div>
          <div className={styles.imageTools}>
            <label><input type="checkbox" checked={showZones} onChange={event => setShowZones(event.target.checked)} /> 영역</label>
            {drawing && <button type="button" onClick={stopDrawing}><X size={14} /> 그리기 취소</button>}
            <button type="button" className={showSettings ? styles.toolActive : ''} onClick={toggleSettings}><Settings2 size={14} /> 자동 탐색 설정</button>
          </div>
        </div>
        <div className={styles.topLegend} aria-label="손상 유형 색상 범례">
          <strong>표시 색상</strong>
          <span><i className={styles.chunkDot} /> 청크</span>
          <span><i className={styles.acceptedDot} /> 티어</span>
        </div>
        {showSettings && <div className={styles.settingsBar}>
          <label>분류 체계<select value={groupId} onChange={event => setGroupId(event.target.value)}>{zones.data.groups.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <label>민감도<select value={sensitivity} onChange={event => setSensitivity(event.target.value as typeof sensitivity)}><option value="low">낮음</option><option value="normal">보통</option><option value="high">높음</option></select></label>
          <label>티어 하한 (mm)<input type="number" min="0" step="0.1" value={tearMinInput} onChange={event => setTearMinInput(event.target.value)} /></label>
          <label>티어 상한 (mm)<input type="number" min="0" step="0.1" placeholder="제한 없음" value={tearMaxInput} onChange={event => setTearMaxInput(event.target.value)} /></label>
          <button onClick={findCandidates} disabled={busy}><ScanSearch size={14} /> {proposal.isPending ? '탐색 중…' : '후보 다시 찾기'}</button>
          <button className={styles.clearAll} onClick={clearAll} disabled={busy}><Trash2 size={14} /> 전체 초기화</button>
        </div>}
        {data.stale && <div className={styles.stale}>영역 지도가 변경되었습니다. 자동 탐색 설정에서 후보를 다시 찾아 주세요.</div>}
        <div className={styles.viewerArea}>
          <ZoneCanvas jobId={id} groupId={currentGroup?.id ?? null} analysis={zones.data} editing={Boolean(drawing)}
            points={manualPoints} closedShapes={[]} selectedVertex={null} opacity={showZones ? .27 : 0}
            freehand={drawing} onFreehandComplete={(points, closed) => {
              setManualPoints(points); setManualClosed(closed); setError('')
            }}
            freehandConfirmation={drawing && manualPoints.length >= 2 ? {
              label: drawingCandidateId ? '윤곽 수정' : '새 손상',
              detail: manualClosed ? `${drawnArea.toFixed(1)} mm² · 청크${drawnArea < (data.chunk_min_area_mm2 ?? 100) ? ' (기준 미만 수동 지정)' : ''}`
                : '열린 선 · 티어 제안',
              disabled: (manualClosed && drawnArea < .5) || manual.isPending,
              saving: manual.isPending,
              error: error || undefined,
              onSave: () => manual.mutate(), onCancel: stopDrawing,
            } : undefined}
            focusShape={null} suggestedPolygon={null} segmentInstances={[]} adjustingSegments={false} selectedSegment={null}
            onSegmentSelect={noop} onSegmentMove={noop} onAdd={noop}
            onInsert={(index, point) => setManualPoints(items => [...items.slice(0, index), point, ...items.slice(index)])}
            onMoveStart={noop} onMove={(index, point) => setManualPoints(items => items.map((item, i) => i === index ? point : item))}
            onSelect={noop} onDelete={index => setManualPoints(items => items.filter((_, i) => i !== index))}
            onClose={noop} onClosedMove={noop} onClosedDelete={noop}
            crackCandidates={drawing ? [] : candidates} selectedCrackIds={highlightedIds}
            onCrackSelect={(candidateId, shift, screen) => {
              if (shift) selectCandidates([candidateId], true, [screen[0], screen[1], screen[0], screen[1]])
              else openCandidate(candidateId, screen)
            }}
            onCrackMarquee={drawing ? undefined : selectCandidates}
            onCrackEmpty={drawing ? undefined : (point, screen) => openTrace(point, null, null, screen)}
            tracePolygon={currentPreview && !preview.isFetching ? preview.data?.polygon : null}
            traceSeed={currentPreview && !preview.isFetching ? preview.data?.seed : null}
            hint={drawing ? '마우스로 손상을 그리세요 · 시작점 근처에서 마치면 닫힌 경계로 인식합니다'
              : '드래그 복수 선택 · Shift로 추가·해제 · Del 삭제 · 빈 사진 클릭: 균열 추출 · 휠 확대 · 휠 버튼 이동'} />
        </div>
      </section>
      <aside className={styles.sidePanel}>
        <div className={styles.sideTitle}><span>LIVE SUMMARY</span><strong>손상 집계</strong><small>자동 티어 길이 {formatNumber(data.tear_min_length_mm ?? 10)} mm 이상{data.tear_max_length_mm == null ? '' : ` · ${formatNumber(data.tear_max_length_mm)} mm 이하`}. 범위 밖 손상은 사진에서 직접 지정할 수 있습니다.</small></div>
        <div className={styles.stats}><div><strong>{formatNumber(data.totals?.accepted ?? 0)}</strong><small>집계</small></div><div><strong>{formatNumber(autoCount)}</strong><small>자동 분류</small></div><div><strong>{formatNumber(editedCount)}</strong><small>사용자 수정</small></div></div>
        <div className={styles.typeCounts}><strong>손상 유형별 집계</strong>{damageTypes.map(item => <div key={item.id}><span>{item.label}</span><b>{formatNumber(data.accepted_by_type?.[item.id] ?? 0)}건 <small>· {formatNumber(data.accepted_area_by_type?.[item.id] ?? 0)} mm²</small></b></div>)}</div>
        <div className={styles.sectionCounts}><strong>영역별 집계</strong>{data.sections?.map(section => <div key={section.id}><span><i style={{ background: section.color }} />{section.name}</span><small title="후보별 표시 면적 합계입니다. 겹친 윤곽은 중복될 수 있습니다.">{data.summary?.[section.id]?.accepted ?? 0}건 · {formatNumber(data.summary?.[section.id]?.area_mm2 ?? 0)} mm²</small></div>)}</div>
        <p className={styles.sideNote}>후보는 사진 판독을 돕는 초안입니다. 검은 홈·그림자와 실제 고무 균열은 사진을 보며 구분해 주세요.</p>
      </aside>
    </div>
    {selectedIds.length > 0 && <div className={styles.bulkPopup} style={{ left: bulkLeft, top: bulkTop }}
      role="dialog" aria-label="선택한 균열 일괄 삭제">
      <strong>{selectedIds.length}개 균열 선택</strong>
      <div>
        <button type="button" onClick={() => { setSelectedIds([]); setSelectionBounds(null) }} disabled={removeMany.isPending}>취소</button>
        <button type="button" onClick={() => removeMany.mutate(selectedIds)} disabled={removeMany.isPending}>
          <Trash2 size={14} /> {removeMany.isPending ? '삭제 중…' : '삭제 · Del'}
        </button>
      </div>
    </div>}
    {popup && <div className={styles.imagePopup} style={{ left: popupLeft, top: popupTop }} role="dialog" aria-label={popup.kind === 'candidate' ? '균열 후보 판정' : '균열 추출'}>
      <div className={styles.popupHeader}><div><span>{popup.kind === 'candidate' ? 'SELECTED REGION' : 'IMAGE SELECTION'}</span><strong>{popup.kind === 'candidate' ? '이 영역 판정' : popup.candidateId ? '균열 경계 수정' : '새 균열 추출'}</strong></div>
        <button onClick={() => setPopup(null)} aria-label="팝업 닫기"><X size={17} /></button></div>
      {popup.kind === 'candidate' && candidate && <>
        <p className={styles.popupMeta}>{data.sections?.find(item => item.id === candidate.section_id)?.name} · {candidate.area_mm2?.toFixed(1)} mm² · {candidate.status === 'pending' ? '미집계' : candidate.decision_source === 'auto' ? '자동 분류' : '사용자 수정'}</p>
        <p className={styles.popupHelp}>{candidate.suggestion_reason ?? '유형을 누르면 집계가 즉시 수정됩니다.'}</p>
        <div className={styles.popupTypes}>{damageTypes.map(type => <button key={type.id} disabled={busy}
          className={candidate.status === 'accepted' && candidate.damage_type === type.id ? styles.typeActive : ''}
          onClick={() => decision.mutate({ candidateId: candidate.id, status: 'accepted', type: type.id })}>{type.label}</button>)}</div>
        <div className={styles.popupActions}>
          <button disabled={busy} onClick={() => openTrace(null, null, candidate.id, popup.screen)}><ScanSearch size={14} /> 자동 윤곽 수정</button>
          <button disabled={busy} onClick={() => startDrawing(candidate.id)}><PenLine size={14} /> 직접 그려 수정</button>
          <button disabled={busy} onClick={() => remove.mutate(candidate.id)}><Trash2 size={14} /> 후보 삭제</button>
        </div>
      </>}
      {popup.kind === 'trace' && <>
        <p className={styles.popupHelp}>검은 영역의 자동 선택을 확인하거나, 사진 위에 손상을 직접 그리세요.</p>
        <div className={styles.traceModes}>
          <span><ScanSearch size={14} /> 자동 윤곽 선택</span>
          <button type="button" onClick={() => startDrawing(popup.candidateId)}><PenLine size={14} /> 직접 그리기</button>
        </div>
        <div className={styles.popupTypes}>{damageTypes.map(type => <button key={type.id} className={traceType === type.id ? styles.typeActive : ''} onClick={() => { setTraceType(type.id); setTraceOffset(type.id === 'tear' ? .4 : 2.5) }}>{type.label}</button>)}</div>
        <label className={styles.popupSlider}>검은색 허용 범위 <b>{traceTolerance}</b><input type="range" min="15" max="100" step="5" value={traceTolerance} onChange={event => setTraceTolerance(Number(event.target.value))} /></label>
        <label className={styles.popupSlider}>경계 여유 <b>{traceOffset.toFixed(1)} mm</b><input type="range" min="0" max="5" step="0.1" value={traceOffset} onChange={event => setTraceOffset(Number(event.target.value))} /></label>
        {(!currentPreview || preview.isFetching) && <p className={styles.popupHelp}>검은 연결 영역을 분석하고 있습니다…</p>}
        {currentPreview && preview.isError && <p className={styles.popupError}>{(preview.error as Error).message}</p>}
        {currentPreview && preview.data && !preview.isFetching && <p className={styles.popupMeta}>미리보기 {preview.data.area_mm2.toFixed(1)} mm²{preview.data.warning ? ' · 경계를 확인해 주세요' : ''}</p>}
        <button className={styles.popupPrimary} disabled={!currentPreview || !preview.data || preview.isFetching || applyTrace.isPending}
          onClick={() => applyTrace.mutate(traceRequest)}><Check size={15} /> {applyTrace.isPending ? '저장 중…' : '이 영역 채택·저장'}</button>
      </>}
    </div>}
  </div>
}
