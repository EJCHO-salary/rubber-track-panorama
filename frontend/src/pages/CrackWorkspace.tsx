import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Check, PenLine, RotateCcw, ScanSearch, Settings2, Trash2, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, formatNumber } from '../api'
import type { CrackCandidate, CrackReview, CrackTraceRequest, Point } from '../types'
import ZoneCanvas from '../components/ZoneCanvas'
import styles from './CrackWorkspace.module.css'

const noop = () => {}
const damageTypes = [
  { id: 'chunk', label: '청크' },
  { id: 'tear', label: '티어' },
  { id: 'chip_cut', label: '칩앤컷' },
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
  const [showSettings, setShowSettings] = useState(false)
  const [showZones, setShowZones] = useState(false)
  const [popup, setPopup] = useState<Popup | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [manualPoints, setManualPoints] = useState<Point[]>([])
  const [tracePoint, setTracePoint] = useState<Point | null>(null)
  const [traceRegion, setTraceRegion] = useState<[number, number, number, number] | null>(null)
  const [traceType, setTraceType] = useState<DamageType>('tear')
  const [traceTolerance, setTraceTolerance] = useState(30)
  const [traceOffset, setTraceOffset] = useState(.4)
  const [refineSeedFor, setRefineSeedFor] = useState<string | null>(null)
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
    mutationFn: () => api.proposeCracks(id, groupId, sensitivity),
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
    mutationFn: () => api.addManualCrack(id, manualPoints),
    onSuccess: data => {
      save(data); setDrawing(false); setManualPoints([])
      setPopup({ kind: 'candidate', candidateId: data.candidates?.[0]?.id ?? null, screen: [window.innerWidth / 2, 360] })
      setNotice('수동 경로를 저장했습니다. 팝업에서 손상 유형을 지정해 주세요.')
    },
    onError: (cause: Error) => setError(cause.message),
  })

  function openCandidate(candidateId: string, screen: Point) {
    setDrawing(false); setRefineSeedFor(null); setTraceRegion(null); setTracePoint(null)
    setPopup({ kind: 'candidate', candidateId, screen })
  }
  function openTrace(point: Point | null, region: [number, number, number, number] | null,
                     candidateId: string | null, screen: Point) {
    const item = review.data?.candidates?.find(entry => entry.id === candidateId)
    const kind = item?.damage_type ?? item?.suggested_damage_type ?? 'tear'
    setTraceType(kind)
    setTraceOffset(kind === 'tear' ? .4 : kind === 'chip_cut' ? 1.5 : 2.5)
    setTraceTolerance(30); setTracePoint(point); setTraceRegion(region)
    setDrawing(false); setRefineSeedFor(null)
    setPopup({ kind: 'trace', candidateId, screen })
  }
  function clearAll() {
    if (!window.confirm('이 작업의 모든 후보와 판정 기록을 삭제할까요? 원본 사진과 영역 지도는 유지됩니다.')) return
    api.clearCracks(id).then(data => { save(data); setPopup(null); setShowSettings(false) })
      .catch((cause: Error) => setError(cause.message))
  }

  if (zones.isLoading || review.isLoading) return <div className={styles.loading}>검토 화면을 준비하고 있습니다…</div>
  if (!zones.data || !review.data) return <div className={styles.loading}>검토 데이터를 불러오지 못했습니다.</div>
  const data = review.data, candidates = data.candidates ?? []
  const currentGroup = zones.data.groups.find(item => item.id === data.group_id) ?? zones.data.groups[0]
  const selectedIds = popup?.candidateId ? [popup.candidateId] : []
  const popupLeft = popup ? Math.max(16, Math.min(popup.screen[0] + 16, window.innerWidth - 328)) : 0
  const popupTop = popup ? Math.max(16, Math.min(popup.screen[1] + 12, window.innerHeight - 470)) : 0
  const busy = decision.isPending || remove.isPending || applyTrace.isPending || proposal.isPending
  return <div className={styles.page}>
    <Link className={styles.back} to={'/jobs/' + id + '/analysis'}><ArrowLeft size={16} /> 영역 편집으로</Link>
    <header className={styles.heading}>
      <div><span>CRACK REVIEW / IMAGE WORKSPACE</span><h2>사진에서 바로 손상 검토</h2>
        <p>후보를 클릭해 판정하고, 빠진 균열은 사진에서 드래그해 추출하세요.</p></div>
      <div className={styles.summary}><strong>{formatNumber(candidates.filter(item => item.source === 'automatic').length)}</strong><small>자동 후보</small></div>
    </header>
    {error && <div className={styles.error} role="alert">{error}<button onClick={() => setError('')} aria-label="오류 닫기"><X size={15} /></button></div>}
    {notice && <div className={styles.notice} role="status">{notice}<button onClick={() => setNotice('')} aria-label="알림 닫기"><X size={15} /></button></div>}
    <div className={styles.layout}>
      <section className={styles.canvasPanel}>
        <div className={styles.canvasHeading}>
          <div><strong>전개 사진</strong><small>후보 클릭: 판정 · 빈 곳 클릭/드래그: 균열 추출 · 휠 확대 · 휠 버튼 이동</small></div>
          <div className={styles.imageTools}>
            <label><input type="checkbox" checked={showZones} onChange={event => setShowZones(event.target.checked)} /> 영역</label>
            <button type="button" className={drawing ? styles.toolActive : ''} onClick={() => { setDrawing(value => !value); setManualPoints([]); setPopup(null) }}><PenLine size={14} /> 선 그리기</button>
            <button type="button" className={showSettings ? styles.toolActive : ''} onClick={() => setShowSettings(value => !value)}><Settings2 size={14} /> 검출 설정</button>
          </div>
        </div>
        {showSettings && <div className={styles.settingsBar}>
          <label>분류 체계<select value={groupId} onChange={event => setGroupId(event.target.value)}>{zones.data.groups.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <label>민감도<select value={sensitivity} onChange={event => setSensitivity(event.target.value as typeof sensitivity)}><option value="low">낮음</option><option value="normal">보통</option><option value="high">높음</option></select></label>
          <button onClick={() => proposal.mutate()} disabled={busy}><ScanSearch size={14} /> {proposal.isPending ? '탐색 중…' : '후보 다시 찾기'}</button>
          <button className={styles.clearAll} onClick={clearAll} disabled={busy}><Trash2 size={14} /> 전체 초기화</button>
        </div>}
        {data.stale && <div className={styles.stale}>영역 지도가 변경되었습니다. 검출 설정에서 후보를 다시 찾아 주세요.</div>}
        <div className={styles.viewerArea}>
          <ZoneCanvas jobId={id} groupId={currentGroup?.id ?? null} analysis={zones.data} editing={drawing}
            points={manualPoints} closedShapes={[]} selectedVertex={null} opacity={showZones ? .27 : 0}
            focusShape={null} suggestedPolygon={null} segmentInstances={[]} adjustingSegments={false} selectedSegment={null}
            onSegmentSelect={noop} onSegmentMove={noop} onAdd={point => setManualPoints(items => [...items, point])}
            onInsert={(index, point) => setManualPoints(items => [...items.slice(0, index), point, ...items.slice(index)])}
            onMoveStart={noop} onMove={(index, point) => setManualPoints(items => items.map((item, i) => i === index ? point : item))}
            onSelect={noop} onDelete={index => setManualPoints(items => items.filter((_, i) => i !== index))}
            onClose={noop} onClosedMove={noop} onClosedDelete={noop}
            crackCandidates={drawing ? [] : candidates} selectedCrackIds={selectedIds}
            onCrackSelect={(candidateId, _shift, screen, point) => {
              if (refineSeedFor) openTrace(point, null, refineSeedFor, screen)
              else openCandidate(candidateId, screen)
            }}
            onCrackRegion={drawing ? undefined : (region, screen) => openTrace(null, region, null, screen)}
            onCrackEmpty={drawing ? undefined : (point, screen) => openTrace(point, null, refineSeedFor, screen)}
            tracePolygon={currentPreview && !preview.isFetching ? preview.data?.polygon : null}
            traceSeed={currentPreview && !preview.isFetching ? preview.data?.seed : null}
            hint={drawing ? '사진을 따라 점을 찍으세요 · 오른쪽 클릭으로 점 삭제 · 완료는 사진 아래에서'
              : refineSeedFor ? '수정할 균열의 검은 부분을 클릭하세요' : '후보 클릭 · 놓친 균열 드래그 · 휠 확대 · 휠 버튼 이동'} />
        </div>
        {drawing && <div className={styles.drawBar}><span>수동 선 {manualPoints.length}점 · 사진을 따라 점을 찍으세요.</span>
          <button onClick={() => { setDrawing(false); setManualPoints([]) }}>취소</button>
          <button disabled={manualPoints.length < 2 || manual.isPending} onClick={() => manual.mutate()}><Check size={14} /> 선 저장</button></div>}
        <div className={styles.legend}><span><i className={styles.pendingDot} /> 미검토</span><span><i className={styles.acceptedDot} /> 채택</span><span><i className={styles.excludedDot} /> 크랙 아님</span><span><i className={styles.selectionDot} /> 선택</span></div>
      </section>
      <aside className={styles.sidePanel}>
        <div className={styles.sideTitle}><span>LIVE SUMMARY</span><strong>손상 집계</strong><small>판정은 사진 위에서 바로 반영됩니다.</small></div>
        <div className={styles.stats}><div><strong>{formatNumber(data.totals?.proposed ?? 0)}</strong><small>미검토</small></div><div><strong>{formatNumber(data.totals?.accepted ?? 0)}</strong><small>채택</small></div><div><strong>{formatNumber(data.totals?.excluded ?? 0)}</strong><small>크랙 아님</small></div></div>
        <div className={styles.typeCounts}><strong>채택된 손상 유형</strong>{damageTypes.map(item => <div key={item.id}><span>{item.label}</span><b>{data.accepted_by_type?.[item.id] ?? 0}</b></div>)}</div>
        <div className={styles.sectionCounts}><strong>영역별 집계</strong>{data.sections?.map(section => <div key={section.id}><span><i style={{ background: section.color }} />{section.name}</span><small>채택 {data.summary?.[section.id]?.accepted ?? 0} · 미검토 {data.summary?.[section.id]?.proposed ?? 0}</small></div>)}</div>
        <p className={styles.sideNote}>후보는 사진 판독을 돕는 초안입니다. 검은 홈·그림자와 실제 고무 균열은 사진을 보며 구분해 주세요.</p>
      </aside>
    </div>
    {popup && <div className={styles.imagePopup} style={{ left: popupLeft, top: popupTop }} role="dialog" aria-label={popup.kind === 'candidate' ? '균열 후보 판정' : '균열 추출'}>
      <div className={styles.popupHeader}><div><span>{popup.kind === 'candidate' ? 'SELECTED REGION' : 'IMAGE SELECTION'}</span><strong>{popup.kind === 'candidate' ? '이 영역 판정' : popup.candidateId ? '균열 경계 수정' : '새 균열 추출'}</strong></div>
        <button onClick={() => { setPopup(null); setRefineSeedFor(null) }} aria-label="팝업 닫기"><X size={17} /></button></div>
      {popup.kind === 'candidate' && candidate && <>
        <p className={styles.popupMeta}>{data.sections?.find(item => item.id === candidate.section_id)?.name} · {candidate.area_mm2?.toFixed(1)} mm² · {candidate.status === 'accepted' ? '채택됨' : candidate.status === 'excluded' ? '크랙 아님' : '미검토'}</p>
        <p className={styles.popupHelp}>유형을 누르면 바로 채택되고 집계에 반영됩니다.</p>
        <div className={styles.popupTypes}>{damageTypes.map(type => <button key={type.id} disabled={busy}
          className={candidate.status === 'accepted' && candidate.damage_type === type.id ? styles.typeActive : ''}
          onClick={() => decision.mutate({ candidateId: candidate.id, status: 'accepted', type: type.id })}>{type.label}</button>)}</div>
        <div className={styles.popupActions}>
          <button disabled={busy} onClick={() => decision.mutate({ candidateId: candidate.id, status: 'excluded' })}><X size={14} /> 크랙 아님</button>
          <button disabled={busy} onClick={() => openTrace(null, null, candidate.id, popup.screen)}><ScanSearch size={14} /> 윤곽 수정</button>
          <button disabled={busy} onClick={() => { setRefineSeedFor(candidate.id); setPopup(null); setNotice('사진에서 실제 검은 균열을 클릭해 수정 시작점을 지정하세요.') }}>시작점 다시 지정</button>
          <button disabled={busy} onClick={() => remove.mutate(candidate.id)}><Trash2 size={14} /> 후보 삭제</button>
        </div>
        {candidate.status !== 'pending' && <button className={styles.popupReset} disabled={busy} onClick={() => decision.mutate({ candidateId: candidate.id, status: 'pending' })}><RotateCcw size={13} /> 미검토로 되돌리기</button>}
      </>}
      {popup.kind === 'trace' && <>
        <p className={styles.popupHelp}>파란색 미리보기를 확인하고 유형과 경계를 조정하세요.</p>
        <div className={styles.popupTypes}>{damageTypes.map(type => <button key={type.id} className={traceType === type.id ? styles.typeActive : ''} onClick={() => { setTraceType(type.id); setTraceOffset(type.id === 'tear' ? .4 : type.id === 'chip_cut' ? 1.5 : 2.5) }}>{type.label}</button>)}</div>
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
