import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Check, PenLine, RotateCcw, ScanSearch, Trash2, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, formatNumber } from '../api'
import type { CrackReview, Point } from '../types'
import ZoneCanvas from '../components/ZoneCanvas'
import styles from './CrackWorkspace.module.css'

const noop = () => {}

export default function CrackWorkspace() {
  const { jobId } = useParams(), id = jobId!
  const queryClient = useQueryClient()
  const canvasPanelRef = useRef<HTMLElement>(null)
  const zones = useQuery({ queryKey: ['zones', id], queryFn: () => api.zones(id) })
  const review = useQuery({ queryKey: ['cracks', id], queryFn: () => api.cracks(id) })
  const [groupId, setGroupId] = useState('geometry')
  const [sensitivity, setSensitivity] = useState<'low' | 'normal' | 'high'>('normal')
  const [sectionId, setSectionId] = useState('all')
  const [status, setStatus] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [focus, setFocus] = useState<{ polygon: Point[]; nonce: number } | null>(null)
  const [showZones, setShowZones] = useState(false)
  const [drawing, setDrawing] = useState(false)
  const [manualPoints, setManualPoints] = useState<Point[]>([])
  const [error, setError] = useState('')
  const proposal = useMutation({ mutationFn: () => api.proposeCracks(id, groupId, sensitivity),
    onSuccess: (data: CrackReview) => { queryClient.setQueryData(['cracks', id], data); setSelectedId(null); setError('') },
    onError: (cause: Error) => setError(cause.message) })
  const decision = useMutation({ mutationFn: ({ candidateId, next }: { candidateId: string; next: 'pending' | 'accepted' | 'excluded' }) => api.reviewCrack(id, candidateId, next),
    onSuccess: (data: CrackReview) => { queryClient.setQueryData(['cracks', id], data); setError('') },
    onError: (cause: Error) => setError(cause.message) })
  const manual = useMutation({ mutationFn: () => api.addManualCrack(id, manualPoints),
    onSuccess: (data: CrackReview) => { queryClient.setQueryData(['cracks', id], data); setDrawing(false); setManualPoints([]);
      setSelectedId(data.candidates?.[0]?.id ?? null); setError('') },
    onError: (cause: Error) => setError(cause.message) })
  const removeManual = useMutation({ mutationFn: (candidateId: string) => api.deleteManualCrack(id, candidateId),
    onSuccess: (data: CrackReview) => { queryClient.setQueryData(['cracks', id], data); setSelectedId(null); setError('') },
    onError: (cause: Error) => setError(cause.message) })
  const data = review.data
  const candidates = data?.candidates ?? []
  const selected = candidates.find(item => item.id === selectedId) ?? null
  const filtered = candidates.filter(item => (sectionId === 'all' || item.section_id === sectionId) && (status === 'all' || item.status === status))
  function choose(candidateId: string, zoom: boolean) {
    const candidate = candidates.find(item => item.id === candidateId)
    if (!candidate) return
    setSelectedId(candidateId)
    if (zoom) {
      setFocus({ polygon: candidate.polygon, nonce: Date.now() })
      requestAnimationFrame(() => canvasPanelRef.current?.scrollIntoView({ block: 'start' }))
    }
  }
  if (zones.isLoading || review.isLoading) return <div className={styles.loading}>크랙 검토 화면을 준비하고 있습니다…</div>
  if (!zones.data || !data) return <div className={styles.loading}>검토 데이터를 불러오지 못했습니다.</div>
  const currentGroup = zones.data.groups.find(item => item.id === data.group_id) ?? zones.data.groups[0]
  return <div className={styles.page}>
    <Link className={styles.back} to={`/jobs/${id}/analysis`}><ArrowLeft size={16} /> 영역 편집으로</Link>
    <header className={styles.heading}><div><span>CRACK REVIEW / FIRST PASS</span><h2>크랙 후보 검토</h2><p>선형 균열의 영상 후보를 찾아 지정된 영역별로 집계합니다. 청크·칩 분류와 깊이 판정은 아직 적용하지 않습니다.</p></div>
      <div className={styles.summary}><strong>{formatNumber(candidates.filter(item => item.source === 'automatic').length)}</strong><small>자동 후보</small></div></header>
    {error && <div className={styles.error} role="alert">{error}<button onClick={() => setError('')} aria-label="오류 닫기"><X size={15} /></button></div>}
    <div className={styles.layout}><section ref={canvasPanelRef} className={styles.canvasPanel}>
      <div className={styles.canvasHeading}><div><strong>전개 사진</strong><small>점을 누르면 후보를 선택할 수 있습니다. 휠로 확대하고 드래그로 이동하세요.</small></div><label><input type="checkbox" checked={showZones} onChange={event => setShowZones(event.target.checked)} /> 영역 색상 표시</label></div>
      <ZoneCanvas jobId={id} groupId={currentGroup?.id ?? null} analysis={zones.data} editing={drawing} points={manualPoints} closedShapes={[]} selectedVertex={null}
        opacity={showZones ? .27 : 0} focusShape={focus} suggestedPolygon={null} segmentInstances={[]} adjustingSegments={false} selectedSegment={null}
        onSegmentSelect={noop} onSegmentMove={noop} onAdd={point => setManualPoints(items => [...items, point])}
        onInsert={(index, point) => setManualPoints(items => [...items.slice(0, index), point, ...items.slice(index)])} onMoveStart={noop}
        onMove={(index, point) => setManualPoints(items => items.map((item, i) => i === index ? point : item))} onSelect={noop}
        onDelete={index => setManualPoints(items => items.filter((_, i) => i !== index))}
        onClose={noop} onClosedMove={noop} onClosedDelete={noop} crackCandidates={drawing ? [] : filtered} selectedCrackId={selectedId}
        onCrackSelect={candidateId => choose(candidateId, false)} hint={drawing ? '사진을 따라 점을 찍어 균열 경로를 그리세요 · 휠 확대 · 휠 버튼으로 이동' : undefined} />
      <div className={styles.legend}><span><i className={styles.pendingDot} /> 미검토</span><span><i className={styles.acceptedDot} /> 크랙으로 채택</span><span><i className={styles.excludedDot} /> 제외</span></div>
    </section><aside className={styles.sidePanel}>
      <div className={styles.setup}><strong>후보 탐색</strong><p>사진의 어두운 선형 특징에서 시작해 2피치 반복 형상을 배경으로 억제합니다. 결과는 검토용 초안입니다.</p>
        <label>분류 체계<select value={groupId} onChange={event => setGroupId(event.target.value)}>{zones.data.groups.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label>민감도<select value={sensitivity} onChange={event => setSensitivity(event.target.value as typeof sensitivity)}><option value="low">낮음 · 강한 후보</option><option value="normal">보통</option><option value="high">높음 · 작은 후보 포함</option></select></label>
        <button className={styles.propose} disabled={proposal.isPending || decision.isPending} onClick={() => proposal.mutate()}><ScanSearch size={16} /> {proposal.isPending ? '탐색 중…' : data.ready ? '후보 다시 찾기' : '크랙 후보 찾기'}</button>
        {data.ready && <small>다시 찾으면 현재 채택·제외 및 수동 표시 기록이 초기화됩니다.</small>}
        {data.stale && <div className={styles.stale}>영역 지도가 변경되었습니다. 후보를 다시 찾아 주세요.</div>}
        {data.ready && !data.stale && <div className={styles.manualTool}><strong>놓친 균열 직접 표시</strong>
          {drawing ? <><p>사진을 따라 2개 이상의 점을 찍으세요. 점을 끌어 경로를 조정할 수 있습니다.</p><div><button onClick={() => { setDrawing(false); setManualPoints([]) }} disabled={manual.isPending}>취소</button>
            <button onClick={() => manual.mutate()} disabled={manualPoints.length < 2 || manual.isPending}><Check size={14} /> {manual.isPending ? '저장 중…' : `경로 저장 (${manualPoints.length}점)`}</button></div></>
            : <button onClick={() => { setDrawing(true); setManualPoints([]) }}><PenLine size={15} /> 수동 균열 그리기</button>}</div>}
      </div>
      {data.ready && <><div className={styles.stats}><div><strong>{formatNumber(data.totals?.proposed ?? 0)}</strong><small>미검토</small></div><div><strong>{formatNumber(data.totals?.accepted ?? 0)}</strong><small>채택</small></div><div><strong>{formatNumber(data.totals?.excluded ?? 0)}</strong><small>제외</small></div></div>
        <div className={styles.sectionCounts}><strong>영역별 집계</strong>{data.sections?.map(section => <div key={section.id}><span><i style={{ background: section.color }} />{section.name}</span><small>후보 {data.summary?.[section.id]?.proposed ?? 0} · 채택 {data.summary?.[section.id]?.accepted ?? 0}</small></div>)}</div>
        {selected && <div className={styles.selected}><div><strong>선택한 후보</strong><span>{selected.source === 'manual' ? '수동 표시' : `영상 점수 ${selected.score.toFixed(2)}`}</span></div><p>{data.sections?.find(item => item.id === selected.section_id)?.name} · 추정 길이 {selected.length_mm} mm</p><div className={styles.decisions}>
          <button disabled={decision.isPending || data.stale} className={selected.status === 'accepted' ? styles.activeAccept : ''} onClick={() => decision.mutate({ candidateId: selected.id, next: 'accepted' })}><Check size={15} /> 채택</button>
          <button disabled={decision.isPending || data.stale} className={selected.status === 'excluded' ? styles.activeExclude : ''} onClick={() => decision.mutate({ candidateId: selected.id, next: 'excluded' })}><X size={15} /> 제외</button>
          <button disabled={decision.isPending || data.stale} onClick={() => decision.mutate({ candidateId: selected.id, next: 'pending' })}><RotateCcw size={14} /> 보류</button></div>
          {selected.source === 'manual' && <button className={styles.deleteManual} disabled={removeManual.isPending} onClick={() => removeManual.mutate(selected.id)}><Trash2 size={14} /> 이 수동 경로 삭제</button>}</div>}
        <div className={styles.candidateHeader}><strong>후보 목록</strong><small>{filtered.length}개 표시</small></div><div className={styles.filters}><select aria-label="영역 필터" value={sectionId} onChange={event => setSectionId(event.target.value)}><option value="all">전체 영역</option>{data.sections?.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select>
          <select aria-label="검토 상태 필터" value={status} onChange={event => setStatus(event.target.value)}><option value="all">전체 상태</option><option value="pending">미검토</option><option value="accepted">채택</option><option value="excluded">제외</option></select></div>
        <div className={styles.candidateList}>{filtered.map((item, index) => <button key={item.id} className={selectedId === item.id ? styles.candidateActive : styles.candidate} onClick={() => choose(item.id, true)}><i data-status={item.status} /><span>{item.source === 'manual' ? '수동 표시' : `후보 ${index+1}`}<small>{data.sections?.find(section => section.id === item.section_id)?.name} · {item.length_mm} mm</small></span><b>{item.source === 'manual' ? '직접' : item.score.toFixed(2)}</b></button>)}</div>
      </>}
    </aside></div>
  </div>
}
