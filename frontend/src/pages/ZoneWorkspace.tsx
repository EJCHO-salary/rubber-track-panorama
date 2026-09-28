import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, ArrowLeft, Check, ChevronRight, Eraser, Eye, Layers3, Magnet, PenLine, Plus, Trash2, Undo2, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, formatNumber } from '../api'
import type { HalfTurnResult, Point, ZoneAnalysis, ZoneGroup, ZoneSection, ZoneShape } from '../types'
import ZoneCanvas from '../components/ZoneCanvas'
import styles from './ZoneWorkspace.module.css'

const palette = ['#6f9e62', '#d2a44d', '#a67db0', '#dc7855', '#5b9aad']
const uid = () => crypto.randomUUID().replaceAll('-', '').slice(0, 12)
const copy = (points: Point[]) => points.map(([x, y]): Point => [x, y])

export default function ZoneWorkspace() {
  const { jobId } = useParams(), id = jobId!
  const queryClient = useQueryClient()
  const job = useQuery({ queryKey: ['job', id], queryFn: () => api.job(id) })
  const zones = useQuery({ queryKey: ['zones', id], queryFn: () => api.zones(id), enabled: job.data?.status === 'complete' })
  const analysis = zones.data
  const [groupId, setGroupId] = useState('geometry')
  const [sectionId, setSectionId] = useState('groove')
  const [newGroup, setNewGroup] = useState(''), [newSection, setNewSection] = useState('')
  const [editing, setEditing] = useState(false), [editId, setEditId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Point[]>([]), [repeat, setRepeat] = useState(true)
  const [closed, setClosed] = useState<ZoneShape[]>([])
  const [repeatPitches, setRepeatPitches] = useState(1)
  const [history, setHistory] = useState<Point[][]>([]), [selectedVertex, setSelectedVertex] = useState<number | null>(null)
  const [focusShape, setFocusShape] = useState<{ polygon: Point[]; nonce: number } | null>(null)
  const [opacity, setOpacity] = useState(.85), [showOverlay, setShowOverlay] = useState(true)
  const [error, setError] = useState(''), [assistNote, setAssistNote] = useState('')
  const [halfTurnProposal, setHalfTurnProposal] = useState<{ source: ZoneShape; result: HalfTurnResult; selected: number } | null>(null)
  const [expandedSaved, setExpandedSaved] = useState<string | null>(null)
  const group = analysis?.groups.find(item => item.id === groupId) ?? analysis?.groups[0]
  const section = group?.sections.find(item => item.id === sectionId) ?? group?.sections[0]
  const groupShapes = analysis?.shapes.filter(item => item.group_id === group?.id) ?? []
  const savedSections = group?.sections.map(item => ({ section: item, shapes: groupShapes.filter(shape => shape.section_id === item.id) })).filter(item => item.shapes.length) ?? []
  const metrics = group ? analysis?.group_metrics[group.id] : undefined
  useEffect(() => {
    if (!editing) setRepeatPitches(analysis?.shapes.find(shape => shape.group_id === group?.id && shape.section_id === section?.id && shape.repeat)?.repeat_pitches ?? 1)
  }, [editing, group?.id, section?.id, analysis?.created_at, analysis?.shapes])
  function cancel() { setEditing(false); setEditId(null); setDraft([]); setClosed([]); setHistory([]); setSelectedVertex(null); setAssistNote(''); setHalfTurnProposal(null) }
  const save = useMutation({ mutationFn: ({ groups, shapes }: { groups: ZoneGroup[]; shapes: ZoneShape[] }) => api.saveZones(id, groups, shapes),
    onSuccess: (data: ZoneAnalysis) => { queryClient.setQueryData(['zones', id], data); setError(''); cancel() },
    onError: (cause: Error) => setError(cause.message) })
  const assist = useMutation({ mutationFn: (polygon: Point[]) => api.assistZone(id, polygon),
    onSuccess: result => { if (result.changed) { setHistory(items => [...items, copy(draft)]); setDraft(result.polygon); setSelectedVertex(null);
      setAssistNote('경계 제안을 확인하고 점을 조정한 뒤 저장하세요.') } else setAssistNote('안정적인 경계를 찾지 못했습니다. 그린 형상을 유지했습니다.') },
    onError: (cause: Error) => setError(cause.message) })
  const halfTurn = useMutation({ mutationFn: (shape: ZoneShape) => api.suggestHalfTurn(id, shape.polygon, shape.repeat_pitches),
    onSuccess: (result, source) => { setHalfTurnProposal({ source, result, selected: 0 });
      if (result.candidates.length) setFocusShape({ polygon: result.candidates[0].polygon, nonce: Date.now() })
      setAssistNote(result.reason ?? '주황색 대칭 후보를 사진에서 확인하고 선택하세요. 아직 저장되지 않았습니다.') },
    onError: (cause: Error) => setError(cause.message) })
  const busy = save.isPending || assist.isPending || halfTurn.isPending
  function commit(groups: ZoneGroup[], shapes = analysis?.shapes ?? []) { if (!busy) save.mutate({ groups, shapes }) }
  function updateGroup(next: ZoneGroup, shapes = analysis?.shapes ?? []) {
    if (analysis) commit(analysis.groups.map(item => item.id === next.id ? next : item), shapes)
  }
  function addGroup() {
    if (!analysis || !newGroup.trim()) return
    const id = uid(), fallback = uid()
    commit([...analysis.groups, { id, name: newGroup.trim(), default_section_id: fallback,
      sections: [{ id: fallback, name: '미지정', color: '#9ca9a2', repeat_mode: 'independent' }] }])
    setGroupId(id); setSectionId(fallback); setNewGroup('')
  }
  function removeGroup(item: ZoneGroup) {
    if (!analysis || !window.confirm(`“${item.name}” 분류 체계와 모든 형상을 삭제할까요?`)) return
    const groups = analysis.groups.filter(group => group.id !== item.id)
    commit(groups, analysis.shapes.filter(shape => shape.group_id !== item.id))
    if (groupId === item.id) { setGroupId(groups[0]?.id ?? ''); setSectionId(groups[0]?.sections[0]?.id ?? '') }
  }
  function addSection() {
    if (!group || !newSection.trim()) return
    const id = uid()
    updateGroup({ ...group, sections: [...group.sections, { id, name: newSection.trim(), color: palette[group.sections.length % palette.length], repeat_mode: 'independent' }] })
    setSectionId(id); setNewSection('')
  }
  function removeSection(item: ZoneSection) {
    if (!group || !analysis) return
    if (group.sections.length <= 1) { setError('마지막 섹션은 삭제할 수 없습니다. 분류 체계를 삭제하거나 다른 섹션을 추가하세요.'); return }
    if (!window.confirm(`“${item.name}” 섹션과 해당 형상을 삭제할까요?`)) return
    const sections = group.sections.filter(section => section.id !== item.id)
    updateGroup({ ...group, sections, default_section_id: group.default_section_id === item.id ? sections[0].id : group.default_section_id },
      analysis.shapes.filter(shape => !(shape.group_id === group.id && shape.section_id === item.id)))
    if (sectionId === item.id) setSectionId(sections[0].id)
  }
  function startEditing(shapes: ZoneShape[], targetSectionId: string) {
    const first = shapes[0]
    setEditing(true); setEditId(null); setDraft([]); setRepeat(first?.repeat ?? true)
    setRepeatPitches(first?.repeat_pitches ?? 1)
    setClosed(shapes.map(shape => ({ ...shape, polygon: copy(shape.polygon) })))
    setHistory([]); setSelectedVertex(null); setAssistNote(''); setError(''); setShowOverlay(true); setHalfTurnProposal(null)
    setSectionId(targetSectionId)
    if (first) setFocusShape({ polygon: first.polygon, nonce: Date.now() })
  }
  function begin(shape?: ZoneShape) {
    if (!group || !section) return
    startEditing(shape ? [shape] : [], shape?.section_id ?? section.id)
  }
  function beginSection(targetSectionId: string) {
    if (!group) return
    startEditing(groupShapes.filter(shape => shape.section_id === targetSectionId), targetSectionId)
  }
  function remember() { setHistory(items => [...items.slice(-29), copy(draft)]) }
  function add(point: Point) { remember(); setDraft(items => [...items, point]); setSelectedVertex(draft.length) }
  function insert(index: number, point: Point) { remember(); setDraft(items => [...items.slice(0, index), point, ...items.slice(index)]); setSelectedVertex(index) }
  function move(index: number, point: Point) { setDraft(items => items.map((item, i) => i === index ? point : item)) }
  function moveClosed(shapeId: string, index: number, point: Point) {
    setHalfTurnProposal(null)
    setClosed(items => items.map(shape => shape.id === shapeId ? { ...shape, polygon: shape.polygon.map((vertex, i) => i === index ? point : vertex) } : shape))
  }
  function deleteClosedPoint(shapeId: string, index: number) {
    setHalfTurnProposal(null)
    setClosed(items => items.map(shape => shape.id === shapeId && shape.polygon.length > 3
      ? { ...shape, polygon: shape.polygon.filter((_, i) => i !== index) } : shape))
  }
  function changeRepeatPitches(value: number) {
    setHalfTurnProposal(null)
    const pitches = Math.max(1, Math.min(2000, value || 1))
    setRepeatPitches(pitches)
    setClosed(items => items.map(shape => ({ ...shape, repeat_pitches: pitches })))
  }
  function changeRepeat(value: boolean) {
    setHalfTurnProposal(null)
    setRepeat(value)
    setClosed(items => items.map(shape => ({ ...shape, repeat: value })))
  }
  function removePoint(index: number) { remember(); setDraft(items => items.filter((_, i) => i !== index)); setSelectedVertex(null) }
  function undo() { if (history.length) { setDraft(history[history.length-1]); setHistory(items => items.slice(0, -1)); setSelectedVertex(null) } }
  function closeCurrent() {
    if (!analysis || !group || !section || draft.length < 3 || busy) return
    const xs = draft.map(point => point[0])
    const anchors = analysis.pitch_anchors_x ?? []
    const widestPitch = Math.max(analysis.pitch_px, ...anchors.slice(1).map((x, i) => x - anchors[i]))
    if (repeat && Math.max(...xs) - Math.min(...xs) > widestPitch * repeatPitches * 1.1) {
      setError('형상 너비가 반복 간격보다 큽니다. 반복 피치 수를 늘리거나 반복을 꺼 주세요.'); return
    }
    const shape: ZoneShape = { id: editId ?? uid(), group_id: group.id, section_id: section.id, polygon: copy(draft), repeat, repeat_pitches: repeatPitches }
    setClosed(items => [...items, shape])
    setHalfTurnProposal(null)
    setDraft([]); setEditId(null); setHistory([]); setSelectedVertex(null)
    setAssistNote('도형을 닫았습니다. 사진의 다음 위치를 누르면 같은 섹션의 새 도형이 시작됩니다.')
    setError('')
  }
  function reopenClosed(index: number) {
    if (draft.length) return
    const shape = closed[index]
    setHalfTurnProposal(null)
    setClosed(items => items.filter((_, i) => i !== index))
    setDraft(copy(shape.polygon)); setEditId(shape.id); setRepeat(shape.repeat); setRepeatPitches(shape.repeat_pitches)
    setHistory([]); setSelectedVertex(null); setAssistNote('첫 점을 눌러 다시 닫으세요.')
  }
  function saveClosed() {
    if (!analysis || !closed.length || draft.length || busy) return
    const replacements = new Map(closed.map(shape => [shape.id, shape]))
    const existing = analysis.shapes.map(shape => replacements.get(shape.id) ?? shape)
    const fresh = closed.filter(shape => !analysis.shapes.some(item => item.id === shape.id))
    commit(analysis.groups, [...existing, ...fresh])
  }
  function acceptHalfTurn() {
    if (!halfTurnProposal || !group || !section) return
    const candidate = halfTurnProposal.result.candidates[halfTurnProposal.selected]
    if (!candidate) return
    setClosed(items => [...items, { ...halfTurnProposal.source, id: uid(), polygon: copy(candidate.polygon) }])
    setHalfTurnProposal(null)
    setAssistNote('대칭 후보를 닫힌 도형에 추가했습니다. 점을 조정하거나 제외한 뒤 한 번에 저장하세요.')
  }
  useEffect(() => {
    if (!editing) return
    function key(event: KeyboardEvent) {
      const target = event.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedVertex !== null) { event.preventDefault(); removePoint(selectedVertex) }
      else if (event.key === 'Escape') { event.preventDefault(); cancel() }
      else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo() }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
    // Editor callbacks use current points on each render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing, selectedVertex, draft, history])
  if (job.isLoading || (job.data?.status === 'complete' && zones.isLoading)) return <div className={styles.loading}>영역 편집 화면을 준비하고 있습니다…</div>
  if (!job.data) return <div className={styles.loading}>작업을 찾지 못했습니다.</div>
  if (job.data.status !== 'complete') return <div className={styles.loading}>전개 작업이 끝난 뒤 영역을 지정할 수 있습니다. <Link to={`/jobs/${id}`}>작업으로 돌아가기</Link></div>
  if (!analysis) return <div className={styles.loading}>영역 데이터를 불러오지 못했습니다. 화면을 새로고침해 주세요.</div>
  return <div className={styles.page}>
    <div className={styles.crumb}><Link to={`/jobs/${id}`}><ArrowLeft size={16} /> 작업 결과</Link><ChevronRight size={14} /><span>영역 편집</span></div>
    <div className={styles.heading}><div><span className={styles.eyebrow}>SURFACE MAP / EDITOR</span><h2>영역 편집기</h2>
      <p>{formatNumber(job.data.settings.width_mm)} mm 폭 · {formatNumber(job.data.settings.pitch_mm)} mm 피치 · 분류 체계별로 독립적인 영역 지도를 만듭니다.</p></div>
      <span className={styles.coverageBadge}><Check size={15} /> {analysis.groups.length}개 분류 체계</span></div>
    <div className={styles.warning}><AlertCircle size={18} /><span>형상·손상 자동 판정은 없습니다. 직접 그리거나 경계 보조 선택을 사용하세요. 각 체계의 남는 부분은 지정한 기본 섹션으로 채웁니다.</span></div>
    {error && <div className={styles.error} role="alert"><AlertCircle size={16} />{error}<button onClick={() => setError('')} aria-label="오류 닫기"><X size={15} /></button></div>}
    <div className={styles.groupBar}><div className={styles.barTitle}><span className={styles.eyebrow}>CLASSIFICATION LAYERS</span><strong>분류 체계</strong><small>체계 간 영역은 겹쳐도 됩니다.</small></div>
      <div className={styles.groupTabs}>{analysis.groups.map(item => <button key={item.id} className={item.id === group?.id ? styles.groupTabActive : styles.groupTab} onClick={() => { cancel(); setGroupId(item.id); setSectionId(item.sections[0].id) }} disabled={busy || editing}>{item.name}</button>)}</div>
      <form className={styles.inlineAdd} onSubmit={event => { event.preventDefault(); addGroup() }}><input aria-label="새 분류 체계 이름" placeholder="새 분류 체계" value={newGroup} onChange={event => setNewGroup(event.target.value)} maxLength={80} disabled={editing} /><button type="submit" disabled={!newGroup.trim() || busy || editing}><Plus size={15} /> 추가</button></form></div>
    <div className={styles.workspace}>
      <section className={styles.mainPanel}>
        <div className={styles.canvasHeader}><div><span className={styles.eyebrow}>PANORAMA CANVAS</span><h3>{group ? `${group.name} · 전개 사진 위에 그리기` : '전개 사진'}</h3></div>
          <div className={styles.layers} role="group" aria-label="표시 레이어"><button className={!showOverlay ? styles.layerActive : ''} onClick={() => setShowOverlay(false)}><Eye size={15} /> 원본</button><button className={showOverlay ? styles.layerActive : ''} onClick={() => setShowOverlay(true)}><Layers3 size={15} /> 영역</button></div></div>
        <ZoneCanvas jobId={id} groupId={group?.id ?? null} analysis={analysis} editing={editing} points={draft} closedShapes={closed} selectedVertex={selectedVertex}
          opacity={showOverlay ? opacity : 0} focusShape={focusShape} suggestedPolygon={halfTurnProposal?.result.candidates[halfTurnProposal.selected]?.polygon ?? null} onAdd={add} onInsert={insert}
          onMoveStart={remember} onMove={move} onSelect={setSelectedVertex} onDelete={removePoint} onClose={closeCurrent}
          onClosedMove={moveClosed} onClosedDelete={deleteClosedPoint} />
        <div className={styles.canvasFooter}>{group?.sections.map(item => <span key={item.id}><i style={{ background: item.color }} />{item.name}</span>)}
          <label className={styles.opacityControl}>색 농도 <input type="range" min="0" max="100" value={Math.round(opacity*100)} onChange={event => setOpacity(Number(event.target.value)/100)} /></label></div>
        {group && <div className={styles.metricGrid}><div><small>전체 픽셀 채움</small><strong>{metrics?.coverage_percent ?? 100}%</strong><small>남는 영역 → {group.sections.find(item => item.id === group.default_section_id)?.name}</small></div>
          {group.sections.map(item => <div key={item.id}><small><i style={{ background: item.color }} />{item.name}</small><strong>{formatNumber(metrics?.areas_mm2[item.id] ?? 0)} <em>mm²</em></strong>
            <small>직접 지정 {formatNumber(metrics?.explicit_areas_mm2?.[item.id] ?? 0)} mm² · 형상 {groupShapes.filter(shape => shape.section_id === item.id).length}개</small></div>)}</div>}
      </section>
      <aside className={styles.inspector}><div className={styles.inspectorBody}>
        {group ? <>
          <div className={styles.sectionIntro}><span className={styles.eyebrow}>01 / CLASSIFICATION</span><h3>분류 체계와 섹션</h3><p>이름은 제품과 분석 목적에 맞게 바꿀 수 있습니다. 제조공법 구분처럼 겹치는 지도는 새 분류 체계로 추가하세요.</p></div>
          <div className={styles.editRow}><input key={group.id} aria-label="분류 체계 이름 수정" defaultValue={group.name} maxLength={80} onBlur={event => { if (event.target.value.trim() && event.target.value.trim() !== group.name) updateGroup({ ...group, name: event.target.value.trim() }) }} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} disabled={busy || editing} />
            <button aria-label="분류 체계 삭제" title="분류 체계 삭제" onClick={() => removeGroup(group)} disabled={busy || editing}><Trash2 size={15} /></button></div>
          <label className={styles.fallbackSelect}>빈 영역에 적용할 섹션<select value={group.default_section_id} onChange={event => updateGroup({ ...group, default_section_id: event.target.value })} disabled={busy || editing}>
            {group.sections.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          {section && <label className={styles.fallbackSelect}>“{section.name}” 반복 방식<select aria-label="반복 방식" value={section.repeat_mode ?? 'independent'}
            onChange={event => updateGroup({ ...group, sections: group.sections.map(item => item.id === section.id ? { ...item, repeat_mode: event.target.value as ZoneSection['repeat_mode'] } : item) })} disabled={busy || editing}>
            <option value="independent">각 도형을 지정한 간격으로 반복</option><option value="examples">여러 도형 중 한 예시만 선택</option></select>
            <small>긴 골부와 짧은 골부처럼 모두 보여야 할 형상은 각각 반복합니다. 대체 예시를 그린 경우에만 한 예시 선택을 사용하세요.</small></label>}
          <div className={styles.sectionList}>{group.sections.map(item => <div key={item.id} className={styles.sectionRow}>
            <button className={item.id === section?.id ? styles.sectionSelectActive : styles.sectionSelect} onClick={() => { setSectionId(item.id); cancel() }} disabled={busy || editing} title="그릴 섹션 선택"><i style={{ background: item.color }} /><Check size={13} /></button>
            <input key={`${group.id}-${item.id}`} aria-label={`${item.name} 이름 수정`} defaultValue={item.name} maxLength={80} onBlur={event => { if (event.target.value.trim() && event.target.value.trim() !== item.name) updateGroup({ ...group, sections: group.sections.map(section => section.id === item.id ? { ...section, name: event.target.value.trim() } : section) }) }} onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur() }} disabled={busy || editing} />
            <input type="color" aria-label={`${item.name} 색상`} value={item.color} onChange={event => updateGroup({ ...group, sections: group.sections.map(section => section.id === item.id ? { ...section, color: event.target.value } : section) })} disabled={busy || editing} />
            <button aria-label={`${item.name} 삭제`} onClick={() => removeSection(item)} disabled={busy || editing}><Trash2 size={14} /></button></div>)}</div>
          <form className={styles.inlineAdd} onSubmit={event => { event.preventDefault(); addSection() }}><input aria-label="새 섹션 이름" placeholder="새 세부 섹션" value={newSection} onChange={event => setNewSection(event.target.value)} maxLength={80} disabled={editing} /><button type="submit" disabled={!newSection.trim() || busy || editing}><Plus size={15} /> 추가</button></form>
          <div className={styles.drawBlock}><strong>{editing ? `${section?.name} 도형 그리기` : '형상 그리기'}</strong>
            <p>{analysis.pitch_anchor_source === 'image' ? `사진에서 피치 기준점 ${analysis.pitch_anchors_x?.length ?? 0}개를 측정해 반복 위치를 보정합니다.` : '사진에서 피치 기준점을 안정적으로 찾지 못해 입력한 피치의 명목 간격을 사용합니다.'}</p>
            <p>{editing ? `그리는 점 ${draft.length}개 · 닫힌 도형 ${closed.length}개. 첫 점을 다시 눌러 닫으면 다음 클릭부터 새 도형을 그립니다.` : '사진에서 점을 찍고 첫 점을 다시 눌러 도형을 닫으세요. 다음 클릭은 같은 섹션의 새 도형을 시작합니다.'}</p>
            {editing ? <><div className={styles.repeatSettings}><label className={styles.repeatToggle}><input type="checkbox" checked={repeat} onChange={event => changeRepeat(event.target.checked)} /> 피치마다 반복 적용</label>
              {repeat && <label className={styles.repeatInterval}>반복 간격 <input type="number" min="1" max="2000" step="1" value={repeatPitches} onChange={event => changeRepeatPitches(Number(event.target.value))} /> 피치마다</label>}</div>
              <div className={styles.drawActions}><button onClick={undo} disabled={!history.length || busy}><Undo2 size={15} /> 되돌리기</button><button onClick={() => selectedVertex !== null && removePoint(selectedVertex)} disabled={selectedVertex === null || busy}><Eraser size={15} /> 점 삭제</button><button onClick={cancel} disabled={busy}><X size={15} /> 취소</button></div>
              <div className={styles.drawActions}><button onClick={() => assist.mutate(draft)} disabled={draft.length < 3 || busy}><Magnet size={15} /> 경계 보조 선택</button><button onClick={closeCurrent} disabled={draft.length < 3 || busy}>현재 도형 닫기 <Check size={15} /></button></div>
              {closed.length > 0 && <div className={styles.pendingShapes}><strong>닫힌 도형 {closed.length}개 · 사진 위 점을 끌어 수정</strong>{closed.map((shape, index) => <div key={shape.id}><span>{index + 1} · {shape.repeat ? `${shape.repeat_pitches}피치 반복` : '반복 없음'}</span><button onClick={() => reopenClosed(index)} disabled={!!draft.length || busy}>점 추가</button><button onClick={() => halfTurn.mutate(shape)} disabled={!!draft.length || busy || !shape.repeat || shape.repeat_pitches > 8}>180° 후보</button><button onClick={() => { setClosed(items => items.filter((_, i) => i !== index)); setHalfTurnProposal(null) }} disabled={busy}>제외</button></div>)}</div>}
              {halfTurnProposal && <div className={styles.symmetryChoices}><strong>중앙 기준 180° 회전 후보 · {halfTurnProposal.result.pitch_anchor_source === 'image' ? '사진 피치 기준' : '명목 피치 기준'}</strong>
                <small>영상 유사도는 확률이 아닙니다. 마모·조명·비대칭 형상에서는 낮을 수 있으니 위치와 경계를 직접 확인하세요.</small>
                <div>{halfTurnProposal.result.candidates.map((candidate, index) => <button key={index} className={index === halfTurnProposal.selected ? styles.symmetryActive : ''}
                  onClick={() => { setHalfTurnProposal({ ...halfTurnProposal, selected: index }); setFocusShape({ polygon: candidate.polygon, nonce: Date.now() }) }}>
                  후보 {index + 1} · 영상 일치도 {candidate.score.toFixed(2)}</button>)}</div>
                <div><button onClick={acceptHalfTurn} disabled={!halfTurnProposal.result.candidates.length}>선택 후보 도형에 추가</button><button onClick={() => setHalfTurnProposal(null)}>닫기</button></div></div>}
              <button className={styles.drawStart} onClick={saveClosed} disabled={!closed.length || !!draft.length || busy}>{save.isPending ? '저장 중…' : `닫힌 도형 ${closed.length}개 한 번에 저장`} <Check size={15} /></button>
              {assistNote && <p className={styles.toolNote}>{assistNote}</p>}</>
              : <button className={styles.drawStart} onClick={() => begin()} disabled={busy}><PenLine size={17} /> {section?.name} 형상 추가 <Plus size={15} /></button>}</div>
          {!editing && section && groupShapes.some(shape => shape.section_id === section.id && shape.repeat) &&
            <div className={styles.bulkRepeat}><strong>{section.name} 저장된 도형 반복 간격</strong><p>이 섹션의 반복 도형 모두에 같은 간격을 적용합니다.</p>
              <input aria-label="저장된 도형 반복 간격" type="number" min="1" max="2000" value={repeatPitches} onChange={event => setRepeatPitches(Math.max(1, Math.min(2000, Number(event.target.value) || 1)))} /> 피치마다
              <button disabled={busy} onClick={() => commit(analysis.groups, analysis.shapes.map(shape => shape.group_id === group.id && shape.section_id === section.id && shape.repeat ? { ...shape, repeat_pitches: repeatPitches } : shape))}>모든 저장 도형에 적용</button></div>}
          <div className={styles.savedSeeds}><div className={styles.miniHead}><strong>저장된 형상</strong><span>{savedSections.length}개 섹션 · {groupShapes.length}개 도형</span></div>
            {savedSections.length ? savedSections.map(({ section: label, shapes }) => <div className={styles.savedSection} key={label.id}>
              <div className={styles.seedRow}><i style={{ background: label.color }} /><button className={styles.savedSectionToggle} aria-expanded={expandedSaved === label.id}
                onClick={() => setExpandedSaved(value => value === label.id ? null : label.id)}>{label.name}<small>도형 {shapes.length}개 · 눌러서 개별 형상 보기</small></button>
                <button aria-label={`${label.name} 형상 모두 편집`} title="이 섹션의 도형 모두 편집" onClick={() => beginSection(label.id)} disabled={busy || editing}><PenLine size={15} /></button></div>
              {expandedSaved === label.id && <div className={styles.savedShapeList}>{shapes.map((shape, index) => <div className={styles.savedShapeRow} key={shape.id}>
                <span>{index + 1}<small>{shape.polygon.length}개 점 · {shape.repeat ? `${shape.repeat_pitches}피치 반복` : '반복 없음'}</small></span>
                <button aria-label={`${label.name} ${index + 1}번 도형 편집`} onClick={() => begin(shape)} disabled={busy || editing}><PenLine size={14} /></button>
                <button aria-label={`${label.name} ${index + 1}번 도형 삭제`} onClick={() => commit(analysis.groups, analysis.shapes.filter(item => item.id !== shape.id))} disabled={busy || editing}><Trash2 size={14} /></button></div>)}</div>}
            </div>) : <p>저장된 형상이 없습니다. 먼저 섹션을 선택하고 사진 위에 그려 주세요.</p>}</div>
        </> : <div className={styles.sectionIntro}><h3>분류 체계를 추가하세요</h3><p>형상별, 제조공법별 등 원하는 이름으로 만들 수 있습니다.</p></div>}
      </div></aside>
    </div>
  </div>
}
