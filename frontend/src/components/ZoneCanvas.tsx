import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { Maximize2, Minus, Plus } from 'lucide-react'
import type { CrackCandidate, Point, ZoneAnalysis, ZoneInstance, ZoneShape } from '../types'
import { fileBase, zoneFileBase } from '../api'
import styles from '../pages/ZoneWorkspace.module.css'

type Props = {
  jobId: string
  groupId: string | null
  analysis: ZoneAnalysis
  editing: boolean
  points: Point[]
  closedShapes: ZoneShape[]
  selectedVertex: number | null
  opacity: number
  focusShape: { polygon: Point[]; nonce: number } | null
  suggestedPolygon: Point[] | null
  segmentInstances: ZoneInstance[]
  adjustingSegments: boolean
  selectedSegment: { shapeId: string; placement: number } | null
  onSegmentSelect: (shapeId: string, placement: number) => void
  onSegmentMove: (shapeId: string, placement: number, offset: Point) => void
  crackCandidates?: CrackCandidate[]
  selectedCrackId?: string | null
  onCrackSelect?: (id: string) => void
  hint?: string
  onAdd: (point: Point) => void
  onInsert: (index: number, point: Point) => void
  onMoveStart: (index: number) => void
  onMove: (index: number, point: Point) => void
  onSelect: (index: number) => void
  onDelete: (index: number) => void
  onClose: () => void
  onClosedMove: (shapeId: string, index: number, point: Point) => void
  onClosedDelete: (shapeId: string, index: number) => void
}

export default function ZoneCanvas({ jobId, groupId, analysis, editing, points, closedShapes, selectedVertex, opacity,
  focusShape, suggestedPolygon, segmentInstances, adjustingSegments, selectedSegment, onSegmentSelect, onSegmentMove,
  crackCandidates = [], selectedCrackId = null, onCrackSelect, hint,
  onAdd, onInsert, onMoveStart, onMove, onSelect, onDelete, onClose, onClosedMove, onClosedDelete }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<SVGSVGElement>(null)
  const viewer = useRef<OpenSeadragon.Viewer | null>(null)
  const dragging = useRef<number | null>(null)
  const draggingClosed = useRef<string | null>(null)
  const pointerStart = useRef<Point | null>(null)
  const moved = useRef(false)
  const panPointer = useRef<{ id: number; position: Point } | null>(null)
  const segmentDrag = useRef<{ key: string; id: number; start: Point; offset: Point; shapeId: string; placement: number } | null>(null)
  const [segmentPreview, setSegmentPreview] = useState<{ key: string; delta: Point } | null>(null)
  const [panning, setPanning] = useState(false)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const [size, setSize] = useState({ width: 1, height: 1 })
  const [, render] = useState(0)

  useEffect(() => {
    if (!host.current) return
    const instance = OpenSeadragon({ element: host.current, tileSources: `${fileBase(jobId)}/deepzoom.dzi`,
      showNavigationControl: false, showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT', animationTime: .2,
      visibilityRatio: .1, minZoomImageRatio: .3,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true } })
    viewer.current = instance
    const update = () => render(value => value + 1)
    instance.addHandler('animation', update)
    instance.addHandler('resize', update)
    instance.addHandler('open', () => {
      setReady(true)
      const [width, height] = analysis.image_size_wh
      const imageHeight = height / width
      const visibleWidth = imageHeight * 1.12 * (host.current!.clientWidth / host.current!.clientHeight)
      instance.viewport.fitBounds(new OpenSeadragon.Rect(.5 - visibleWidth / 2, -.06 * imageHeight,
        visibleWidth, 1.12 * imageHeight), true)
      update()
    })
    instance.addHandler('open-failed', () => setFailed(true))
    const observer = new ResizeObserver(() => {
      if (host.current) setSize({ width: host.current.clientWidth, height: host.current.clientHeight })
      update()
    })
    observer.observe(host.current)
    setSize({ width: host.current.clientWidth, height: host.current.clientHeight })
    return () => { observer.disconnect(); viewer.current = null; instance.destroy() }
    // The panorama coordinate system is immutable for the life of this job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  useEffect(() => { viewer.current?.setMouseNavEnabled(!editing) }, [editing])

  useEffect(() => {
    const overlay = overlayRef.current
    if (!editing || !ready || !overlay || !host.current) return
    const wheelZoom = (event: WheelEvent) => {
      const instance = viewer.current
      const element = host.current
      if (!instance || !element) return
      event.preventDefault()
      event.stopPropagation()
      const bounds = element.getBoundingClientRect()
      const pointer = new OpenSeadragon.Point(event.clientX - bounds.left, event.clientY - bounds.top)
      const anchor = instance.viewport.viewerElementToViewportCoordinates(pointer)
      const factor = Math.max(.65, Math.min(1.5, Math.exp(-event.deltaY * .0012)))
      instance.viewport.zoomBy(factor, anchor)
      instance.viewport.applyConstraints()
    }
    overlay.addEventListener('wheel', wheelZoom, { passive: false })
    return () => overlay.removeEventListener('wheel', wheelZoom)
  }, [editing, ready])

  useEffect(() => {
    if (!ready || !viewer.current || !focusShape?.polygon.length) return
    const xs = focusShape.polygon.map(point => point[0])
    const ys = focusShape.polygon.map(point => point[1])
    const instance = viewer.current
    const center = instance.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(
      (Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2))
    const imageRatio = analysis.image_size_wh[1] / analysis.image_size_wh[0]
    instance.viewport.zoomTo(Math.max(instance.viewport.getZoom(), 1 / (imageRatio * 1.8)), undefined, true)
    const bounds = instance.viewport.getBounds(true)
    instance.viewport.panTo(new OpenSeadragon.Point(
      Math.max(bounds.width/2, Math.min(1-bounds.width/2, center.x)),
      Math.max(bounds.height/2, Math.min(imageRatio-bounds.height/2, center.y))), true)
    instance.viewport.applyConstraints()
  }, [focusShape, ready, analysis.image_size_wh])

  function project(point: Point): Point {
    if (!viewer.current || !ready) return [0, 0]
    const mapped = viewer.current.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(...point))
    return [mapped.x, mapped.y]
  }
  function unproject(event: React.PointerEvent | React.MouseEvent): Point | null {
    if (!host.current || !viewer.current) return null
    const rect = host.current.getBoundingClientRect()
    const mapped = viewer.current.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top))
    const [width, height] = analysis.image_size_wh
    if (mapped.x < 0 || mapped.x >= width || mapped.y < 0 || mapped.y >= height) return null
    return [Math.round(mapped.x), Math.round(mapped.y)]
  }
  function rawImagePoint(event: React.PointerEvent): Point {
    const rect = host.current!.getBoundingClientRect()
    const mapped = viewer.current!.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top))
    return [mapped.x, mapped.y]
  }
  function beginSegmentDrag(event: React.PointerEvent<SVGGElement>, instance: ZoneInstance) {
    if (event.button !== 0 || !viewer.current || !host.current) return
    event.preventDefault(); event.stopPropagation()
    const key = `${instance.shape_id}:${instance.placement}`
    segmentDrag.current = { key, id: event.pointerId, start: rawImagePoint(event), offset: instance.offset,
      shapeId: instance.shape_id, placement: instance.placement }
    setSegmentPreview({ key, delta: [0, 0] })
    onSegmentSelect(instance.shape_id, instance.placement)
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  function segmentDelta(event: React.PointerEvent, drag: NonNullable<typeof segmentDrag.current>): Point {
    const at = rawImagePoint(event)
    const maxX = analysis.pitch_px * 1.5, maxY = analysis.image_size_wh[1] * .5
    const absolute: Point = [Math.max(-maxX, Math.min(maxX, drag.offset[0] + at[0] - drag.start[0])),
      Math.max(-maxY, Math.min(maxY, drag.offset[1] + at[1] - drag.start[1]))]
    return [absolute[0] - drag.offset[0], absolute[1] - drag.offset[1]]
  }
  function dragSegment(event: React.PointerEvent<SVGGElement>) {
    const drag = segmentDrag.current
    if (!drag || drag.id !== event.pointerId) return
    event.preventDefault(); event.stopPropagation()
    setSegmentPreview({ key: drag.key, delta: segmentDelta(event, drag) })
  }
  function endSegmentDrag(event: React.PointerEvent<SVGGElement>, save: boolean) {
    const drag = segmentDrag.current
    if (!drag || drag.id !== event.pointerId) return
    event.preventDefault(); event.stopPropagation()
    const delta = segmentDelta(event, drag)
    if (save && Math.hypot(...delta) > 1) {
      onSegmentMove(drag.shapeId, drag.placement, [drag.offset[0] + delta[0], drag.offset[1] + delta[1]])
    }
    segmentDrag.current = null; setSegmentPreview(null)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }
  function startMiddlePan(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 1 || !ready || !viewer.current || !host.current) return
    event.preventDefault()
    event.stopPropagation()
    panPointer.current = { id: event.pointerId, position: [event.clientX, event.clientY] }
    event.currentTarget.setPointerCapture(event.pointerId)
    setPanning(true)
  }
  function moveMiddlePan(event: React.PointerEvent<HTMLDivElement>) {
    const pan = panPointer.current
    const instance = viewer.current
    const element = host.current
    if (!pan || pan.id !== event.pointerId || !instance || !element) return
    event.preventDefault()
    const bounds = element.getBoundingClientRect()
    const previous = instance.viewport.viewerElementToViewportCoordinates(
      new OpenSeadragon.Point(pan.position[0] - bounds.left, pan.position[1] - bounds.top))
    const current = instance.viewport.viewerElementToViewportCoordinates(
      new OpenSeadragon.Point(event.clientX - bounds.left, event.clientY - bounds.top))
    instance.viewport.panBy(previous.minus(current), true)
    instance.viewport.applyConstraints()
    pan.position = [event.clientX, event.clientY]
  }
  function endMiddlePan(event: React.PointerEvent<HTMLDivElement>) {
    if (panPointer.current?.id !== event.pointerId) return
    event.preventDefault()
    panPointer.current = null
    setPanning(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }
  function lostMiddlePan(event: React.PointerEvent<HTMLDivElement>) {
    if (panPointer.current?.id === event.pointerId) { panPointer.current = null; setPanning(false) }
  }
  const polygon = (items: Point[]) => items.map(point => project(point).join(',')).join(' ')
  const [imageWidth, imageHeight] = analysis.image_size_wh
  const left = project([0, 0]), right = project([imageWidth, imageHeight])
  const gridPositions = analysis.pitch_anchors_x?.length ? analysis.pitch_anchors_x
    : Array.from({ length: Math.min(201, Math.ceil(imageWidth / analysis.pitch_px) + 1) }, (_, index) => index * analysis.pitch_px)
  const grid = editing ? gridPositions.filter(x => {
      const screen = project([x, 0])[0]
      return screen >= -30 && screen <= size.width + 30
    }) : []

  return <div className={styles.canvasShell} onPointerDownCapture={startMiddlePan}
    onPointerMove={moveMiddlePan} onPointerUp={endMiddlePan} onPointerCancel={endMiddlePan}
    onLostPointerCapture={lostMiddlePan}
    onAuxClick={event => { if (event.button === 1) event.preventDefault() }}>
    <div ref={host} className={styles.deepzoom} aria-label="확대 가능한 트랙 전개 사진" style={{ cursor: panning ? 'grabbing' : undefined }} />
    {failed && <div className={styles.canvasFailure}>확대 이미지를 불러오지 못했습니다.</div>}
    {ready && <svg ref={overlayRef} className={styles.overlay} width={size.width} height={size.height}
      style={{ pointerEvents: editing ? 'auto' : 'none', cursor: panning ? 'grabbing' : editing ? 'crosshair' : 'default' }}
      onClick={event => { const point = unproject(event); if (editing && event.button === 0 && point) onAdd(point) }}>
      {groupId && <image href={`${zoneFileBase(jobId, groupId)}/overlay.png?v=${encodeURIComponent(analysis.created_at)}`}
        x={left[0]} y={left[1]} width={right[0]-left[0]} height={right[1]-left[1]}
        preserveAspectRatio="none" opacity={opacity} pointerEvents="none" />}
      {grid.map((x, index) => {
        const a = project([x, 0]), b = project([x, imageHeight])
        return <line key={index} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={styles.pitchGuide} />
      })}
      {closedShapes.map((shape, index) => <g key={shape.id}>
        <polygon points={polygon(shape.polygon)} className={styles.pendingFill} />
        <text x={project(shape.polygon[0])[0] + 8} y={project(shape.polygon[0])[1] + 17} className={styles.pendingIndex}>{index + 1}</text>
        {editing && shape.polygon.map((point, vertex) => {
          const [x, y] = project(point), key = `${shape.id}:${vertex}`
          return <circle key={key} cx={x} cy={y} r={7} className={styles.closedPoint}
            onPointerDown={event => { event.stopPropagation(); draggingClosed.current = key; event.currentTarget.setPointerCapture(event.pointerId) }}
            onPointerMove={event => { if (draggingClosed.current !== key) return; const at = unproject(event); if (at) onClosedMove(shape.id, vertex, at) }}
            onPointerUp={event => { event.stopPropagation(); draggingClosed.current = null; event.currentTarget.releasePointerCapture(event.pointerId) }}
            onPointerCancel={() => { draggingClosed.current = null }}
            onClick={event => event.stopPropagation()}
            onContextMenu={event => { event.preventDefault(); event.stopPropagation(); onClosedDelete(shape.id, vertex) }} />
        })}
      </g>)}
      {suggestedPolygon && <polygon points={polygon(suggestedPolygon)} className={styles.symmetrySuggestion} />}
      {crackCandidates.map(candidate => {
        const [x, y, width, height] = candidate.bbox
        const [cx, cy] = project([x + width/2, y + height/2])
        if (cx < -15 || cx > size.width + 15 || cy < -15 || cy > size.height + 15) return null
        const selected = selectedCrackId === candidate.id
        return <g key={candidate.id} style={{ pointerEvents: 'all', cursor: 'pointer' }} onClick={event => { event.stopPropagation(); onCrackSelect?.(candidate.id) }}>
          {selected && <polygon points={polygon(candidate.polygon)} className={styles.crackOutline} />}
          <circle cx={cx} cy={cy} r={selected ? 9 : 5} className={candidate.status === 'accepted' ? styles.crackAccepted : candidate.status === 'excluded' ? styles.crackExcluded : styles.crackPending} />
        </g>
      })}
      {adjustingSegments && segmentInstances.map(instance => {
        const key = `${instance.shape_id}:${instance.placement}`
        const xs = instance.polygon.map(point => point[0]), ys = instance.polygon.map(point => point[1])
        const center: Point = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2]
        const [cx, cy] = project(center)
        if (cx < -20 || cx > size.width + 20 || cy < -20 || cy > size.height + 20) return null
        const delta = segmentPreview?.key === key ? segmentPreview.delta : [0, 0]
        const shifted = instance.polygon.map(([x, y]): Point => [x + delta[0], y + delta[1]])
        const [hx, hy] = project([center[0] + delta[0], center[1] + delta[1]])
        const active = selectedSegment?.shapeId === instance.shape_id && selectedSegment.placement === instance.placement
        return <g key={key} className={styles.segmentHandle} style={{ pointerEvents: 'all' }}
          onPointerDown={event => beginSegmentDrag(event, instance)} onPointerMove={dragSegment}
          onPointerUp={event => endSegmentDrag(event, true)} onPointerCancel={event => endSegmentDrag(event, false)}
          onClick={event => event.stopPropagation()}>
          {(active || segmentPreview?.key === key) && <polygon points={polygon(shifted)} className={styles.segmentOutline} pointerEvents="none" />}
          <circle cx={hx} cy={hy} r={active ? 13 : 11} className={active ? styles.segmentHandleActive : styles.segmentHandleCircle} />
          <path d={`M ${hx-5} ${hy} h 10 M ${hx} ${hy-5} v 10`} className={styles.segmentHandleCross} />
        </g>
      })}
      {points.length > 1 && <polyline points={polygon(points)} className={styles.draftLine} />}
      {editing && points.length >= 2 && points.map((point, index) => {
        const next = points[(index + 1) % points.length]
        if (index === points.length-1) return null
        const a = project(point), b = project(next)
        return <line key={`edge-${index}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
          className={styles.insertEdge} onClick={event => { event.stopPropagation(); const at = unproject(event); if (at) onInsert(index+1, at) }} />
      })}
      {editing && points.map((point, index) => {
        const [x, y] = project(point)
        const closing = index === 0 && points.length >= 3
        return <circle key={index} cx={x} cy={y} r={closing ? 10 : selectedVertex === index ? 7 : 6}
          className={closing ? styles.closingPoint : selectedVertex === index ? styles.selectedPoint : styles.draftPoint}
          onPointerDown={event => { event.stopPropagation(); dragging.current = index; pointerStart.current = [event.clientX, event.clientY]; moved.current = false; onMoveStart(index); onSelect(index); event.currentTarget.setPointerCapture(event.pointerId) }}
          onPointerMove={event => {
            if (dragging.current !== index) return
            if (pointerStart.current && Math.hypot(event.clientX - pointerStart.current[0], event.clientY - pointerStart.current[1]) > 4) moved.current = true
            if (moved.current) { const at = unproject(event); if (at) onMove(index, at) }
          }}
          onPointerUp={event => { event.stopPropagation(); dragging.current = null; event.currentTarget.releasePointerCapture(event.pointerId) }}
          onClick={event => { event.stopPropagation(); if (closing && !moved.current) onClose(); else onSelect(index) }}
          onContextMenu={event => { event.preventDefault(); event.stopPropagation(); onDelete(index) }} />
      })}
    </svg>}
    <div className={styles.zoomControls}><button onClick={() => viewer.current?.viewport.zoomBy(1.4)} aria-label="확대"><Plus size={17} /></button>
      <button onClick={() => viewer.current?.viewport.zoomBy(1/1.4)} aria-label="축소"><Minus size={17} /></button>
      <button onClick={() => viewer.current?.viewport.goHome()} aria-label="전체 보기"><Maximize2 size={17} /></button></div>
    <div className={styles.canvasHint}>{hint ?? (editing ? '첫 점 클릭으로 도형 닫기 · 휠 확대·축소 · 휠 버튼 드래그로 이동 · 닫힌 도형 점 드래그'
      : adjustingSegments ? '십자 핸들 드래그로 이 세그먼트만 이동 · 빈 사진 드래그로 화면 이동'
      : '휠로 확대 · 왼쪽 또는 휠 버튼 드래그로 이동')}</div>
  </div>
}
