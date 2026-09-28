import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { Maximize2, Minus, Plus } from 'lucide-react'
import type { DamageAnalysis, DamageCandidate, Point } from '../types'
import { damageFileBase, fileBase } from '../api'
import styles from '../pages/DamageWorkspace.module.css'

export type CanvasLayer = 'source' | 'zones' | 'candidates'
type Props = {
  jobId: string
  analysis: DamageAnalysis
  layer: CanvasLayer
  drawing: boolean
  draft: Point[]
  selectedId: string | null
  focusTarget: { candidate: DamageCandidate; nonce: number } | null
  onAddPoint: (point: Point) => void
  onSelectCandidate: (id: string) => void
}

export default function DamageCanvas({ jobId, analysis, layer, drawing, draft, selectedId, focusTarget, onAddPoint, onSelectCandidate }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const instance = useRef<OpenSeadragon.Viewer | null>(null)
  const [frame, setFrame] = useState(0)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const [size, setSize] = useState({ width: 1, height: 1 })

  useEffect(() => {
    if (!host.current) return
    const viewer = OpenSeadragon({
      element: host.current, tileSources: `${fileBase(jobId)}/deepzoom.dzi`,
      showNavigationControl: false, showNavigator: true, navigatorPosition: 'BOTTOM_RIGHT',
      animationTime: .2, visibilityRatio: .1, minZoomImageRatio: .3,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true },
    })
    instance.current = viewer
    const update = () => setFrame(value => value + 1)
    viewer.addHandler('animation', update)
    viewer.addHandler('resize', update)
    viewer.addHandler('open', () => {
      setReady(true)
      const [width, height] = analysis.image_size_wh
      const imageHeight = height / width
      const visibleWidth = imageHeight * 1.12 * (host.current!.clientWidth / host.current!.clientHeight)
      viewer.viewport.fitBounds(new OpenSeadragon.Rect(.5 - visibleWidth / 2, -.06 * imageHeight, visibleWidth, 1.12 * imageHeight), true)
      update()
    })
    viewer.addHandler('open-failed', () => setFailed(true))
    const observer = new ResizeObserver(() => {
      if (host.current) setSize({ width: host.current.clientWidth, height: host.current.clientHeight })
      update()
    })
    observer.observe(host.current)
    setSize({ width: host.current.clientWidth, height: host.current.clientHeight })
    return () => { observer.disconnect(); instance.current = null; viewer.destroy() }
    // The panorama and its pixel coordinate system stay fixed for this job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  useEffect(() => { instance.current?.setMouseNavEnabled(!drawing) }, [drawing])

  useEffect(() => {
    const viewer = instance.current
    if (!viewer || !ready || !focusTarget) return
    const [x, y, w, h] = focusTarget.candidate.bbox_xywh
    const point = viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(x + w / 2, y + h / 2))
    const imageRatio = analysis.image_size_wh[1] / analysis.image_size_wh[0]
    viewer.viewport.zoomTo(Math.max(viewer.viewport.getZoom(), 1 / (imageRatio * 1.7)), undefined, true)
    const bounds = viewer.viewport.getBounds(true)
    const centerX = Math.max(bounds.width / 2, Math.min(1 - bounds.width / 2, point.x))
    const centerY = Math.max(bounds.height / 2, Math.min(imageRatio - bounds.height / 2, point.y))
    viewer.viewport.panTo(new OpenSeadragon.Point(centerX, centerY), true)
    viewer.viewport.applyConstraints()
  }, [focusTarget, ready, analysis.image_size_wh])

  function project(point: Point): Point {
    const viewer = instance.current
    if (!viewer || !ready) return [0, 0]
    const mapped = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(point[0], point[1]))
    return [mapped.x, mapped.y]
  }

  function polygon(points: Point[]) { return points.map(point => project(point).join(',')).join(' ') }

  function clickStage(event: React.MouseEvent<SVGSVGElement>) {
    if (!drawing || !host.current || !instance.current) return
    const rect = host.current.getBoundingClientRect()
    const mapped = instance.current.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top))
    const [width, height] = analysis.image_size_wh
    if (mapped.x < 0 || mapped.x >= width || mapped.y < 0 || mapped.y >= height) return
    onAddPoint([Math.round(mapped.x), Math.round(mapped.y)])
  }

  function zoom(factor: number) { instance.current?.viewport.zoomBy(factor); instance.current?.viewport.applyConstraints() }
  const [imageWidth, imageHeight] = analysis.image_size_wh
  const left = project([0, 0])
  const right = project([imageWidth, imageHeight])
  const grid = drawing ? Array.from({ length: Math.min(201, Math.ceil(imageWidth / analysis.pitch_px) + 1) }, (_, index) => index * analysis.pitch_px)
    .filter(x => { const screen = project([x, 0])[0]; return screen >= -30 && screen <= size.width + 30 }) : []
  void frame

  return <div className={styles.canvasShell}>
    <div ref={host} className={styles.deepzoom} aria-label="확대 가능한 트랙 전개 사진" />
    {failed && <div className={styles.canvasFailure}>확대 이미지를 불러오지 못했습니다. 전개 결과 파일을 확인해 주세요.</div>}
    {ready && <svg className={styles.overlay} width={size.width} height={size.height} onClick={clickStage}
      style={{ pointerEvents: drawing ? 'auto' : 'none', cursor: drawing ? 'crosshair' : 'default' }}>
      {layer === 'zones' && <image href={`${damageFileBase(jobId)}/zone_overlay.png?v=${encodeURIComponent(analysis.created_at)}`}
        x={left[0]} y={left[1]} width={right[0] - left[0]} height={right[1] - left[1]} preserveAspectRatio="none" pointerEvents="none" />}
      {layer === 'candidates' && analysis.candidates.map(candidate => <polygon key={candidate.id}
        points={polygon(candidate.polygon)} className={candidate.id === selectedId ? styles.candidateSelected : candidate.included ? styles.candidateIncluded : styles.candidateExcluded}
        data-mode={candidate.mode} onClick={event => { event.stopPropagation(); if (!drawing) onSelectCandidate(candidate.id) }}
        style={{ pointerEvents: drawing ? 'none' : 'auto' }} />)}
      {grid.map((x, index) => {
        const top = project([x, 0]); const bottom = project([x, imageHeight])
        return <line key={index} x1={top[0]} y1={top[1]} x2={bottom[0]} y2={bottom[1]} className={styles.pitchGuide} />
      })}
      {draft.length > 1 && <polyline points={polygon(draft)} className={styles.draftLine} />}
      {draft.length >= 3 && <polygon points={polygon(draft)} className={styles.draftFill} />}
      {draft.map((point, index) => { const [x, y] = project(point); return <circle key={index} cx={x} cy={y} r={4} className={styles.draftPoint} /> })}
    </svg>}
    <div className={styles.zoomControls}><button onClick={() => zoom(1.4)} aria-label="확대"><Plus size={17} /></button><button onClick={() => zoom(1 / 1.4)} aria-label="축소"><Minus size={17} /></button><button onClick={() => instance.current?.viewport.goHome()} aria-label="전체 보기"><Maximize2 size={17} /></button></div>
    <div className={styles.canvasHint}>{drawing ? '한 피치에서 경계를 따라 점을 찍으세요 · 드래그 이동은 잠시 꺼집니다' : '휠로 확대 · 드래그로 이동 · 후보를 눌러 선택'}</div>
  </div>
}
