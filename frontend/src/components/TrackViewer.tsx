import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import { Columns3, Maximize2, Minus, Plus, ScanSearch } from 'lucide-react'
import styles from '../App.module.css'

type Props = { base: string; hasTiles: boolean }

export default function TrackViewer({ base, hasTiles }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const viewer = useRef<OpenSeadragon.Viewer | null>(null)
  const [mode, setMode] = useState<'inspect' | 'sections'>('inspect')
  const [tileFailed, setTileFailed] = useState(false)

  useEffect(() => {
    if (!host.current || mode !== 'inspect' || !hasTiles) return
    setTileFailed(false)
    const instance = OpenSeadragon({ element: host.current, tileSources: `${base}/deepzoom.dzi`,
      showNavigator: false, showNavigationControl: false, animationTime: 0.25,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true },
    })
    viewer.current = instance
    instance.addHandler('open', () => {
      instance.viewport.zoomTo(instance.viewport.getHomeZoom() * 6)
      instance.viewport.panTo(new OpenSeadragon.Point(0.5, instance.viewport.getCenter().y))
      instance.viewport.applyConstraints()
    })
    instance.addHandler('open-failed', () => setTileFailed(true))
    return () => { viewer.current = null; instance.destroy() }
  }, [base, hasTiles, mode])

  function zoom(factor: number) { viewer.current?.viewport.zoomBy(factor); viewer.current?.viewport.applyConstraints() }
  return <div className={styles.viewerCard}>
    <div className={styles.viewerHead}><div><span className={styles.kicker}>SURFACE INSPECTION</span><h3>전개 사진 검토</h3></div><div className={styles.viewerTabs}><button className={mode === 'inspect' ? styles.viewerTabActive : styles.viewerTab} onClick={() => setMode('inspect')}><ScanSearch size={17} /> 확대 검사</button><button className={mode === 'sections' ? styles.viewerTabActive : styles.viewerTab} onClick={() => setMode('sections')}><Columns3 size={17} /> 4구간 보기</button></div></div>
    {mode === 'sections' ? <div className={styles.sectionsView}><img src={`${base}/review.jpg`} alt="완성된 전개 사진을 네 구간으로 나누어 표시" /></div>
      : <div className={styles.panoramaStage}>
          {hasTiles && !tileFailed ? <div ref={host} className={styles.deepzoom} aria-label="확대 및 이동 가능한 트랙 전개 사진" /> : <div className={styles.fallbackImage}><img src={`${base}/panorama.jpg`} alt="러버트랙 전개 사진" /></div>}
          {hasTiles && !tileFailed && <div className={styles.zoomControls}><button onClick={() => zoom(1.5)} aria-label="확대"><Plus size={18} /></button><button onClick={() => zoom(1 / 1.5)} aria-label="축소"><Minus size={18} /></button><button onClick={() => viewer.current?.viewport.goHome()} aria-label="전체 맞춤"><Maximize2 size={17} /></button></div>}
          <span className={styles.viewerHint}>{hasTiles ? '휠로 확대 · 드래그로 이동' : '좌우로 스크롤하여 검사'}</span>
        </div>}
    <div className={styles.viewerFooter}><span><b /> 촬영된 실제 표면을 사용한 결과</span><span>연결부는 확대하여 확인해 주세요.</span></div>
  </div>
}
