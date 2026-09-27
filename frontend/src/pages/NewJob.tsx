import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, FormEvent } from 'react'
import { useNavigate } from 'react-router'
import Uppy from '@uppy/core'
import XHRUpload from '@uppy/xhr-upload'
import { z } from 'zod'
import { ArrowDown, ArrowRight, ArrowUp, Check, GripVertical, ImagePlus, Info, Layers3, Plus, Trash2, UploadCloud } from 'lucide-react'
import { formatNumber } from '../api'
import styles from '../App.module.css'

type Photo = { id: string; file: File; url: string }
const validTypes = new Set(['image/jpeg', 'image/png', 'image/webp'])
const specSchema = z.object({ width: z.coerce.number().min(10).max(3000), pitch: z.coerce.number().min(5), links: z.string() })
  .refine(data => data.pitch <= data.width, '피치는 트랙 폭보다 클 수 없습니다.')
  .refine(data => data.links.trim() === '' || (/^[0-9]+$/.test(data.links.trim()) && Number(data.links) >= 1 && Number(data.links) <= 2000), '링크 수는 1~2000 범위의 정수여야 합니다.')

export default function NewJob() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const objectUrls = useRef<string[]>([])
  const uppyRef = useRef<Uppy | null>(null)
  const [width, setWidth] = useState('230')
  const [pitch, setPitch] = useState('48')
  const [links, setLinks] = useState('')
  const [photos, setPhotos] = useState<Photo[]>([])
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  useEffect(() => () => { objectUrls.current.forEach(URL.revokeObjectURL); uppyRef.current?.cancelAll() }, [])

  function addFiles(incoming: File[]) {
    if (!incoming.length) return
    if (photos.length + incoming.length > 100) { setError('한 작업에는 사진을 최대 100장 넣을 수 있습니다.'); return }
    if (incoming.some(file => !validTypes.has(file.type) || file.size > 30 * 1024 * 1024)) {
      setError('JPEG, PNG, WebP 사진을 사용하고 사진당 30MB 이내로 선택해 주세요.'); return
    }
    if ([...photos.map(photo => photo.file), ...incoming].reduce((sum, file) => sum + file.size, 0) > 500 * 1024 * 1024) {
      setError('한 작업의 사진 합계는 500MB 이내여야 합니다.'); return
    }
    const existing = new Set(photos.map(photo => `${photo.file.name}:${photo.file.size}:${photo.file.lastModified}`))
    const next = incoming.filter(file => !existing.has(`${file.name}:${file.size}:${file.lastModified}`)).map(file => {
      const url = URL.createObjectURL(file); objectUrls.current.push(url)
      return { id: crypto.randomUUID(), file, url }
    })
    setPhotos(previous => [...previous, ...next]); setError('')
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []))
    event.target.value = ''
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault()
    if (!event.dataTransfer.files.length) return
    addFiles(Array.from(event.dataTransfer.files))
  }

  function move(from: number, to: number) {
    if (from === to || to < 0 || to >= photos.length) return
    setPhotos(previous => { const updated = [...previous]; const [item] = updated.splice(from, 1); updated.splice(to, 0, item); return updated })
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const parsed = specSchema.safeParse({ width, pitch, links })
    if (!parsed.success) { setError(parsed.error.issues[0].message); return }
    if (!photos.length) { setError('제품 사진을 한 장 이상 추가해 주세요.'); return }
    setError(''); setUploading(true); setProgress(0)
    const uppy = new Uppy({ autoProceed: false, restrictions: { maxNumberOfFiles: 100, maxFileSize: 30 * 1024 * 1024 } })
    uppyRef.current = uppy
    uppy.setMeta({ width_mm: String(parsed.data.width), pitch_mm: String(parsed.data.pitch), total_links: parsed.data.links.trim() })
    uppy.use(XHRUpload, {
      endpoint: '/api/jobs', method: 'POST', formData: true, bundle: true, fieldName: 'images',
      allowedMetaFields: ['width_mm', 'pitch_mm', 'total_links'],
      getResponseData: xhr => JSON.parse(xhr.responseText),
    })
    uppy.on('progress', value => setProgress(value ?? 0))
    try {
      photos.forEach(photo => uppy.addFile({ name: photo.file.name, type: photo.file.type, data: photo.file, source: 'local' }))
      const result = await uppy.upload()
      if (!result || result.failed?.length) throw new Error(result?.failed?.[0]?.error ?? '사진 업로드에 실패했습니다.')
      const body = result.successful?.[0]?.response?.body as { id?: string } | undefined
      if (!body?.id) throw new Error('작업 번호를 받지 못했습니다. 다시 시도해 주세요.')
      navigate(`/jobs/${body.id}`)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '업로드를 완료하지 못했습니다.')
    } finally {
      setUploading(false); uppyRef.current = null; uppy.destroy()
    }
  }

  const length = Number(links) > 0 && Number(pitch) > 0 ? Number(links) * Number(pitch) : null
  return <form onSubmit={submit} className={styles.pageStack}>
    <div className={styles.pageIntro}><div><span className={styles.kicker}>CREATE A NEW PANORAMA</span><h2>제품 정보를 입력하고 사진을 올려주세요.</h2><p>사진의 순서와 겹침을 확인한 뒤 한 장의 트랙 전개 사진을 만듭니다.</p></div><div className={styles.stepCounter}>01 <span>/ 02</span></div></div>
    <div className={styles.createGrid}>
      <div className={styles.createMain}>
        <section className={styles.formCard}>
          <div className={styles.cardHead}><span className={styles.stepNumber}>01</span><div><h3>트랙 규격</h3><p>폭과 피치는 필수입니다. 링크 수는 알고 있을 때만 입력하세요.</p></div></div>
          <div className={styles.fields}>
            <label className={styles.field}><span>트랙 폭 <b>필수</b></span><div className={styles.inputUnit}><input type="number" inputMode="decimal" min="10" max="3000" step="any" value={width} onChange={e => setWidth(e.target.value)} required disabled={uploading} /><small>mm</small></div><em>바깥쪽에서 잰 제품의 전체 폭</em></label>
            <label className={styles.field}><span>피치 <b>필수</b></span><div className={styles.inputUnit}><input type="number" inputMode="decimal" min="5" step="any" value={pitch} onChange={e => setPitch(e.target.value)} required disabled={uploading} /><small>mm</small></div><em>인접한 링크 사이의 길이</em></label>
            <label className={styles.field}><span>전체 링크 수 <i>선택</i></span><div className={styles.inputUnit}><input type="number" inputMode="numeric" min="1" max="2000" step="1" placeholder="사진에서 추정" value={links} onChange={e => setLinks(e.target.value)} disabled={uploading} /><small>링크</small></div><em>완료 후에도 수정할 수 있습니다.</em></label>
          </div>
          <div className={styles.formNote}><Info size={17} /><span>한 바퀴의 같은 표면이 확인되면 사진에서 링크 수를 제안합니다. 사진이 일부만 있으면 전체 링크 수를 임의로 채우지 않습니다.</span></div>
        </section>

        <section className={styles.formCard}>
          <div className={styles.cardHead}><span className={styles.stepNumber}>02</span><div><h3>제품 사진</h3><p>사진을 촬영 순서대로 놓아주세요. 순서는 작업 시작 전에 바꿀 수 있습니다.</p></div></div>
          <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={onInputChange} hidden />
          <div className={styles.dropzone} onDragOver={e => e.preventDefault()} onDrop={onDrop} onClick={() => !uploading && inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && !uploading) { e.preventDefault(); inputRef.current?.click() } }}>
            <span className={styles.dropIcon}><UploadCloud size={26} /></span><strong>사진을 여기에 놓거나 선택하세요</strong><p>여러 장을 한 번에 선택할 수 있습니다.</p><span className={styles.dropSelect}><Plus size={15} /> 사진 선택</span>
            <small>JPG · PNG · WebP / 사진당 최대 30MB</small>
          </div>
          {photos.length > 0 && <><div className={styles.photoListHead}><div><strong>촬영 순서</strong><span>{photos.length}장 선택됨</span></div><button type="button" onClick={() => inputRef.current?.click()} disabled={uploading}><ImagePlus size={16} /> 사진 추가</button></div>
            <div className={styles.photoList}>{photos.map((photo, index) => <div className={styles.photoRow} key={photo.id} draggable={!uploading} onDragStart={event => { setDragIndex(index); event.dataTransfer.effectAllowed = 'move' }} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); event.stopPropagation(); if (dragIndex !== null) move(dragIndex, index); setDragIndex(null) }} onDragEnd={() => setDragIndex(null)}>
              <GripVertical size={17} className={styles.grip} /><span className={styles.photoOrder}>{String(index + 1).padStart(2, '0')}</span><img src={photo.url} alt="" /><div className={styles.photoName}><strong title={photo.file.name}>{photo.file.name}</strong><small>{formatNumber(photo.file.size / 1024 / 1024)} MB</small></div>
              <div className={styles.photoActions}><button type="button" aria-label={`${photo.file.name} 앞 순서로`} onClick={() => move(index, index - 1)} disabled={index === 0 || uploading}><ArrowUp size={15} /></button><button type="button" aria-label={`${photo.file.name} 뒤 순서로`} onClick={() => move(index, index + 1)} disabled={index === photos.length - 1 || uploading}><ArrowDown size={15} /></button><button type="button" aria-label={`${photo.file.name} 제거`} onClick={() => setPhotos(previous => previous.filter(item => item.id !== photo.id))} disabled={uploading}><Trash2 size={15} /></button></div>
            </div>)}</div></>}
        </section>
      </div>

      <aside className={styles.createAside}>
        <div className={styles.summaryCard}><span className={styles.kicker}>JOB SUMMARY</span><h3>작업 요약</h3><div className={styles.summaryRows}><div><span>트랙 폭</span><strong>{width || '—'} <small>mm</small></strong></div><div><span>피치</span><strong>{pitch || '—'} <small>mm</small></strong></div><div><span>업로드 사진</span><strong>{photos.length} <small>장</small></strong></div><div><span>링크 수</span><strong>{links || '사진에서 추정'}</strong></div></div><div className={styles.lengthPreview}><span>계산되는 총 길이</span><strong>{length === null ? '링크 수 확인 후 표시' : `${formatNumber(length)} mm`}</strong>{length !== null && <small>{formatNumber(length / 1000)} m = {pitch} mm × {links}링크</small>}</div><div className={styles.summaryDivider} /><div className={styles.checkList}><span><Check size={16} /> 폭과 피치로 크기 보정</span><span><Check size={16} /> 겹친 표면 자동 식별</span><span><Check size={16} /> 원본 결함과 마모 유지</span></div>
          {error && <div className={styles.formError} role="alert">{error}</div>}
          {uploading && <div className={styles.uploadProgress}><div><span>사진 업로드 중</span><strong>{progress}%</strong></div><progress value={progress} max="100" /></div>}
          <button className={styles.submitButton} type="submit" disabled={uploading}>{uploading ? '업로드 중…' : '전개 작업 시작'} <ArrowRight size={18} /></button><p className={styles.summaryFoot}><Layers3 size={15} /> 작업이 시작되면 상태 화면으로 이동합니다.</p>
        </div>
      </aside>
    </div>
  </form>
}
