import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, ArrowLeft, ArrowRight, Check, CheckCircle2, Clock3, Download, FileJson2, Image as ImageIcon, Pencil, RotateCcw, ScanLine, ScanSearch, X } from 'lucide-react'
import { Link, useParams } from 'react-router'
import { api, fileBase, formatDate, formatNumber, sampleBase } from '../api'
import type { Links } from '../types'
import TrackViewer from '../components/TrackViewer'
import styles from '../App.module.css'

function LinkEditor({ jobId, links, pitch }: { jobId: string; links: Links; pitch: number }) {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState('')
  const [error, setError] = useState('')
  const queryClient = useQueryClient()
  const update = useMutation({ mutationFn: (count: number | null) => api.updateLinks(jobId, count),
    onSuccess: data => { queryClient.setQueryData(['job', jobId], data); queryClient.invalidateQueries({ queryKey: ['jobs'] }); setOpen(false); setError('') },
    onError: cause => setError(cause instanceof Error ? cause.message : '수정 내용을 저장하지 못했습니다.'),
  })
  function save() {
    const count = Number(value)
    if (!Number.isInteger(count) || count < 1 || count > 2000) { setError('링크 수는 1~2000 범위의 정수여야 합니다.'); return }
    update.mutate(count)
  }
  return <Dialog.Root open={open} onOpenChange={next => { setOpen(next); if (next) { setValue(String(links.entered_links ?? links.suggested_links ?? '')); setError('') } }}>
    <Dialog.Trigger asChild><button className={styles.editButton}><Pencil size={15} /> 링크 수 수정</button></Dialog.Trigger>
    <Dialog.Portal><Dialog.Overlay className={styles.dialogOverlay} /><Dialog.Content className={styles.dialogContent}>
      <div className={styles.dialogTop}><div><span className={styles.kicker}>LINK COUNT</span><Dialog.Title>링크 수 수정</Dialog.Title></div><Dialog.Close className={styles.dialogClose} aria-label="닫기"><X size={20} /></Dialog.Close></div>
      <Dialog.Description className={styles.dialogDescription}>알고 있는 실제 링크 수를 입력할 수 있습니다. 계산된 총 길이는 피치 × 적용 링크 수입니다. 사진 결과 자체는 바뀌지 않습니다.</Dialog.Description>
      {links.suggested_links !== null && <div className={styles.suggestedBox}><ScanLine size={18} /><span>사진에서 추정한 값</span><strong>{links.suggested_links}링크</strong></div>}
      <label className={styles.dialogField}>적용할 링크 수<div className={styles.inputUnit}><input type="number" min="1" max="2000" step="1" value={value} onChange={event => setValue(event.target.value)} autoFocus /><small>링크</small></div></label>
      <div className={styles.dialogFormula}>{value && Number(value) > 0 ? `${formatNumber(Number(value) * pitch)} mm = ${pitch} mm × ${value}링크` : '링크 수를 입력하면 총 길이가 표시됩니다.'}</div>
      {error && <p role="alert" className={styles.formError}>{error}</p>}
      <div className={styles.dialogActions}><button className={styles.ghostButton} onClick={() => update.mutate(null)} disabled={update.isPending}><RotateCcw size={15} /> {links.suggested_links === null ? '입력 지우기' : '사진 추정값 사용'}</button><button className={styles.primaryButton} onClick={save} disabled={update.isPending}>적용하기 <ArrowRight size={16} /></button></div>
    </Dialog.Content></Dialog.Portal>
  </Dialog.Root>
}

export default function JobDetail({ sample = false }: { sample?: boolean }) {
  const { jobId, caseName } = useParams()
  const job = useQuery({ queryKey: ['job', jobId], queryFn: () => api.job(jobId!), enabled: !sample && !!jobId,
    refetchInterval: query => ['queued', 'processing'].includes(query.state.data?.status ?? '') ? 3000 : false })
  const samples = useQuery({ queryKey: ['samples'], queryFn: api.samples, enabled: sample })
  const selected = sample ? samples.data?.items.find(item => item.id === caseName) : undefined
  const data = sample ? selected : job.data
  if ((sample ? samples.isLoading : job.isLoading)) return <div className={styles.loadingCard}>작업 정보를 불러오고 있습니다…</div>
  if (!data) return <div className={styles.emptyState}><AlertCircle size={28} /><strong>작업을 찾지 못했습니다.</strong><Link to="/">작업 목록으로 돌아가기</Link></div>

  const settings = data.settings
  const status = sample ? 'complete' : job.data!.status
  const result = sample ? selected : job.data!.result
  const links: Links = sample
    ? { suggested_links: selected?.verified_loop_links ?? null, entered_links: null, effective_links: selected?.verified_loop_links ?? null,
        source: selected?.verified_loop_links ? 'image' : 'unknown', total_length_mm: selected?.verified_loop_links ? selected.verified_loop_links * settings.pitch_mm : null,
        total_length_m: selected?.verified_loop_links ? selected.verified_loop_links * settings.pitch_mm / 1000 : null, disagrees_with_image: false }
    : job.data!.links
  const base = sample ? sampleBase(caseName!) : fileBase(jobId!)
  const complete = status === 'complete'
  const loop = result?.full_loop_verified ?? false
  const title = `${settings.width_mm} / ${settings.pitch_mm} mm`

  return <div className={styles.pageStack}>
    <div className={styles.breadcrumb}><Link to="/"><ArrowLeft size={16} /> 작업 목록</Link><span>/</span><span>{sample ? '샘플 결과' : `작업 ${jobId?.slice(0, 8)}`}</span></div>
    <div className={styles.detailHeading}><div><span className={styles.kicker}>{sample ? 'REFERENCE RESULT' : 'PANORAMA JOB'}</span><h2>{title} <span>폭 / 피치</span></h2><p>{sample ? `${selected?.input_count}장의 촬영 사진으로 생성한 검토용 결과` : `${job.data!.input_count}장 업로드 · ${formatDate(job.data!.created_at)} 생성`}</p></div><div className={styles.detailHeadingRight}><span className={styles.status} data-status={status}>{status === 'complete' ? '전개 완료' : status === 'processing' ? '처리 중' : status === 'queued' ? '대기 중' : '확인 필요'}</span>{complete && <a className={styles.primaryButton} href={`${base}/panorama.png`} download={`track-${sample ? caseName : jobId}.png`}><Download size={17} /> PNG 다운로드</a>}</div></div>

    {!complete && <section className={styles.processingCard}><div className={styles.processingIcon}>{status === 'needs_review' ? <AlertCircle size={26} /> : <Clock3 size={26} />}</div><div><span className={styles.kicker}>{status === 'needs_review' ? 'REVIEW NEEDED' : status === 'queued' ? 'WAITING IN QUEUE' : 'PROCESSING'}</span><h3>{status === 'needs_review' ? '연결 정보를 확인해 주세요' : status === 'queued' ? '작업 순서를 기다리고 있습니다' : '사진을 분석하고 있습니다'}</h3><p>{sample ? '' : job.data!.message}</p>{status === 'queued' && job.data!.queue_position && <small>대기 순서 {job.data!.queue_position}번 · 완료되면 이 화면에 자동으로 결과가 표시됩니다.</small>}</div>{status !== 'needs_review' && <span className={styles.processingPulse} />}</section>}

    <div className={styles.detailGrid}>
      <div className={styles.detailMain}>
        {complete ? <TrackViewer base={base} hasTiles={result?.has_deepzoom ?? (sample ? selected!.has_deepzoom : false)} />
          : <div className={styles.placeholderViewer}><ScanLine size={35} /><strong>전개 사진을 준비하고 있습니다.</strong><span>분석이 끝나면 여기에서 확대하여 확인할 수 있습니다.</span></div>}
        {complete && !sample && <div className={styles.damageEntry}><span className={styles.damageEntryIcon}><ScanSearch size={22} /></span><div><span className={styles.kicker}>REGION EDITOR</span><strong>사진 위에 분류 영역을 지정하세요</strong><p>분류 체계와 세부 섹션을 만들고, 형상을 그려 피치 반복 여부를 선택할 수 있습니다.</p></div><Link to={`/jobs/${jobId}/analysis`}>영역 편집 열기 <ArrowRight size={16} /></Link></div>}
        {complete && <div className={styles.qualityNote}><CheckCircle2 size={18} /><div><strong>{loop ? '한 바퀴의 중복 구간을 확인했습니다.' : '촬영된 구간의 결과입니다.'}</strong><p>{loop ? '첫 사진과 마지막 사진의 동일한 표면을 찾아 겹친 부분을 제거했습니다.' : '첫 사진과 마지막 사진으로 한 바퀴 폐합을 확인하지 못해 전체 링크 수는 제안하지 않습니다.'}</p></div></div>}
      </div>
      <aside className={styles.detailAside}>
        <div className={styles.insightCard}><div className={styles.insightHead}><div><span className={styles.kicker}>PRODUCT DIMENSIONS</span><h3>제품 치수</h3></div><ScanLine size={20} /></div><div className={styles.metricSplit}><div><span>트랙 폭</span><strong>{formatNumber(settings.width_mm)} <small>mm</small></strong></div><div><span>피치</span><strong>{formatNumber(settings.pitch_mm)} <small>mm</small></strong></div></div>
          <div className={styles.linkMetric}><div><span>사진에서 제안한 링크 수</span><strong>{links.suggested_links === null ? '확인 불가' : `${links.suggested_links} 링크`}</strong></div><small>{links.suggested_links === null ? '전체 한 바퀴가 확인된 사진이 필요합니다.' : '사진에서 한 바퀴 폐합을 확인한 추정값'}</small></div>
          <div className={styles.effectiveMetric}><div><span>적용 링크 수</span><strong>{links.effective_links === null ? '미입력' : `${links.effective_links} 링크`}</strong></div>{!sample && <LinkEditor jobId={jobId!} links={links} pitch={settings.pitch_mm} />}</div>
          {links.disagrees_with_image && <div className={styles.conflictNote}><AlertCircle size={16} /><span>입력값이 사진 추정값과 다릅니다. 총 길이는 입력값으로 계산하며, 사진의 전개 범위는 바뀌지 않습니다.</span></div>}
          <div className={styles.lengthResult}><span>계산된 트랙 총 길이</span><strong>{links.total_length_mm === null ? '—' : formatNumber(links.total_length_mm)} <small>mm</small></strong><p>{links.effective_links === null ? '링크 수가 확인되거나 입력되면 표시됩니다.' : `${formatNumber(settings.pitch_mm)} mm × ${links.effective_links} 링크 = ${formatNumber(links.total_length_m!)} m`}</p></div>
        </div>
        <div className={styles.infoCard}><span className={styles.kicker}>JOB DETAILS</span><h3>작업 정보</h3><div><span>업로드 사진</span><strong>{data.input_count}장</strong></div><div><span>전개 범위</span><strong>{loop ? '한 바퀴 확인' : '사진에 보인 구간'}</strong></div>{complete && result?.output_size_wh && <div><span>이미지 해상도</span><strong>{formatNumber(result.output_size_wh[0])} × {formatNumber(result.output_size_wh[1])} px</strong></div>}{!sample && complete && job.data!.result?.skipped_exact_duplicates.length ? <div><span>완전 중복 파일</span><strong>{job.data!.result.skipped_exact_duplicates.length}장 제외</strong></div> : null}</div>
        {complete && <div className={styles.exportCard}><span className={styles.kicker}>EXPORT FILES</span><h3>결과 파일</h3><a href={`${base}/panorama.png`} download><ImageIcon size={17} /><span>원본 크기 PNG</span><Download size={16} /></a><a href={`${base}/panorama.jpg`} download><ImageIcon size={17} /><span>가벼운 JPG</span><Download size={16} /></a><a href={`${base}/quality_report.json`} download><FileJson2 size={17} /><span>분석 보고서</span><Download size={16} /></a></div>}
      </aside>
    </div>
    <div className={styles.bottomCallout}><div><Check size={17} /><span>완료된 결과는 팀의 작업 목록에서 다시 열 수 있습니다.</span></div><Link to="/new">다른 제품 작업하기 <ArrowRight size={16} /></Link></div>
  </div>
}
