import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, Check, Clock3, Files, Image as ImageIcon, ScanLine } from 'lucide-react'
import { Link } from 'react-router'
import { api, formatDate, formatNumber, sampleBase } from '../api'
import type { JobStatus } from '../types'
import JobDeleteMenu from '../components/JobDeleteMenu'
import styles from '../App.module.css'

const statusLabel: Record<JobStatus, string> = {
  queued: '대기 중', processing: '처리 중', complete: '완료', needs_review: '확인 필요',
}

export default function Dashboard() {
  const [notice, setNotice] = useState('')
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 5000 })
  const samples = useQuery({ queryKey: ['samples'], queryFn: api.samples })
  const items = jobs.data?.items ?? []
  const complete = items.filter(job => job.status === 'complete').length
  const active = items.filter(job => job.status === 'queued' || job.status === 'processing').length

  return <div className={styles.pageStack}>
    <section className={styles.hero}>
      <div className={styles.heroContent}>
        <span className={styles.heroEyebrow}><ScanLine size={15} /> TRACK SURFACE WORKFLOW</span>
        <h2>여러 장의 촬영을<br /><em>하나의 정확한 흐름으로.</em></h2>
        <p>트랙 폭과 피치를 기준으로 외곽을 맞추고 중복된 구간을 연결합니다. 한 바퀴가 확인되면 링크 수와 총 길이를 제안합니다.</p>
        <Link to="/new" className={styles.heroButton}>새 전개 작업 시작 <ArrowRight size={18} /></Link>
      </div>
      <div className={styles.heroVisual} aria-hidden="true">
        <div className={styles.heroVisualTop}><span>PRODUCT SURFACE / 01</span><span>● PROCESS READY</span></div>
        <div className={styles.heroTrack}>
          {Array.from({ length: 8 }, (_, i) => <span key={i}><b /></span>)}
        </div>
        <div className={styles.heroVisualBottom}><span>WIDTH → PITCH → PANORAMA</span><span>01 / 03</span></div>
      </div>
    </section>

    <section className={styles.stats} aria-label="작업 현황">
      <div className={styles.statCard}><span className={styles.statIcon}><Files size={20} /></span><div><small>전체 작업</small><strong>{jobs.isLoading ? '—' : formatNumber(jobs.data?.total ?? 0)}</strong></div><span>누적</span></div>
      <div className={styles.statCard}><span className={styles.statIcon}><Clock3 size={20} /></span><div><small>진행 및 대기</small><strong>{jobs.isLoading ? '—' : formatNumber(active)}</strong></div><span>작업 중</span></div>
      <div className={styles.statCard}><span className={styles.statIcon}><Check size={20} /></span><div><small>결과 완료</small><strong>{jobs.isLoading ? '—' : formatNumber(complete)}</strong></div><span>다운로드 가능</span></div>
    </section>

    <section className={styles.section}>
      <div className={styles.sectionHead}><div><span className={styles.kicker}>RECENT ACTIVITY</span><h2>최근 전개 작업</h2><p>팀에서 생성한 작업과 현재 처리 상태를 확인합니다.</p></div><Link to="/new" className={styles.textAction}>새 작업 만들기 <ArrowRight size={17} /></Link></div>
      {notice && <p className={styles.actionNotice} role="status">{notice}</p>}
      <div className={styles.tableCard}>
        {jobs.isError ? <div className={styles.emptyState}>작업 목록을 불러오지 못했습니다. 서버 연결을 확인해 주세요.</div>
          : items.length === 0 ? <div className={styles.emptyState}><ImageIcon size={28} /><strong>아직 전개 작업이 없습니다.</strong><span>제품 규격과 사진을 넣어 첫 결과를 만들어 보세요.</span><Link to="/new">작업 시작하기 <ArrowRight size={16} /></Link></div>
          : <><div className={styles.tableHeader}><span>작업</span><span>사진</span><span>링크 수</span><span>상태</span><span>생성일</span><span /><span>메뉴</span></div>{items.map(job =>
            <div className={styles.tableRow} key={job.id}>
              <Link to={`/jobs/${job.id}`} className={styles.rowMain}>
                <span className={styles.jobIdentity}><span className={styles.jobThumb}><ScanLine size={18} /></span><span><strong>{job.settings?.width_mm} / {job.settings?.pitch_mm} mm</strong><small>폭 / 피치 · {job.id.slice(0, 8)}</small></span></span>
                <span>{job.input_count ?? '—'}장</span>
                <span>{job.links.effective_links === null ? '미확인' : `${job.links.effective_links}링크`}</span>
                <span><b className={styles.status} data-status={job.status}>{statusLabel[job.status]}</b></span>
                <span className={styles.muted}>{formatDate(job.created_at)}</span><ArrowRight size={17} />
              </Link>
              <JobDeleteMenu job={job} onDeleted={setNotice} />
            </div>)}</>}
      </div>
    </section>

    {samples.data && samples.data.items.length > 0 && <section className={styles.section}>
      <div className={styles.sectionHead}><div><span className={styles.kicker}>REFERENCE WORK</span><h2>검토용 샘플</h2><p>기존 촬영 세트의 합성 결과를 열어볼 수 있습니다.</p></div></div>
      <div className={styles.sampleGrid}>{samples.data.items.map(sample => <Link to={`/samples/${sample.id}`} className={styles.sampleCard} key={sample.id}>
        <div className={styles.sampleImage}><img src={`${sampleBase(sample.id)}/review.jpg`} alt={`${sample.id} 전개 결과 일부`} loading="lazy" /></div>
        <div className={styles.sampleContent}><div><span className={styles.kicker}>{sample.id.toUpperCase()} / {sample.input_count} PHOTOS</span><h3>{sample.settings.width_mm} / {sample.settings.pitch_mm} mm</h3><p>{sample.full_loop_verified ? `${sample.verified_loop_links}링크 한 바퀴 추정` : '제공된 사진 범위'}</p></div><ArrowRight size={18} /></div>
      </Link>)}</div>
    </section>}
  </div>
}
