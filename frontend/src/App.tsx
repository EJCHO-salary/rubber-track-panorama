import { lazy, Suspense, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BriefcaseBusiness, ChevronRight, CircleHelp, Layers3, Plus, ScanLine, X } from 'lucide-react'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router'
import { api, formatDate } from './api'
import JobDeleteMenu from './components/JobDeleteMenu'
import styles from './App.module.css'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const NewJob = lazy(() => import('./pages/NewJob'))
const JobDetail = lazy(() => import('./pages/JobDetail'))

const titles: Record<string, { title: string; eyebrow: string }> = {
  '/': { title: '작업 개요', eyebrow: 'WORKSPACE / OVERVIEW' },
  '/new': { title: '새 전개 작업', eyebrow: 'WORKSPACE / NEW JOB' },
}

export default function App() {
  const [sidebarNotice, setSidebarNotice] = useState('')
  const location = useLocation()
  const page = titles[location.pathname] ?? (location.pathname.startsWith('/samples/')
    ? { title: '샘플 결과', eyebrow: 'WORKSPACE / SAMPLE' }
    : { title: '작업 결과', eyebrow: 'WORKSPACE / JOB DETAIL' })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 5000 })
  const recent = jobs.data?.items.slice(0, 5) ?? []

  return <div className={styles.shell}>
    <aside className={styles.sidebar} aria-label="주 메뉴">
      <Link to="/" className={styles.brand} aria-label="Track Studio 홈">
        <span className={styles.brandMark}><ScanLine size={23} strokeWidth={2.1} /></span>
        <span><strong>TRACK STUDIO</strong><small>Surface imaging workspace</small></span>
      </Link>
      <div className={styles.sidebarLabel}>WORKSPACE</div>
      <nav className={styles.nav}>
        <NavLink to="/" end className={({ isActive }) => isActive ? styles.navActive : styles.navLink}>
          <BriefcaseBusiness size={18} /> 작업 목록 <ChevronRight size={15} className={styles.chevron} />
        </NavLink>
        <NavLink to="/new" className={({ isActive }) => isActive ? styles.navActive : styles.navLink}>
          <Plus size={18} /> 새 작업 만들기 <ChevronRight size={15} className={styles.chevron} />
        </NavLink>
      </nav>
      <div className={styles.sidebarLabel}>최근 작업</div>
      <div className={styles.recent}>
        {recent.length === 0 ? <p className={styles.sidebarEmpty}>아직 생성된 작업이 없습니다.</p> : recent.map(job =>
          <div className={styles.recentRow} key={job.id}>
            <Link to={`/jobs/${job.id}`} className={styles.recentLink}>
              <span className={styles.recentDot} data-status={job.status} />
              <span><strong>{job.settings.width_mm} / {job.settings.pitch_mm} mm</strong><small>{formatDate(job.created_at)}</small></span>
            </Link>
            <JobDeleteMenu job={job} sidebar onDeleted={setSidebarNotice} />
          </div>) }
      </div>
      <div className={styles.sidebarFoot}><Layers3 size={16} /> 팀 공용 작업 공간 <span>INTERNAL</span></div>
    </aside>
    {sidebarNotice && <div className={styles.sidebarToast} role="status"><span>{sidebarNotice}</span><button aria-label="알림 닫기" onClick={() => setSidebarNotice('')}><X size={16} /></button></div>}
    <div className={styles.content}>
      <header className={styles.topbar}>
        <div><div className={styles.eyebrow}>{page.eyebrow}</div><h1>{page.title}</h1></div>
        <div className={styles.topActions}><span className={styles.internalBadge}>● 사내 작업 공간</span><Link to="/new" className={styles.topCreate}><Plus size={17} /> 새 작업</Link></div>
      </header>
      <main className={styles.main}>
        <Suspense fallback={<div className={styles.loadingCard}>화면을 준비하고 있습니다…</div>}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/new" element={<NewJob />} />
          <Route path="/jobs/:jobId" element={<JobDetail />} />
          <Route path="/samples/:caseName" element={<JobDetail sample />} />
          <Route path="*" element={<div className={styles.notFound}><CircleHelp size={26} /><h2>페이지를 찾지 못했습니다.</h2><Link to="/">작업 목록으로 돌아가기</Link></div>} />
        </Routes>
        </Suspense>
      </main>
    </div>
  </div>
}
