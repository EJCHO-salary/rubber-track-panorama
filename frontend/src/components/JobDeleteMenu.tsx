import { useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { MoreHorizontal, ScanLine, Trash2, X } from 'lucide-react'
import { api } from '../api'
import type { JobSummary } from '../types'
import styles from '../App.module.css'

type Props = {
  job: JobSummary
  sidebar?: boolean
  onDeleted?: (message: string) => void
}

export default function JobDeleteMenu({ job, sidebar = false, onDeleted }: Props) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const remove = useMutation({
    mutationFn: () => api.deleteJob(job.id),
    onSuccess: result => {
      setOpen(false)
      setError('')
      onDeleted?.(result.pending
        ? '처리 중인 작업을 중단하고 사진 파일을 삭제하고 있습니다.'
        : '작업과 원본·결과 사진을 삭제했습니다.')
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
    onError: cause => setError(cause instanceof Error ? cause.message : '작업을 삭제하지 못했습니다.'),
  })

  return <>
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button className={sidebar ? styles.recentMenuButton : styles.jobMenuButton} aria-label={`${job.id.slice(0, 8)} 작업 메뉴`}>
          <MoreHorizontal size={sidebar ? 18 : 20} />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={styles.jobMenuContent} align="end" sideOffset={5}>
          <DropdownMenu.Item className={styles.jobMenuItem} onSelect={() => { setError(''); setOpen(true) }}>
            <Trash2 size={16} /> 작업 삭제
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>

    <Dialog.Root open={open} onOpenChange={next => { if (!remove.isPending) { setOpen(next); if (!next) setError('') } }}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.dialogOverlay} />
        <Dialog.Content className={styles.dialogContent}>
          <div className={styles.dialogTop}>
            <div><span className={styles.kicker}>DELETE JOB</span><Dialog.Title>작업을 삭제할까요?</Dialog.Title></div>
            <Dialog.Close className={styles.dialogClose} aria-label="닫기" disabled={remove.isPending}><X size={20} /></Dialog.Close>
          </div>
          <Dialog.Description className={styles.dialogDescription}>
            {job.settings.width_mm} / {job.settings.pitch_mm} mm 작업의 업로드 원본, 전개 결과, 확대 타일, 분석 보고서를 서버에서 영구 삭제합니다. 삭제 후 복구할 수 없습니다.
          </Dialog.Description>
          <div className={styles.deleteJobSummary}>
            <ScanLine size={18} />
            <div><strong>{job.settings.width_mm} / {job.settings.pitch_mm} mm</strong><small>{job.input_count}장 · 작업 {job.id.slice(0, 8)}</small></div>
          </div>
          {error && <p className={styles.formError} role="alert">{error}</p>}
          <div className={styles.deleteDialogActions}>
            <Dialog.Close className={styles.ghostButton} disabled={remove.isPending}>취소</Dialog.Close>
            <button className={styles.dangerButton} disabled={remove.isPending} onClick={() => remove.mutate()}>
              <Trash2 size={16} /> {remove.isPending ? '삭제 중…' : '작업과 사진 삭제'}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </>
}
