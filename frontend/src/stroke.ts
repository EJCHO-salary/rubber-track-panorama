import type { Point } from './types'

/** A freehand loop is closed only after it returns near its start and encloses a visible span. */
export function strokeCloses(screens: Point[]): boolean {
  if (screens.length < 4) return false
  const first = screens[0], last = screens[screens.length - 1]
  const xs = screens.map(point => point[0]), ys = screens.map(point => point[1])
  const length = screens.slice(1).reduce((sum, point, index) =>
    sum + Math.hypot(point[0] - screens[index][0], point[1] - screens[index][1]), 0)
  return Math.hypot(first[0] - last[0], first[1] - last[1]) <= 24 &&
    Math.max(...xs) - Math.min(...xs) >= 8 && Math.max(...ys) - Math.min(...ys) >= 8 && length >= 50
}
