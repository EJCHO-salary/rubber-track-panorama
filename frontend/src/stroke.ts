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

/** Place a small confirmation panel beside the stroke whenever there is room. */
export function placeStrokeControls(points: Point[], viewport: { width: number; height: number },
                                    panel: { width: number; height: number }): Point {
  const margin = 12, gap = 14
  const xs = points.map(point => point[0]), ys = points.map(point => point[1])
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const clamp = (value: number, maximum: number) => Math.max(margin, Math.min(value, maximum - margin))
  const middleX = (minX + maxX) / 2, middleY = (minY + maxY) / 2
  const sideTop = clamp(middleY - panel.height / 2, viewport.height - panel.height)
  const verticalLeft = clamp(middleX - panel.width / 2, viewport.width - panel.width)
  if (viewport.width - maxX >= panel.width + gap + margin)
    return [maxX + gap, sideTop]
  if (minX >= panel.width + gap + margin)
    return [minX - gap - panel.width, sideTop]
  if (minY >= panel.height + gap + margin)
    return [verticalLeft, minY - gap - panel.height]
  if (viewport.height - maxY >= panel.height + gap + margin)
    return [verticalLeft, maxY + gap]
  // Very large or partially off-screen strokes may leave no non-overlapping spot.
  return [clamp(maxX + gap, viewport.width - panel.width), sideTop]
}
