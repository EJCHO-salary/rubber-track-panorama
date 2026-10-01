import type { Point } from './types'

type Rect = [number, number, number, number]

function insideRect([x, y]: Point, [left, top, right, bottom]: Rect) {
  return x >= left && x <= right && y >= top && y <= bottom
}

function insidePolygon([x, y]: Point, polygon: Point[]) {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [ax, ay] = polygon[i], [bx, by] = polygon[j]
    if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) inside = !inside
  }
  return inside
}

function segmentsCross(a: Point, b: Point, c: Point, d: Point) {
  const cross = (p: Point, q: Point, r: Point) =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
  const ab1 = cross(a, b, c), ab2 = cross(a, b, d)
  const cd1 = cross(c, d, a), cd2 = cross(c, d, b)
  const between = (p: Point, q: Point, r: Point) =>
    r[0] >= Math.min(p[0], q[0]) && r[0] <= Math.max(p[0], q[0]) &&
    r[1] >= Math.min(p[1], q[1]) && r[1] <= Math.max(p[1], q[1])
  if (ab1 === 0 && between(a, b, c) || ab2 === 0 && between(a, b, d) ||
      cd1 === 0 && between(c, d, a) || cd2 === 0 && between(c, d, b)) return true
  return ab1 * ab2 < 0 && cd1 * cd2 < 0
}

export function polygonIntersectsRect(polygon: Point[], rect: Rect) {
  if (!polygon.length) return false
  if (polygon.some(point => insideRect(point, rect))) return true
  const [left, top, right, bottom] = rect
  const corners: Point[] = [[left, top], [right, top], [right, bottom], [left, bottom]]
  if (corners.some(point => insidePolygon(point, polygon))) return true
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length]
    for (let j = 0; j < 4; j++) {
      if (segmentsCross(a, b, corners[j], corners[(j + 1) % 4])) return true
    }
  }
  return false
}

export function toggleSelection(current: string[], hits: string[], shift: boolean) {
  if (!shift) return [...new Set(hits)]
  const next = new Set(current)
  for (const id of hits) next.has(id) ? next.delete(id) : next.add(id)
  return [...next]
}
