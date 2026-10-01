import assert from 'node:assert/strict'
import { test } from 'node:test'
import { polygonIntersectsRect, toggleSelection } from '../src/crackSelection.ts'

test('marquee selects painted polygon intersections rather than a wide bounding box', () => {
  const thinDiagonal = [[0, 0], [2, 0], [100, 100], [98, 100]]
  assert.equal(polygonIntersectsRect(thinDiagonal, [60, 0, 70, 10]), false)
  assert.equal(polygonIntersectsRect(thinDiagonal, [48, 48, 52, 52]), true)
  assert.equal(polygonIntersectsRect([[20, 20], [80, 20], [80, 80], [20, 80]], [40, 40, 45, 45]), true)
})

test('Shift toggles each hit while ordinary drag replaces selection', () => {
  assert.deepEqual(toggleSelection(['a', 'b'], ['b', 'c'], true), ['a', 'c'])
  assert.deepEqual(toggleSelection(['a', 'b'], ['c'], false), ['c'])
  assert.deepEqual(toggleSelection(['a'], ['a'], true), [])
})
