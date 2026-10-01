import assert from 'node:assert/strict'
import { test } from 'node:test'
import { placeStrokeControls, strokeCloses } from '../src/stroke.ts'

test('a line remains open, even when longer than a closed loop', () => {
  assert.equal(strokeCloses([[10, 10], [10, 60], [10, 120], [10, 180]]), false)
})

test('returning near the start closes a freehand damage boundary', () => {
  assert.equal(strokeCloses([[10, 10], [80, 10], [80, 80], [10, 80], [14, 14]]), true)
})

test('tiny movements and simple retracing do not create an area', () => {
  assert.equal(strokeCloses([[10, 10], [13, 10], [13, 13], [10, 13], [11, 11]]), false)
  assert.equal(strokeCloses([[10, 10], [11, 50], [10, 100], [11, 10]]), false)
})

test('confirmation stays beside the mark and flips away from the viewport edge', () => {
  const viewport = { width: 800, height: 600 }, panel = { width: 244, height: 98 }
  const right = placeStrokeControls([[200, 200], [230, 320]], viewport, panel)
  assert.ok(right[0] > 230)
  const left = placeStrokeControls([[740, 200], [760, 320]], viewport, panel)
  assert.ok(left[0] + panel.width < 740)
  const above = placeStrokeControls([[200, 500], [650, 540]], viewport, panel)
  assert.ok(above[1] + panel.height < 500)
  const below = placeStrokeControls([[200, 10], [650, 30]], viewport, panel)
  assert.ok(below[1] > 30)
})
