import assert from 'node:assert/strict'
import { test } from 'node:test'
import { strokeCloses } from '../src/stroke.ts'

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
