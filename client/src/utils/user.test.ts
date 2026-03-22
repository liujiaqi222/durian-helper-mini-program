import { describe, expect, it } from 'vitest'
import {
  buildInviteSharePath,
  normalizeInviteCode,
  readInviterCodeFromQuery,
} from './user'

describe('normalizeInviteCode', () => {
  it('trims whitespace and normalizes uppercase invite codes', () => {
    expect(normalizeInviteCode('  invabc1  ')).toBe('INVABC1')
  })
})

describe('readInviterCodeFromQuery', () => {
  it('returns a normalized inviter code from launch query', () => {
    expect(readInviterCodeFromQuery({ inviterCode: '  invabc1 ' })).toBe('INVABC1')
  })

  it('returns an empty string when launch query does not contain an inviter code', () => {
    expect(readInviterCodeFromQuery({ foo: 'bar' })).toBe('')
  })
})

describe('buildInviteSharePath', () => {
  it('builds a share path with the current user invite code', () => {
    expect(buildInviteSharePath('invabc1')).toBe('/pages/index/index?inviterCode=INVABC1')
  })
})
