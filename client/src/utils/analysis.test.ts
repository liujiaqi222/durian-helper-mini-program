import { describe, expect, it } from 'vitest'
import type { AnalysisResult, AnalysisTaskItem } from '../types/analysis'
import {
  findRecommendedItem,
  getStatusDescription,
  isTerminalTaskStatus,
} from './analysis'

function createItem(overrides: Partial<AnalysisTaskItem>): AnalysisTaskItem {
  return {
    label: 'A',
    score: 80,
    summary: '成熟度较好',
    reasons: ['果形饱满'],
    risks: ['建议尽快食用'],
    buyPriority: 2,
    cropImageUrl: null,
    ...overrides,
  }
}

describe('analysis utils', () => {
  it('maps backend task status to readable copy', () => {
    expect(getStatusDescription('PENDING')).toBe('任务已创建，准备开始分析')
    expect(getStatusDescription('DETECTING')).toBe('正在识别榴莲位置和编号')
    expect(getStatusDescription('SCORING')).toBe('正在生成评分和购买建议')
    expect(getStatusDescription('DONE')).toBe('分析完成')
    expect(getStatusDescription('FAILED')).toBe('分析失败，请重试')
  })

  it('finds the recommended item by recommended label first', () => {
    const result: AnalysisResult = {
      sourceImageUrl: 'https://example.com/source.jpg',
      annotatedImageUrl: 'https://example.com/annotated.jpg',
      recommendedLabel: 'B',
      aiSummary: '推荐 B',
      items: [
        createItem({ label: 'A', buyPriority: 2 }),
        createItem({ label: 'B', buyPriority: 3, score: 92 }),
      ],
    }

    expect(findRecommendedItem(result)?.label).toBe('B')
  })

  it('falls back to the smallest buy priority and highest score', () => {
    const result: AnalysisResult = {
      sourceImageUrl: 'https://example.com/source.jpg',
      annotatedImageUrl: null,
      recommendedLabel: null,
      aiSummary: null,
      items: [
        createItem({ label: 'A', buyPriority: 3, score: 86 }),
        createItem({ label: 'B', buyPriority: 1, score: 70 }),
        createItem({ label: 'C', buyPriority: 1, score: 90 }),
      ],
    }

    expect(findRecommendedItem(result)?.label).toBe('C')
  })

  it('treats DONE and FAILED as terminal statuses', () => {
    expect(isTerminalTaskStatus('DONE')).toBe(true)
    expect(isTerminalTaskStatus('FAILED')).toBe(true)
    expect(isTerminalTaskStatus('SCORING')).toBe(false)
  })
})
