import { Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { useEffect, useRef, useState } from 'react'
import { getAnalysisResult, getAnalysisTask, retryAnalysisTask } from '../../services/api'
import { useAnalysisStore } from '../../store/analysis'
import { useUserStore } from '../../store/user'
import type { AnalysisTaskDetectionItem } from '../../types/analysis'
import {
  findRecommendedItem,
  getInProgressMessages,
  getStatusDescription,
  isTerminalTaskStatus,
  resolveResultPreview,
  sortItemsForDisplay,
} from '../../utils/analysis'

const POLL_INTERVAL = 1500
const MAX_POLL_ATTEMPTS = 40

export default function ResultPage() {
  const defaultOverlayColor = '#4B5563'

  const taskId = useAnalysisStore((state) => state.taskId)
  const taskStatus = useAnalysisStore((state) => state.taskStatus)
  const taskDetail = useAnalysisStore((state) => state.taskDetail)
  const localImagePath = useAnalysisStore((state) => state.localImagePath)
  const result = useAnalysisStore((state) => state.result)
  const errorMessage = useAnalysisStore((state) => state.errorMessage)
  const setTaskStatus = useAnalysisStore((state) => state.setTaskStatus)
  const setTaskDetail = useAnalysisStore((state) => state.setTaskDetail)
  const setResult = useAnalysisStore((state) => state.setResult)
  const setErrorMessage = useAnalysisStore((state) => state.setErrorMessage)
  const clearErrorMessage = useAnalysisStore((state) => state.clearErrorMessage)
  const resetAnalysis = useAnalysisStore((state) => state.resetAnalysis)
  const profile = useUserStore((state) => state.profile)
  const [isRetrying, setIsRetrying] = useState(false)
  const [previewSize, setPreviewSize] = useState<{ width: number; height: number } | null>(null)
  const pollAttemptRef = useRef(0)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isMountedRef = useRef(true)

  const recommendedItem = result ? findRecommendedItem(result) : null
  const displayItems = sortItemsForDisplay(result?.items || [])
  const highlightedLabel = recommendedItem?.label || null

  useEffect(() => {
    return () => {
      isMountedRef.current = false
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!taskId) {
      void Taro.reLaunch({
        url: '/pages/index/index',
      })
      return
    }

    void syncTaskStatus()
  }, [taskId])

  async function syncTaskStatus() {
    if (!taskId) {
      return
    }

    try {
      clearErrorMessage()
      const nextTask = await getAnalysisTask(taskId)
      setTaskDetail(nextTask)
      setTaskStatus(nextTask.status)

      if (nextTask.status === 'DONE') {
        await loadResult()
        return
      }

      if (nextTask.status === 'FAILED') {
        setErrorMessage(nextTask.errorMessage || '分析失败，请重试')
        return
      }

      scheduleNextPoll()
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '查询任务状态失败')
    }
  }

  function scheduleNextPoll() {
    if (!taskId) {
      return
    }

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    if (pollAttemptRef.current >= MAX_POLL_ATTEMPTS) {
      setTaskStatus('FAILED')
      setErrorMessage('等待分析结果超时，请稍后重试')
      return
    }

    pollAttemptRef.current += 1
    timeoutRef.current = setTimeout(() => {
      void syncTaskStatus()
    }, POLL_INTERVAL)
  }

  async function loadResult() {
    if (!taskId) {
      return
    }

    try {
      const nextResult = await getAnalysisResult(taskId)
      if (!isMountedRef.current) {
        return
      }

      setResult(nextResult)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '获取分析结果失败')
    }
  }

  async function handleRetry() {
    if (!taskId || isRetrying) {
      return
    }

    setIsRetrying(true)
    clearErrorMessage()
    pollAttemptRef.current = 0

    try {
      const nextTask = await retryAnalysisTask(taskId)
      setTaskStatus(nextTask.status)
      scheduleNextPoll()
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '重试失败，请稍后再试')
    } finally {
      setIsRetrying(false)
    }
  }

  async function handleRestart() {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }
    resetAnalysis()
    await Taro.reLaunch({
      url: '/pages/index/index',
    })
  }

  const statusText = taskStatus ? getStatusDescription(taskStatus) : '正在准备结果页配置...'
  const preview = resolveResultPreview({
    sourceImageUrl: result?.sourceImageUrl || taskDetail?.sourceImageUrl || null,
    localImagePath,
  })
  const detectingLabels = result?.items.map((item) => item.label) || taskDetail?.detectedLabels || []
  const inProgressMessages = taskStatus
    ? getInProgressMessages({
        detectedCount: taskDetail?.detectedCount || 0,
        labels: detectingLabels,
        status: taskStatus,
        hasPreview: Boolean(preview),
      })
    : []
  const overlaySourceItems: AnalysisTaskDetectionItem[] = result?.items?.length
    ? result.items
    : taskDetail?.rawResult?.items || []
  const overlayItems = previewSize && overlaySourceItems.length
    ? overlaySourceItems.map((item) => {
        const width = previewSize.width || 1
        const height = previewSize.height || 1
        const boxWidth = Math.max(item.bbox.x2 - item.bbox.x1, 1)
        const boxHeight = Math.max(item.bbox.y2 - item.bbox.y1, 1)

        return {
          heightPercent: (boxHeight / height) * 100,
          isHighlighted: item.label === highlightedLabel,
          item,
          leftPercent: (item.bbox.x1 / width) * 100,
          topPercent: (item.bbox.y1 / height) * 100,
          widthPercent: (boxWidth / width) * 100,
        }
      })
    : []

  function handlePreviewLoad(event: { detail?: { width?: number | string; height?: number | string } }) {
    const width = Number(event.detail?.width)
    const height = Number(event.detail?.height)

    if (!width || !height) {
      return
    }

    setPreviewSize({ width, height })
  }

  useEffect(() => {
    setPreviewSize(null)
  }, [preview?.imageUrl])

  return (
    <View className='min-h-screen bg-gradient-to-br from-yellow-50 via-white to-amber-50 px-4 py-6 text-gray-800 pb-12'>
      <View className='flex flex-col gap-5'>
        {/* Header Title */}
        <View className='flex flex-col gap-1 rounded-3xl bg-white p-6 shadow-sm border border-yellow-100'>
          <Text className='text-xs font-bold uppercase tracking-widest text-amber-500'>Analysis Result</Text>
          <Text className='text-2xl font-extrabold leading-tight text-gray-900'>识别与评分结果</Text>
          <Text className='text-sm leading-relaxed text-gray-500 mt-1'>{statusText}</Text>
          {profile ? (
            <Text className='mt-1 text-sm leading-relaxed text-amber-600'>
              当前剩余识别次数：{profile.remainingCredits}
            </Text>
          ) : null}
        </View>

        {preview ? (
          <View className='flex flex-col gap-3 rounded-3xl bg-white p-4 shadow-sm border border-yellow-100'>
            <Text className='px-2 text-lg font-bold text-gray-900'>{preview.title}</Text>
            <View className='relative overflow-hidden rounded-2xl bg-gray-50 ring-1 ring-gray-100'>
              <Image
                className='block w-full'
                mode='widthFix'
                onLoad={handlePreviewLoad}
                src={preview.imageUrl}
              />
              {overlayItems.length > 0 ? (
                <View className='absolute inset-0'>
                  {overlayItems.map((entry) => (
                    <View
                      key={`${entry.item.label}-overlay`}
                      className='absolute box-border flex items-center justify-center rounded-[18px]'
                      style={{
                        border: entry.isHighlighted ? '4px solid #f59e0b' : `2px solid ${defaultOverlayColor}`,
                        boxShadow: 'none',
                        left: `${entry.leftPercent}%`,
                        top: `${entry.topPercent}%`,
                        width: `${entry.widthPercent}%`,
                        height: `${entry.heightPercent}%`,
                      }}
                    >
                      <View
                        className='flex items-center justify-center'
                        style={{
                          backgroundColor: 'transparent',
                        }}
                      >
                        <Text
                          className='text-2xl font-extrabold leading-none'
                          style={{
                            color: entry.isHighlighted ? '#f59e0b' : defaultOverlayColor,
                          }}
                        >
                          {entry.item.label}
                        </Text>
                      </View>
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
          </View>
        ) : null}

        {taskStatus && !isTerminalTaskStatus(taskStatus) ? (
          <View className='flex flex-col gap-3 rounded-3xl bg-amber-50/50 p-6 shadow-sm border border-amber-200'>
            <View className='flex items-center gap-2'>
              <View className='h-3 w-3 rounded-full bg-amber-400 animate-pulse' />
              <Text className='text-lg font-bold text-amber-900'>分析进行中</Text>
            </View>
            <Text className='text-sm leading-relaxed text-amber-700/80'>请稍候，我们正在为您仔细检查每一个榴莲。</Text>
            {inProgressMessages.map((message, index) => (
              <View key={message} className={index === 0 ? 'mt-2 rounded-xl bg-white/60 p-3' : 'mt-1 rounded-xl bg-white/60 p-3'}>
                <Text className='text-sm font-medium text-amber-800'>{message}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {recommendedItem ? (
          <View className='flex flex-col items-center gap-2 rounded-3xl bg-gradient-to-b from-amber-100 to-amber-50 p-8 text-center shadow-md border border-amber-200'>
            <Text className='text-sm font-bold tracking-wide text-amber-700'>🌟 最佳推荐</Text>
            <Text className='text-6xl font-black text-amber-600 drop-shadow-sm my-2'>{recommendedItem.label}</Text>
            <View className='rounded-full bg-white/80 px-4 py-1 shadow-sm'>
              <Text className='text-lg font-bold text-amber-950'>
                {recommendedItem.score !== null ? `${recommendedItem.score} 综合评分` : '暂无确切分数'}
              </Text>
            </View>
            <Text className='mt-3 text-sm leading-relaxed text-amber-800 font-medium'>
              {recommendedItem.summary || result?.overallSummary || '后端暂未返回总体评价'}
            </Text>
          </View>
        ) : null}

        {result?.overallSummary && !recommendedItem ? (
          <View className='flex flex-col gap-2 rounded-3xl bg-white p-6 shadow-sm border border-yellow-100'>
            <Text className='text-lg font-bold text-gray-900'>整体建议</Text>
            <Text className='text-sm leading-relaxed text-gray-600'>{result.overallSummary}</Text>
          </View>
        ) : null}

        {displayItems.length > 0 ? (
          <View className='flex flex-col gap-4 mt-2'>
            <Text className='px-2 text-lg font-bold text-gray-900'>所有榴莲明细</Text>
            <View className='flex flex-col gap-4'>
              {displayItems.map((item) => (
                <View
                  className='flex flex-col gap-3 rounded-3xl bg-white p-5 shadow-sm border'
                  key={item.label}
                  style={{
                    borderColor: item.label === highlightedLabel ? '#f59e0b' : 'rgba(254, 243, 199, 0.7)',
                    boxShadow: item.label === highlightedLabel ? '0 16px 32px rgba(217, 119, 6, 0.12)' : undefined,
                  }}
                >
                  <View className='flex items-center justify-between border-b border-gray-100 pb-3'>
                    <View className='flex items-center gap-2'>
                      <Text className='text-xl font-extrabold text-gray-900'>{item.label}</Text>
                      {item.buyPriority === 1 ? (
                        <Text className='rounded-md bg-green-100 px-2 py-0.5 text-xs font-bold text-green-700'>首选推荐</Text>
                      ) : item.buyPriority !== null ? (
                        <Text className='rounded-md bg-amber-100 px-2 py-0.5 text-xs font-bold text-amber-900'>顺位 {item.buyPriority}</Text>
                      ) : null}
                    </View>
                    <Text className='text-lg font-black text-amber-500'>
                      {item.score !== null ? `${item.score}分` : '--'}
                    </Text>
                  </View>

                  {item.summary ? <Text className='text-sm leading-relaxed text-gray-600'>{item.summary}</Text> : null}
                  
                  
                  {item.reasons?.length ? (
                    <View className='flex flex-wrap gap-2 mt-1'>
                      {item.reasons.map((reason) => (
                        <Text
                          className='rounded-xl bg-amber-50/80 px-3 py-1 text-xs font-medium text-amber-700 border border-amber-100'
                          key={`${item.label}-reason-${reason}`}
                        >
                          👍 {reason}
                        </Text>
                      ))}
                    </View>
                  ) : null}

                  {item.risks?.length ? (
                    <View className='flex flex-wrap gap-2'>
                      {item.risks.map((risk) => (
                        <Text
                          className='rounded-xl bg-red-50/80 px-3 py-1 text-xs font-medium text-red-600 border border-red-100'
                          key={`${item.label}-risk-${risk}`}
                        >
                          ⚠️ {risk}
                        </Text>
                      ))}
                    </View>
                  ) : null}
                </View>
              ))}
            </View>
          </View>
        ) : null}

        {errorMessage ? (
          <View className='flex flex-col gap-2 rounded-2xl bg-red-50 p-5 border border-red-100 mt-2'>
            <Text className='text-base font-bold text-red-600'>任务异常停止</Text>
            <Text className='text-sm leading-relaxed text-red-500'>{errorMessage}</Text>
          </View>
        ) : null}

        <View className='flex flex-col gap-3 mt-4'>
          {taskStatus === 'FAILED' ? (
            <View
              className={`flex w-full items-center justify-center rounded-2xl py-3 text-base font-bold transition-all ${
                isRetrying
                  ? 'bg-amber-300 text-white opacity-70 cursor-not-allowed'
                  : 'bg-amber-500 text-white shadow-sm active:scale-95 active:bg-amber-600'
              }`}
              onClick={isRetrying ? undefined : handleRetry}
            >
              <Text>{isRetrying ? '重试中...' : '重试分析任务'}</Text>
            </View>
          ) : null}

          <View
            className='flex w-full items-center justify-center rounded-2xl border border-amber-200 bg-white py-3 text-base font-bold text-amber-700 shadow-sm transition-all active:scale-95 active:bg-amber-50'
            onClick={handleRestart}
          >
            <Text>返回首页，重新挑选</Text>
          </View>
        </View>
      </View>
    </View>
  )
}
