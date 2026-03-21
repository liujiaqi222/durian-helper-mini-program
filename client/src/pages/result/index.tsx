import { Button, Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { useEffect, useRef, useState } from 'react'
import { getAnalysisResult, getAnalysisTask, retryAnalysisTask } from '../../services/api'
import { useAnalysisStore } from '../../store/analysis'
import {
  findRecommendedItem,
  getStatusDescription,
  isTerminalTaskStatus,
  resolveResultPreview,
  sortItemsForDisplay,
} from '../../utils/analysis'

const POLL_INTERVAL = 1500
const MAX_POLL_ATTEMPTS = 40

export default function ResultPage() {
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
  const [isRetrying, setIsRetrying] = useState(false)
  const pollAttemptRef = useRef(0)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isMountedRef = useRef(true)

  const recommendedItem = result ? findRecommendedItem(result) : null
  const displayItems = sortItemsForDisplay(result?.items || [])

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
      void Taro.redirectTo({
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
    resetAnalysis()
    await Taro.redirectTo({
      url: '/pages/index/index',
    })
  }

  const statusText = taskStatus ? getStatusDescription(taskStatus) : '正在准备结果页配置...'
  const annotatedImageUrl = result?.annotatedImageUrl || taskDetail?.annotatedImageUrl || null
  const preview = resolveResultPreview({
    annotatedImageUrl,
    sourceImageUrl: result?.sourceImageUrl || taskDetail?.sourceImageUrl || null,
    localImagePath,
  })
  const detectingLabels = result?.items.map((item) => item.label) || taskDetail?.detectedLabels || []

  return (
    <View className='min-h-screen bg-gradient-to-br from-yellow-50 via-white to-amber-50 px-4 py-6 text-gray-800 pb-12'>
      <View className='flex flex-col gap-5'>
        {/* Header Title */}
        <View className='flex flex-col gap-1 rounded-3xl bg-white p-6 shadow-sm border border-yellow-100'>
          <Text className='text-xs font-bold uppercase tracking-widest text-amber-500'>Analysis Result</Text>
          <Text className='text-2xl font-extrabold leading-tight text-gray-900'>识别与评分结果</Text>
          <Text className='text-sm leading-relaxed text-gray-500 mt-1'>{statusText}</Text>
        </View>

        {preview ? (
          <View className='flex flex-col gap-3 rounded-3xl bg-white p-4 shadow-sm border border-yellow-100'>
            <Text className='px-2 text-lg font-bold text-gray-900'>{preview.title}</Text>
            <Image className='w-full overflow-hidden rounded-2xl bg-gray-50 ring-1 ring-gray-100' mode='widthFix' src={preview.imageUrl} />
          </View>
        ) : null}

        {taskStatus && !isTerminalTaskStatus(taskStatus) ? (
          <View className='flex flex-col gap-3 rounded-3xl bg-amber-50/50 p-6 shadow-sm border border-amber-200'>
            <View className='flex items-center gap-2'>
              <View className='h-3 w-3 rounded-full bg-amber-400 animate-pulse' />
              <Text className='text-lg font-bold text-amber-900'>分析进行中</Text>
            </View>
            <Text className='text-sm leading-relaxed text-amber-700/80'>请稍候，我们正在为您仔细检查每一个榴莲。</Text>
            {taskDetail?.detectedCount ? (
              <View className='mt-2 rounded-xl bg-white/60 p-3'>
                <Text className='text-sm font-medium text-amber-800'>
                  当前进展：已识别 {taskDetail.detectedCount} 个榴莲 ({detectingLabels.join('、')})
                </Text>
              </View>
            ) : null}
            {taskStatus === 'SCORING' && annotatedImageUrl ? (
              <View className='mt-1 rounded-xl bg-white/60 p-3'>
                <Text className='text-sm font-medium text-amber-800'>
                  当前进展：已完成编号，正在为 {detectingLabels.join('、') || '当前目标'} 综合打分...
                </Text>
              </View>
            ) : null}
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
                <View className='flex flex-col gap-3 rounded-3xl bg-white p-5 shadow-sm border border-yellow-100/50' key={item.label}>
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
                  
                  <View className='flex items-center gap-2 text-xs text-gray-400'>
                    <Text>置信区间：{Math.round(item.confidence * 100)}%</Text>
                    <Text className='scale-75 text-gray-300'>|</Text>
                    <Text>坐标：({item.bbox.x1}, {item.bbox.y1})</Text>
                  </View>

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
            <Button
              className='w-full rounded-2xl bg-amber-500 text-base font-bold text-white shadow-md border-none after:border-none'
              loading={isRetrying}
              onClick={handleRetry}
            >
              重试分析任务
            </Button>
          ) : null}

          <Button
            className='w-full rounded-2xl bg-white text-base font-bold text-amber-700 border border-amber-200 shadow-sm after:border-none'
            onClick={handleRestart}
          >
            返回首页，重新鉴定
          </Button>
        </View>
      </View>
    </View>
  )
}
