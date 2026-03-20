import { Button, Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { useEffect, useRef, useState } from 'react'
import { getAnalysisResult, getAnalysisTask, retryAnalysisTask } from '../../services/api'
import { useAnalysisStore } from '../../store/analysis'
import { findRecommendedItem, getStatusDescription, isTerminalTaskStatus, sortItemsForDisplay } from '../../utils/analysis'

const POLL_INTERVAL = 1500
const MAX_POLL_ATTEMPTS = 40

export default function ResultPage() {
  const taskId = useAnalysisStore((state) => state.taskId)
  const taskStatus = useAnalysisStore((state) => state.taskStatus)
  const localImagePath = useAnalysisStore((state) => state.localImagePath)
  const result = useAnalysisStore((state) => state.result)
  const errorMessage = useAnalysisStore((state) => state.errorMessage)
  const setTaskStatus = useAnalysisStore((state) => state.setTaskStatus)
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

  const statusText = taskStatus ? getStatusDescription(taskStatus) : '正在准备结果页'
  const previewImage = result?.sourceImageUrl || localImagePath

  return (
    <View className='min-h-screen bg-gradient-to-b from-[#f8f1e6] to-[#efe4d1] px-6 py-6 text-[#2e2017]'>
      <View className='flex flex-col gap-5'>
        <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
          <Text className='text-[12px] font-bold uppercase tracking-[2px] text-[#a67b4f]'>Analysis Status</Text>
          <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>榴莲识别与评分结果</Text>
          <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>{statusText}</Text>
        </View>

        {previewImage ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>原图</Text>
            <Image className='w-full overflow-hidden rounded-[18px]' mode='widthFix' src={previewImage} />
          </View>
        ) : null}

        {result?.annotatedImageUrl ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>编号标注图</Text>
            <Image className='w-full overflow-hidden rounded-[18px]' mode='widthFix' src={result.annotatedImageUrl} />
          </View>
        ) : null}

        {taskStatus && !isTerminalTaskStatus(taskStatus) ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>分析进行中</Text>
            <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>请稍等，结果页会自动刷新当前任务状态。</Text>
          </View>
        ) : null}

        {recommendedItem ? (
          <View className='flex flex-col items-center gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 text-center shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>推荐榴莲</Text>
            <Text className='text-[64px] font-bold leading-none text-[#7b4b22]'>{recommendedItem.label}</Text>
            <Text className='mt-2 text-[18px] font-semibold text-[#2e2017]'>
              {recommendedItem.score !== null ? `${recommendedItem.score} 分` : '暂无分数'}
            </Text>
            <Text className='mt-3 text-[14px] leading-[1.6] text-[#4d3d31]'>
              {recommendedItem.summary || result?.aiSummary || '后端暂未返回推荐摘要'}
            </Text>
          </View>
        ) : null}

        {result?.aiSummary ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>整体说明</Text>
            <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>{result.aiSummary}</Text>
          </View>
        ) : null}

        {displayItems.length > 0 ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>全部评分列表</Text>
            <View className='flex flex-col gap-4'>
              {displayItems.map((item) => (
                <View className='flex flex-col gap-[10px] rounded-[18px] bg-[#fffdf8] p-[18px]' key={item.label}>
                  <View className='flex items-center justify-between'>
                    <Text className='text-[22px] font-bold text-[#2e2017]'>{item.label}</Text>
                    <Text className='text-[14px] font-semibold text-[#7b4b22]'>
                      {item.score !== null ? `${item.score} 分` : '暂无分数'}
                    </Text>
                  </View>

                  {item.cropImageUrl ? (
                    <Image className='h-[220px] w-full rounded-[16px]' mode='aspectFill' src={item.cropImageUrl} />
                  ) : null}

                  {item.summary ? <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>{item.summary}</Text> : null}
                  {item.buyPriority !== null ? (
                    <Text className='text-[14px] leading-[1.7] text-[#87674c]'>购买优先级：{item.buyPriority}</Text>
                  ) : null}

                  {item.reasons?.length ? (
                    <View className='flex flex-wrap gap-2'>
                      {item.reasons.map((reason) => (
                        <Text
                          className='rounded-full bg-[#f0e3ca] px-[10px] py-[6px] text-[12px] leading-[1.4] text-[#5a422f]'
                          key={`${item.label}-reason-${reason}`}
                        >
                          {reason}
                        </Text>
                      ))}
                    </View>
                  ) : null}

                  {item.risks?.length ? (
                    <View className='flex flex-wrap gap-2'>
                      {item.risks.map((risk) => (
                        <Text
                          className='rounded-full bg-[#f7d7cf] px-[10px] py-[6px] text-[12px] leading-[1.4] text-[#7f2d1f]'
                          key={`${item.label}-risk-${risk}`}
                        >
                          {risk}
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
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fff1ee] p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>任务异常</Text>
            <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>{errorMessage}</Text>
          </View>
        ) : null}

        <View className='flex flex-col gap-3'>
          {taskStatus === 'FAILED' ? (
            <Button
              className='w-full rounded-full bg-gradient-to-r from-[#8c5f32] to-[#b77a3d] text-[16px] font-semibold text-white'
              loading={isRetrying}
              onClick={handleRetry}
            >
              重试任务
            </Button>
          ) : null}

          <Button
            className='w-full rounded-full bg-[#f6ead7] text-[16px] font-semibold text-[#6d4c2f]'
            onClick={handleRestart}
          >
            重新选择图片
          </Button>
        </View>
      </View>
    </View>
  )
}
