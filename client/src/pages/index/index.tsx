import { Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { useState } from 'react'
import { createAnalysisTask } from '../../services/api'
import { useAnalysisStore } from '../../store/analysis'

export default function Index() {
  const localImagePath = useAnalysisStore((state) => state.localImagePath)
  const setLocalImage = useAnalysisStore((state) => state.setLocalImage)
  const setSubmissionContext = useAnalysisStore((state) => state.setSubmissionContext)
  const errorMessage = useAnalysisStore((state) => state.errorMessage)
  const setErrorMessage = useAnalysisStore((state) => state.setErrorMessage)
  const clearErrorMessage = useAnalysisStore((state) => state.clearErrorMessage)
  const resetAnalysis = useAnalysisStore((state) => state.resetAnalysis)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleChooseImage() {
    clearErrorMessage()

    try {
      const result = await Taro.chooseImage({
        count: 1,
        sizeType: ['compressed'],
        sourceType: ['album', 'camera'],
      })

      const nextPath = result.tempFilePaths[0]
      if (!nextPath) {
        return
      }

      setLocalImage(nextPath)
    } catch (error) {
      if (error && typeof error === 'object' && 'errMsg' in error) {
        const errMsg = String((error as { errMsg: unknown }).errMsg)
        if (errMsg.includes('cancel')) {
          return
        }
      }

      setErrorMessage('选择图片失败，请重试')
    }
  }

  async function handleStartAnalysis() {
    if (!localImagePath || isSubmitting) {
      return
    }

    setIsSubmitting(true)
    clearErrorMessage()

    try {
      const task = await createAnalysisTask(localImagePath)

      setSubmissionContext({
        taskId: task.taskId,
        taskStatus: task.status,
      })

      await Taro.navigateTo({
        url: '/pages/result/index',
      })
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '创建分析任务失败，请稍后重试')
    } finally {
      setIsSubmitting(false)
    }
  }

  function handleReset() {
    resetAnalysis()
  }

  return (
    <View className='min-h-screen bg-gradient-to-br from-yellow-50 via-white to-amber-50 px-4 py-6'>
      <View className='flex flex-col gap-5'>
        {/* Welcome Section */}
        <View className='flex flex-col gap-2 rounded-3xl bg-white p-6 shadow-sm border border-yellow-100'>
          <Text className='text-xs font-bold uppercase tracking-widest text-amber-500'>Durian Picker</Text>
          <Text className='text-2xl font-extrabold leading-tight text-gray-900'>智能挑选最棒的榴莲</Text>
          <Text className='mb-1 text-sm leading-relaxed text-gray-500'>
            上传货架照片，不仅能帮你进行目标检测和编号，还能为您提供详细的评分建议，买榴莲不再踩坑。
          </Text>
        </View>

        {/* Action Section */}
        <View className='flex flex-col gap-4 rounded-3xl bg-white p-6 shadow-sm border border-yellow-100'>
          <Text className='text-lg font-bold text-gray-900'>开始挑选</Text>

          {localImagePath ? (
            <View className='w-full overflow-hidden rounded-2xl bg-gray-50 ring-1 ring-gray-100'>
              <Image className='h-64 w-full' mode='aspectFill' src={localImagePath} />
            </View>
          ) : (
            <View 
              className='flex h-48 w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-amber-300 bg-amber-50/50 active:bg-amber-100'
              onClick={handleChooseImage}
            >
              <Text className='text-4xl'>📸</Text>
              <Text className='text-sm font-medium text-amber-700'>点击选择或拍摄榴莲图片</Text>
            </View>
          )}

          <View className='flex flex-col gap-3 mt-2'>
            <View
              className={`flex w-full items-center justify-center rounded-2xl py-3 text-base font-bold transition-all ${
                !localImagePath || isSubmitting
                  ? 'bg-amber-300 text-white opacity-70 cursor-not-allowed'
                  : 'bg-amber-500 text-white shadow-sm active:scale-95 active:bg-amber-600'
              }`}
              onClick={(!localImagePath || isSubmitting) ? undefined : handleStartAnalysis}
            >
              <Text>{isSubmitting ? '分析中...' : '立即开始挑选'}</Text>
            </View>

            {localImagePath ? (
              <View className='flex gap-3'>
                <View
                  className={`flex flex-1 items-center justify-center rounded-2xl py-3 text-sm font-medium transition-all ${
                    isSubmitting
                      ? 'bg-amber-50/50 text-amber-700/50 cursor-not-allowed'
                      : 'bg-amber-50 text-amber-700 active:scale-95 active:bg-amber-100'
                  }`}
                  onClick={isSubmitting ? undefined : handleChooseImage}
                >
                  <Text>重新选择图片</Text>
                </View>
                <View
                  className={`flex flex-1 items-center justify-center rounded-2xl py-3 text-sm font-medium transition-all ${
                    isSubmitting
                      ? 'bg-gray-50/50 text-gray-600/50 cursor-not-allowed'
                      : 'bg-gray-50 text-gray-600 active:scale-95 active:bg-gray-100'
                  }`}
                  onClick={isSubmitting ? undefined : handleReset}
                >
                  <Text>清空当前进度</Text>
                </View>
              </View>
            ) : null}
          </View>
        </View>

        {/* Error Message */}
        {errorMessage ? (
          <View className='flex flex-col gap-2 rounded-2xl bg-red-50 p-5 border border-red-100'>
            <Text className='text-base font-bold text-red-600'>哎呀，出错了</Text>
            <Text className='text-sm leading-relaxed text-red-500'>{errorMessage}</Text>
          </View>
        ) : null}
      </View>
    </View>
  )
}
