import { Button, Image, Text, View } from '@tarojs/components'
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
    <View className='min-h-screen bg-gradient-to-b from-[#f8f1e6] to-[#efe4d1] px-6 py-6 text-[#2e2017]'>
      <View className='flex flex-col gap-6'>
        <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
          <Text className='text-[12px] font-bold uppercase tracking-[2px] text-[#a67b4f]'>Durian Picker MVP</Text>
          <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>上传货架照片，找出更值得买的榴莲</Text>
          <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>
          小程序会把图片发送到后端，完成目标检测、编号和评分建议，再返回推荐结果。
          </Text>
        </View>

        <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
          <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>1. 选择图片</Text>
          <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>支持拍照或从相册中选择一张榴莲货架图。</Text>

          <Button
            className='w-full rounded-full bg-gradient-to-r from-[#8c5f32] to-[#b77a3d] text-[16px] font-semibold text-white'
            onClick={handleChooseImage}
          >
            {localImagePath ? '重新选择图片' : '选择图片'}
          </Button>

          {localImagePath ? (
            <View className='w-full overflow-hidden rounded-[18px] bg-[#f2e8d8]'>
              <Image className='h-[240px] w-full' mode='aspectFill' src={localImagePath} />
            </View>
          ) : (
            <View className='flex min-h-[240px] w-full items-center justify-center rounded-[18px] border-2 border-dashed border-[#d2b893] bg-[#f2e8d8]'>
              <Text className='text-[14px] text-[#8b6a4d]'>暂未选择图片</Text>
            </View>
          )}
        </View>

        <View className='flex flex-col gap-3 rounded-[22px] bg-[#fffbf4]/90 p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
          <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>2. 发起分析</Text>
          <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>
            点击后会直接上传图片并开始分析，结果页负责轮询任务状态并展示推荐结果。
          </Text>

          <Button
            className='w-full rounded-full bg-gradient-to-r from-[#8c5f32] to-[#b77a3d] text-[16px] font-semibold text-white disabled:opacity-50'
            disabled={!localImagePath || isSubmitting}
            loading={isSubmitting}
            onClick={handleStartAnalysis}
          >
            {isSubmitting ? '正在上传并开始分析' : '开始分析'}
          </Button>

          {localImagePath ? (
            <Button
              className='w-full rounded-full bg-[#f6ead7] text-[16px] font-semibold text-[#6d4c2f]'
              onClick={handleReset}
            >
              清空当前图片
            </Button>
          ) : null}
        </View>

        {errorMessage ? (
          <View className='flex flex-col gap-3 rounded-[22px] bg-[#fff1ee] p-5 shadow-[0_14px_35px_rgba(92,63,36,0.08)]'>
            <Text className='text-[22px] font-bold leading-[1.4] text-[#2e2017]'>操作失败</Text>
            <Text className='text-[14px] leading-[1.7] text-[#4d3d31]'>{errorMessage}</Text>
          </View>
        ) : null}
      </View>
    </View>
  )
}
