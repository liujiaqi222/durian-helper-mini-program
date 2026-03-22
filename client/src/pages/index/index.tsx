import { Button, Image, Text, View } from '@tarojs/components'
import Taro, { useShareAppMessage } from '@tarojs/taro'
import { useState } from 'react'
import { createAnalysisTask } from '../../services/api'
import { useAnalysisStore } from '../../store/analysis'
import { useUserStore } from '../../store/user'
import { buildInviteSharePath } from '../../utils/user'

export default function Index() {
  const localImagePath = useAnalysisStore((state) => state.localImagePath)
  const setLocalImage = useAnalysisStore((state) => state.setLocalImage)
  const setSubmissionContext = useAnalysisStore((state) => state.setSubmissionContext)
  const errorMessage = useAnalysisStore((state) => state.errorMessage)
  const setErrorMessage = useAnalysisStore((state) => state.setErrorMessage)
  const clearErrorMessage = useAnalysisStore((state) => state.clearErrorMessage)
  const resetAnalysis = useAnalysisStore((state) => state.resetAnalysis)
  const profile = useUserStore((state) => state.profile)
  const isBootstrapping = useUserStore((state) => state.isBootstrapping)
  const authError = useUserStore((state) => state.authError)
  const setProfile = useUserStore((state) => state.setProfile)
  const [isSubmitting, setIsSubmitting] = useState(false)

  useShareAppMessage(() => ({
    title: '我在用榴莲挑选助手，分享给你一起领额外次数',
    path: buildInviteSharePath(profile?.inviteCode || ''),
  }))

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
    if (!localImagePath || isSubmitting || isBootstrapping || !profile?.remainingCredits) {
      return
    }

    setIsSubmitting(true)
    clearErrorMessage()

    try {
      const task = await createAnalysisTask(localImagePath)

      setProfile({
        ...profile,
        remainingCredits: task.remainingCredits,
        usedCredits: profile.usedCredits + 1,
      })

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

  const remainingCreditsText = isBootstrapping ? '登录中...' : `剩余 ${profile?.remainingCredits ?? 0} 次`
  const primaryButtonText = isSubmitting
    ? '分析中...'
    : !localImagePath
      ? '先上传图片'
      : !profile?.remainingCredits && !isBootstrapping
        ? '邀请好友获取次数'
        : '开始挑选'
  const primaryButtonDisabled =
    isSubmitting || isBootstrapping || (!!localImagePath && !profile?.remainingCredits)

  return (
    <View className='min-h-screen bg-gradient-to-br from-yellow-50 via-white to-amber-50 px-4 py-6'>
      <View className='flex flex-col gap-5'>
        <View className='flex flex-col gap-3 rounded-3xl border border-yellow-100 bg-white px-6 py-5 shadow-sm'>
          <Text className='text-[34px] font-extrabold leading-[1.15] text-gray-900'>智能挑选最棒的榴莲</Text>
          <Text className='text-sm leading-relaxed text-gray-500'>
            拍一张货架图，快速挑出更值得买的那颗。
          </Text>
        </View>

        <View className='flex flex-col gap-4 rounded-3xl border border-yellow-100 bg-white p-6 shadow-sm'>
          <Text className='text-lg font-bold text-gray-900'>开始挑选</Text>

          {localImagePath ? (
            <View
              className='relative w-full overflow-hidden rounded-2xl bg-gray-50 ring-1 ring-gray-100 active:opacity-95'
              onClick={isSubmitting ? undefined : handleChooseImage}
            >
              <Image className='h-64 w-full' mode='aspectFill' src={localImagePath} />
              <View
                className={`absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full ${isSubmitting ? 'bg-white/50' : 'bg-white/90'
                  }`}
                onClick={(event) => {
                  event.stopPropagation?.()
                  if (!isSubmitting) {
                    handleReset()
                  }
                }}
              >
                <Text className={`text-lg font-bold ${isSubmitting ? 'text-gray-400' : 'text-gray-600'}`}>×</Text>
              </View>
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

          <View className='mt-2 flex flex-col gap-3'>

            <View
              className={`flex w-full items-center justify-center rounded-2xl py-3 text-base font-bold transition-all ${primaryButtonDisabled
                  ? 'bg-amber-300 text-white opacity-70 cursor-not-allowed'
                  : 'bg-amber-500 text-white shadow-sm active:scale-95 active:bg-amber-600'
                }`}
              onClick={
                primaryButtonDisabled
                  ? undefined
                  : localImagePath
                    ? handleStartAnalysis
                    : handleChooseImage
              }
            >
              <Text>{primaryButtonText}</Text>
            </View>
            <View className='flex items-center justify-between rounded-2xl border border-amber-100 bg-white px-1 py-1'>
              <View className='flex-1 px-3 py-2'>
                <View className='text-sm font-semibold text-gray-900'>{remainingCreditsText}</View>
                <Text className='mt-1 text-xs leading-relaxed text-gray-500'>邀请好友可增加次数，新用户登录后自动到账。</Text>
              </View>
              <Button
                className='m-0 mr-1 flex h-9 items-center justify-center rounded-full bg-gray-900 px-4 text-xs font-bold text-white'
                openType='share'
              >
                去邀请
              </Button>
            </View>

            {authError ? (
              <Text className='text-sm leading-relaxed text-red-500'>{authError}</Text>
            ) : null}
          </View>
        </View>

        {errorMessage ? (
          <View className='flex flex-col gap-2 rounded-2xl border border-red-100 bg-red-50 p-5'>
            <Text className='text-base font-bold text-red-600'>哎呀，出错了</Text>
            <Text className='text-sm leading-relaxed text-red-500'>{errorMessage}</Text>
          </View>
        ) : null}
      </View>
    </View>
  )
}
