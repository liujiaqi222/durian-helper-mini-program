import { Text, View } from '@tarojs/components'
import Taro, { useLoad } from '@tarojs/taro'
import { useState } from 'react'
import { getAnalysisHistory } from '../../services/api'
import type { AnalysisHistoryItem } from '../../types/analysis'
import { getStatusDescription } from '../../utils/analysis'

function formatHistoryTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return '时间未知'
  }

  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  const hours = `${date.getHours()}`.padStart(2, '0')
  const minutes = `${date.getMinutes()}`.padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day} ${hours}:${minutes}`
}

function getStatusText(item: AnalysisHistoryItem): string {
  if (item.status === 'DONE') {
    return item.recommendedLabel ? `推荐 ${item.recommendedLabel}` : '分析完成'
  }

  if (item.status === 'FAILED') {
    return '分析失败'
  }

  return '分析中'
}

export default function HistoryPage() {
  const [items, setItems] = useState<AnalysisHistoryItem[]>([])
  const [errorMessage, setErrorMessage] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useLoad(() => {
    void loadHistory()
  })

  async function loadHistory() {
    setIsLoading(true)
    setErrorMessage('')

    try {
      const nextItems = await getAnalysisHistory()
      setItems(nextItems)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : '获取历史记录失败')
    } finally {
      setIsLoading(false)
    }
  }

  async function handleOpenTask(taskId: string) {
    await Taro.navigateTo({
      url: `/pages/result/index?taskId=${taskId}`,
    })
  }

  return (
    <View className='min-h-screen bg-linear-to-br from-yellow-50 via-white to-amber-50 px-4 py-4 pb-8'>
      <View className='flex flex-col gap-4'>
        <View className='rounded-3xl border border-yellow-100 bg-white px-5 py-4 shadow-sm'>
          <View className='text-lg font-bold text-gray-900'>最近 20 条分析榴莲记录</View>
        
        </View>

        {isLoading ? (
          <View className='rounded-3xl border border-yellow-100 bg-white px-5 py-8 text-center shadow-sm'>
            <Text className='text-sm text-gray-500'>正在加载历史记录...</Text>
          </View>
        ) : null}

        {!isLoading && errorMessage ? (
          <View className='rounded-3xl border border-red-100 bg-red-50 px-5 py-4 shadow-sm'>
            <Text className='text-sm leading-relaxed text-red-500'>{errorMessage}</Text>
          </View>
        ) : null}

        {!isLoading && !errorMessage && items.length === 0 ? (
          <View className='rounded-3xl border border-yellow-100 bg-white px-5 py-8 text-center shadow-sm'>
            <Text className='text-sm text-gray-500'>还没有历史记录，先去分析一张图片。</Text>
          </View>
        ) : null}

        {!isLoading && !errorMessage
          ? items.map((item) => (
              <View
                key={item.id}
                className='flex flex-col gap-3 rounded-3xl border border-yellow-100 bg-white px-5 py-4 shadow-sm active:scale-[0.99] active:bg-amber-50'
                onClick={() => {
                  void handleOpenTask(item.id)
                }}
              >
                <View className='flex items-center justify-between gap-3'>
                  <Text className='text-base font-bold text-gray-900'>{getStatusText(item)}</Text>
                  <Text className='text-xs text-gray-400'>{formatHistoryTime(item.createdAt)}</Text>
                </View>
                <View className='flex items-center justify-between gap-3'>
                  <Text className='text-sm text-gray-500'>
                    识别到 {item.detectedCount} 个榴莲
                  </Text>
                  <Text className='text-xs font-medium text-amber-700'>
                    {getStatusDescription(item.status)}
                  </Text>
                </View>
              </View>
            ))
          : null}
      </View>
    </View>
  )
}
