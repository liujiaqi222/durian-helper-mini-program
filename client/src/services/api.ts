import Taro from '@tarojs/taro'
import type {
  AnalysisResult,
  AnalysisTask,
  CreateAnalysisTaskResponse,
} from '../types/analysis'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:3000/api/v1'
const API_BASE_URL = __API_BASE_URL__ || DEFAULT_API_BASE_URL

interface RequestOptions {
  method?: 'GET' | 'POST'
  data?: Record<string, unknown>
}

interface ApiEnvelope<T> {
  code: number
  data: T
  message: string
}

function joinUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await Taro.request<ApiEnvelope<T>>({
    url: joinUrl(path),
    method: options.method || 'GET',
    data: options.data,
    header: {
      'content-type': 'application/json',
    },
  })

  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw new Error(readErrorMessage(response.data) || '请求失败，请稍后重试')
  }

  return response.data.data
}

function readErrorMessage(data: unknown): string | null {
  if (!data || typeof data !== 'object') {
    return null
  }

  const candidate = data as Record<string, unknown>
  if (typeof candidate.message === 'string') {
    return candidate.message
  }

  if (Array.isArray(candidate.message)) {
    return candidate.message.join('，')
  }

  return null
}

export async function createAnalysisTask(filePath: string): Promise<CreateAnalysisTaskResponse> {
  const response = await Taro.uploadFile({
    url: joinUrl('/durians/analyze'),
    filePath,
    name: 'file',
  })

  if (response.statusCode < 200 || response.statusCode >= 300) {
    let message = '图片上传失败，请稍后重试'

    try {
      const parsedData = JSON.parse(response.data)
      message = readErrorMessage(parsedData) || message
    } catch {
      // Keep the fallback message when the response is not JSON.
    }

    throw new Error(message)
  }

  return (JSON.parse(response.data) as ApiEnvelope<CreateAnalysisTaskResponse>).data
}

export function getAnalysisTask(taskId: string): Promise<AnalysisTask> {
  return request<AnalysisTask>(`/durians/tasks/${taskId}`)
}

export function getAnalysisResult(taskId: string): Promise<AnalysisResult> {
  return request<AnalysisResult>(`/durians/tasks/${taskId}/result`)
}

export function retryAnalysisTask(taskId: string): Promise<AnalysisTask> {
  return request<AnalysisTask>(`/durians/tasks/${taskId}/retry`, {
    method: 'POST',
  })
}
