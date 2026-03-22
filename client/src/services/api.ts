import Taro from '@tarojs/taro'
import type {
  AnalysisResult,
  AnalysisTask,
  CreateAnalysisTaskResponse,
} from '../types/analysis'
import type { LoginResponse, UserProfile } from '../types/user'

const API_BASE_URL = process.env.TARO_APP_API_BASE_URL
const AUTH_TOKEN_STORAGE_KEY = 'durian_auth_token'
const AUTH_USER_STORAGE_KEY = 'durian_auth_user'

interface RequestOptions {
  method?: 'GET' | 'POST'
  data?: Record<string, unknown>
  requiresAuth?: boolean
}

interface ApiEnvelope<T> {
  code: number
  data: T
  message: string
}

interface AuthSession {
  token: string
  user: UserProfile
}

let authSession: AuthSession | null = null
let loginPromise: Promise<AuthSession> | null = null

function joinUrl(path: string): string {
  if (!API_BASE_URL) {
    throw new Error('缺少环境变量 TARO_APP_API_BASE_URL')
  }

  return `${API_BASE_URL}${path}`
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const session = options.requiresAuth === false ? null : await ensureAuthSession()
  const response = await Taro.request<ApiEnvelope<T>>({
    url: joinUrl(path),
    method: options.method || 'GET',
    data: options.data,
    header: {
      ...(session ? { Authorization: `Bearer ${session.token}` } : {}),
      'content-type': 'application/json',
    },
  })

  if (response.statusCode === 401 && options.requiresAuth !== false) {
    clearAuthSession()
    const refreshedSession = await ensureAuthSession(true)
    authSession = refreshedSession

    return request<T>(path, {
      ...options,
      requiresAuth: true,
    })
  }

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

function readStoredSession(): AuthSession | null {
  try {
    const token = Taro.getStorageSync<string>(AUTH_TOKEN_STORAGE_KEY)
    const user = Taro.getStorageSync<UserProfile>(AUTH_USER_STORAGE_KEY)

    if (!token || !user) {
      return null
    }

    return { token, user }
  } catch {
    return null
  }
}

function persistAuthSession(session: AuthSession): void {
  authSession = session
  Taro.setStorageSync(AUTH_TOKEN_STORAGE_KEY, session.token)
  Taro.setStorageSync(AUTH_USER_STORAGE_KEY, session.user)
}

function clearAuthSession(): void {
  authSession = null
  Taro.removeStorageSync(AUTH_TOKEN_STORAGE_KEY)
  Taro.removeStorageSync(AUTH_USER_STORAGE_KEY)
}

async function loginWithMiniProgram(inviterCode?: string): Promise<AuthSession> {
  const loginResult = await Taro.login()
  if (!loginResult.code) {
    throw new Error('微信登录失败，请稍后重试')
  }

  const response = await Taro.request<ApiEnvelope<LoginResponse>>({
    url: joinUrl('/auth/login'),
    method: 'POST',
    data: {
      code: loginResult.code,
      ...(inviterCode ? { inviterCode } : {}),
    },
    header: {
      'content-type': 'application/json',
    },
  })

  if (response.statusCode < 200 || response.statusCode >= 300) {
    throw new Error(readErrorMessage(response.data) || '登录失败，请稍后重试')
  }

  const session = response.data.data
  persistAuthSession(session)
  return session
}

async function ensureAuthSession(forceRefresh = false, inviterCode?: string): Promise<AuthSession> {
  if (!forceRefresh) {
    if (!authSession) {
      authSession = readStoredSession()
    }

    if (authSession) {
      return authSession
    }

    if (loginPromise) {
      return loginPromise
    }
  }

  loginPromise = loginWithMiniProgram(inviterCode).finally(() => {
    loginPromise = null
  })

  return loginPromise
}

export async function bootstrapSession(inviterCode?: string): Promise<UserProfile> {

  try {
    const profile = await request<UserProfile>('/users/me')
    updateCachedUser(profile)
    return profile
  } catch {
    clearAuthSession()
    const refreshed = await ensureAuthSession(true, inviterCode)
    return refreshed.user
  }
}

export function getCachedUserProfile(): UserProfile | null {
  if (!authSession) {
    authSession = readStoredSession()
  }

  return authSession?.user || null
}

export function updateCachedUser(profile: UserProfile): void {
  const session = authSession || readStoredSession()
  if (!session) {
    return
  }

  persistAuthSession({
    token: session.token,
    user: profile,
  })
}

export async function createAnalysisTask(filePath: string): Promise<CreateAnalysisTaskResponse> {
  const session = await ensureAuthSession()
  const response = await Taro.uploadFile({
    url: joinUrl('/durians/analyze'),
    filePath,
    name: 'file',
    header: {
      Authorization: `Bearer ${session.token}`,
    },
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

  const data = (JSON.parse(response.data) as ApiEnvelope<CreateAnalysisTaskResponse>).data
  if (authSession) {
    updateCachedUser({
      ...authSession.user,
      remainingCredits: data.remainingCredits,
      usedCredits: authSession.user.usedCredits + 1,
    })
  }

  return data
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

export function grantAdReward(): Promise<UserProfile> {
  return request<UserProfile>('/users/me/rewards/ad', {
    method: 'POST',
    data: {},
  }).then((profile) => {
    updateCachedUser(profile)
    return profile
  })
}
