import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { LoginResponse, UserProfile } from '../types/user'

const taroMock = {
  getStorageSync: vi.fn(),
  setStorageSync: vi.fn(),
  removeStorageSync: vi.fn(),
  login: vi.fn(),
  request: vi.fn(),
  uploadFile: vi.fn(),
}

vi.mock('@tarojs/taro', () => ({
  __esModule: true,
  default: taroMock,
}))

function createUserProfile(overrides: Partial<UserProfile> = {}): UserProfile {
  return {
    publicId: 'user_123',
    name: null,
    phone: null,
    remainingCredits: 1,
    usedCredits: 0,
    inviteCode: 'INVITE01',
    adRewardCount: 0,
    inviteRewardCount: 0,
    createdAt: '2026-03-23T00:00:00.000Z',
    updatedAt: '2026-03-23T00:00:00.000Z',
    ...overrides,
  }
}

describe('bootstrapSession', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
    process.env.TARO_APP_API_BASE_URL = 'https://api.example.com'
    taroMock.getStorageSync.mockReturnValue(null)
  })

  it('passes inviterCode during first mini program login when opening from an invite link', async () => {
    const inviterCode = 'INVABC1'
    const loginUser = createUserProfile()
    const latestUser = createUserProfile({ remainingCredits: 2, inviteRewardCount: 1 })

    taroMock.login.mockResolvedValue({ code: 'wx-login-code' })
    taroMock.request.mockImplementation(async (options: { url: string; data?: Record<string, unknown>; header?: Record<string, string> }) => {
      if (options.url.endsWith('/auth/login')) {
        const response: LoginResponse = { token: 'token-1', user: loginUser }
        return {
          statusCode: 200,
          data: {
            code: 0,
            message: 'ok',
            data: response,
          },
        }
      }

      if (options.url.endsWith('/users/me')) {
        expect(options.header?.Authorization).toBe('Bearer token-1')
        return {
          statusCode: 200,
          data: {
            code: 0,
            message: 'ok',
            data: latestUser,
          },
        }
      }

      throw new Error(`Unexpected request URL: ${options.url}`)
    })

    const { bootstrapSession } = await import('./api')
    const profile = await bootstrapSession(inviterCode)

    expect(profile).toEqual(latestUser)

    const loginCall = taroMock.request.mock.calls.find(
      ([requestOptions]) =>
        typeof requestOptions?.url === 'string' && requestOptions.url.endsWith('/auth/login'),
    )

    expect(loginCall).toBeTruthy()
    expect(loginCall?.[0]?.data?.inviterCode).toBe(inviterCode)
  })
})
