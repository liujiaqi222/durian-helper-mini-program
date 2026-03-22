import { PropsWithChildren } from 'react'
import { useLaunch } from '@tarojs/taro'
import { bootstrapSession } from './services/api'
import { useUserStore } from './store/user'
import { readInviterCodeFromQuery } from './utils/user'

import './app.css'

function App({ children }: PropsWithChildren<any>) {
  useLaunch((options) => {
    const store = useUserStore.getState()
    const inviterCode = readInviterCodeFromQuery(options?.query)
    store.setBootstrapping(true)
    store.clearAuthError()

    void bootstrapSession(inviterCode || undefined)
      .then((profile) => {
        useUserStore.getState().setProfile(profile)
      })
      .catch((error) => {
        useUserStore.getState().setAuthError(
          error instanceof Error ? error.message : '登录失败，请稍后重试',
        )
      })
      .finally(() => {
        useUserStore.getState().setBootstrapping(false)
      })
  })

  return children
}

export default App
