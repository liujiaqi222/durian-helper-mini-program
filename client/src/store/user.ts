import { create } from 'zustand'
import type { UserProfile } from '../types/user'

interface UserState {
  profile: UserProfile | null
  isBootstrapping: boolean
  authError: string
  setProfile: (profile: UserProfile | null) => void
  setBootstrapping: (value: boolean) => void
  setAuthError: (message: string) => void
  clearAuthError: () => void
}

export const useUserStore = create<UserState>((set) => ({
  profile: null,
  isBootstrapping: false,
  authError: '',
  setProfile: (profile) => set(() => ({ profile })),
  setBootstrapping: (value) => set(() => ({ isBootstrapping: value })),
  setAuthError: (message) => set(() => ({ authError: message })),
  clearAuthError: () => set(() => ({ authError: '' })),
}))
