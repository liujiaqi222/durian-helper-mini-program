export interface UserProfile {
  publicId: string
  name: string | null
  phone: string | null
  remainingCredits: number
  usedCredits: number
  inviteCode: string
  adRewardCount: number
  inviteRewardCount: number
  createdAt: string
  updatedAt: string
}

export interface LoginResponse {
  token: string
  user: UserProfile
}
