export function normalizeInviteCode(value: string): string {
  return value.trim().toUpperCase()
}

export function readInviterCodeFromQuery(query?: Record<string, unknown> | null): string {
  const inviterCode = query?.inviterCode
  if (typeof inviterCode !== 'string') {
    return ''
  }

  return normalizeInviteCode(inviterCode)
}

export function buildInviteSharePath(inviteCode: string): string {
  const normalizedInviteCode = normalizeInviteCode(inviteCode)
  if (!normalizedInviteCode) {
    return '/pages/index/index'
  }

  return `/pages/index/index?inviterCode=${encodeURIComponent(normalizedInviteCode)}`
}

export function buildResultSharePath(inviteCode: string, taskId: string): string {
  const normalizedTaskId = taskId.trim()
  if (!normalizedTaskId) {
    return buildInviteSharePath(inviteCode)
  }

  const normalizedInviteCode = normalizeInviteCode(inviteCode)
  if (!normalizedInviteCode) {
    return `/pages/result/index?taskId=${encodeURIComponent(normalizedTaskId)}`
  }

  return `/pages/result/index?taskId=${encodeURIComponent(normalizedTaskId)}&inviterCode=${encodeURIComponent(normalizedInviteCode)}`
}
