import { create } from 'zustand'
import type { AnalysisResult, AnalysisTaskStatus } from '../types/analysis'

interface AnalysisState {
  localImagePath: string
  taskId: string
  taskStatus: AnalysisTaskStatus | null
  result: AnalysisResult | null
  errorMessage: string
  setLocalImage: (path: string) => void
  setSubmissionContext: (payload: {
    taskId: string
    taskStatus: AnalysisTaskStatus
  }) => void
  setTaskStatus: (status: AnalysisTaskStatus) => void
  setResult: (result: AnalysisResult) => void
  setErrorMessage: (message: string) => void
  clearErrorMessage: () => void
  resetAnalysis: () => void
}

const initialState = {
  localImagePath: '',
  taskId: '',
  taskStatus: null,
  result: null,
  errorMessage: '',
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  ...initialState,
  setLocalImage: (path) =>
    set(() => ({
      ...initialState,
      localImagePath: path,
    })),
  setSubmissionContext: (payload) =>
    set(() => ({
      taskId: payload.taskId,
      taskStatus: payload.taskStatus,
      result: null,
      errorMessage: '',
    })),
  setTaskStatus: (status) =>
    set(() => ({
      taskStatus: status,
    })),
  setResult: (result) =>
    set(() => ({
      result,
      taskStatus: 'DONE',
      errorMessage: '',
    })),
  setErrorMessage: (message) =>
    set(() => ({
      errorMessage: message,
    })),
  clearErrorMessage: () =>
    set(() => ({
      errorMessage: '',
    })),
  resetAnalysis: () =>
    set(() => ({
      ...initialState,
    })),
}))
