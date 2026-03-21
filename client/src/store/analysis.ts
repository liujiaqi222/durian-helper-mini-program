import { create } from 'zustand'
import type { AnalysisResult, AnalysisTask, AnalysisTaskStatus } from '../types/analysis'

interface AnalysisState {
  localImagePath: string
  taskId: string
  taskStatus: AnalysisTaskStatus | null
  taskDetail: AnalysisTask | null
  result: AnalysisResult | null
  errorMessage: string
  setLocalImage: (path: string) => void
  setSubmissionContext: (payload: {
    taskId: string
    taskStatus: AnalysisTaskStatus
  }) => void
  setTaskDetail: (task: AnalysisTask) => void
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
  taskDetail: null,
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
      taskDetail: null,
      result: null,
      errorMessage: '',
    })),
  setTaskDetail: (task) =>
    set(() => ({
      taskDetail: task,
      taskStatus: task.status,
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
