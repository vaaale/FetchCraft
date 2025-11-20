// API Client for Fetchcraft Admin

export interface Message {
  id: string
  job_id: string
  job_name: string
  source: string
  status: string
  current_step: string | null
  step_statuses: Record<string, string>
  pipeline_steps: string[]
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  error_step: string | null
  retry_count: number
}

export interface MessagesResponse {
  messages: Message[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface QueueStats {
  total_messages: number
  by_state: Record<string, number>
  by_queue: Record<string, number>
  failed_messages: number
  oldest_pending: string | null
}

export interface IngestionStatus {
  status: 'running' | 'stopped' | 'error'
  pid: number | null
}

const API_BASE = '/api'

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

export const api = {
  // Queue Messages
  getMessages: (params: {
    job_id: string
    state?: string
    limit?: number
    offset?: number
  }): Promise<MessagesResponse> => {
    const searchParams = new URLSearchParams()
    searchParams.append('job_id', params.job_id)
    if (params.state && params.state !== 'all') searchParams.append('state', params.state)
    if (params.limit) searchParams.append('limit', params.limit.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())

    return fetchAPI(`/messages?${searchParams}`)
  },

  // Queue Statistics
  getStats: (): Promise<QueueStats> => {
    return fetchAPI('/stats')
  },

  // Ingestion Control
  startIngestion: (): Promise<IngestionStatus> => {
    return fetchAPI('/ingestion/start', { method: 'POST' })
  },

  stopIngestion: (): Promise<IngestionStatus> => {
    return fetchAPI('/ingestion/stop', { method: 'POST' })
  },

  getIngestionStatus: (): Promise<IngestionStatus> => {
    return fetchAPI('/ingestion/status')
  },
}
