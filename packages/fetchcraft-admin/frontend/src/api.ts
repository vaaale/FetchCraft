// API Client for Fetchcraft Admin

export interface Message {
  id: string
  queue: string
  state: string
  attempts: number
  available_at: string | null
  lease_until: string | null
  body_preview: string
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
    state?: string
    queue?: string
    limit?: number
    offset?: number
  }): Promise<MessagesResponse> => {
    const searchParams = new URLSearchParams()
    if (params.state && params.state !== 'all') searchParams.append('state', params.state)
    if (params.queue && params.queue !== 'all') searchParams.append('queue', params.queue)
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
