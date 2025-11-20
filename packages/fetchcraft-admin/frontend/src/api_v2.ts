// API Client for Fetchcraft Admin V2 - Enhanced Job and Document Tracking

export interface Job {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  source_path: string
  document_root: string
  pipeline_steps: string[]
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
}

export interface JobListResponse {
  jobs: Job[]
  total: number
}

export interface Document {
  id: string
  job_id: string
  source: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  current_step: string | null
  step_statuses: Record<string, string>
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  error_step: string | null
  retry_count: number
}

export interface DocumentListResponse {
  documents: Document[]
  total: number
}

export interface DirectoryItem {
  name: string
  path: string
  is_directory: boolean
}

export interface DirectoryListResponse {
  items: DirectoryItem[]
  current_path: string
}

export interface CreateJobRequest {
  name: string
  source_path: string
}

export interface RetryResponse {
  retried_count: number
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

export const apiV2 = {
  // Health Check
  healthCheck: (): Promise<{ status: string; database_connected: boolean }> => {
    return fetchAPI('/health')
  },

  // Directory Browsing
  listDirectories: (path?: string): Promise<DirectoryListResponse> => {
    const params = path ? `?path=${encodeURIComponent(path)}` : ''
    return fetchAPI(`/directories${params}`)
  },

  // Jobs
  createJob: (request: CreateJobRequest): Promise<Job> => {
    return fetchAPI('/jobs', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  },

  listJobs: (params?: {
    status?: string
    limit?: number
    offset?: number
  }): Promise<JobListResponse> => {
    const searchParams = new URLSearchParams()
    if (params?.status) searchParams.append('status', params.status)
    if (params?.limit) searchParams.append('limit', params.limit.toString())
    if (params?.offset) searchParams.append('offset', params.offset.toString())

    const query = searchParams.toString()
    return fetchAPI(`/jobs${query ? `?${query}` : ''}`)
  },

  getJob: (jobId: string): Promise<Job> => {
    return fetchAPI(`/jobs/${jobId}`)
  },

  deleteJob: (jobId: string, deleteDocuments: boolean = true): Promise<{ message: string; job_id: string }> => {
    return fetchAPI(`/jobs/${jobId}?delete_documents=${deleteDocuments}`, {
      method: 'DELETE',
    })
  },

  restartJob: (jobId: string): Promise<Job> => {
    return fetchAPI(`/jobs/${jobId}/restart`, {
      method: 'POST',
    })
  },

  // Documents
  listJobDocuments: (
    jobId: string,
    params?: {
      status?: string
      limit?: number
      offset?: number
    }
  ): Promise<DocumentListResponse> => {
    const searchParams = new URLSearchParams()
    if (params?.status) searchParams.append('status', params.status)
    if (params?.limit) searchParams.append('limit', params.limit.toString())
    if (params?.offset) searchParams.append('offset', params.offset.toString())

    const query = searchParams.toString()
    return fetchAPI(`/jobs/${jobId}/documents${query ? `?${query}` : ''}`)
  },

  retryFailedDocuments: (jobId: string): Promise<RetryResponse> => {
    return fetchAPI(`/jobs/${jobId}/retry`, { method: 'POST' })
  },
}
