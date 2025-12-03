// API Client for Fetchcraft File Finder

export interface FileResult {
  filename: string
  source: string
  score: number
  text_preview: string
}

export interface FindFilesResponse {
  files: FileResult[]
  total: number
  offset: number
}

// Get API base URL from embedded config or use relative path
function getApiBase(): string {
  const serverUrl = (window as any).__SEARCH_RESULTS__?.serverUrl
  return serverUrl ? `${serverUrl}/api` : '/api'
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${endpoint}`, {
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
  // Find Files
  findFiles: (params: {
    query: string
    num_results?: number
    offset?: number
  }): Promise<FindFilesResponse> => {
    const searchParams = new URLSearchParams()
    searchParams.append('query', params.query)
    if (params.num_results) searchParams.append('num_results', params.num_results.toString())
    if (params.offset) searchParams.append('offset', params.offset.toString())

    return fetchAPI(`/find-files?${searchParams}`)
  },
}
