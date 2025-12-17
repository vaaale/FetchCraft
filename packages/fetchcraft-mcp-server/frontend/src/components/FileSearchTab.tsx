import { useState, useEffect } from 'react'
import { Search, File, ChevronLeft, ChevronRight, Loader, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { api, FileResult } from '../api'

// Check for embedded search results from MCP tool
declare global {
  interface Window {
    __SEARCH_RESULTS__?: {
      query: string
      files: FileResult[]
      total: number
      offset: number
      serverUrl?: string
    }
  }
}

export default function FileSearchTab() {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<FileResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [resultsPerPage, setResultsPerPage] = useState(10)
  const [hasSearched, setHasSearched] = useState(false)
  
  const performSearch = async (searchQuery: string, page: number, numResults: number) => {
    if (!searchQuery.trim()) {
      setError('Please enter a search query')
      return
    }

    setLoading(true)
    setError(null)
    setCurrentPage(page)
    setHasSearched(true)

    try {
      const offset = (page - 1) * numResults
      const response = await api.findFiles({
        query: searchQuery.trim(),
        num_results: numResults,
        offset,
      })

      setFiles(response.files)
      setTotal(response.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search files')
      setFiles([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }

  // Load embedded search results or URL query params on mount
  useEffect(() => {
    if (window.__SEARCH_RESULTS__) {
      // From MCP tool embedded data
      const embedded = window.__SEARCH_RESULTS__
      setQuery(embedded.query)
      setFiles(embedded.files)
      setTotal(embedded.total)
      setHasSearched(true)
      setCurrentPage(1)
    } else {
      // Check URL query parameters
      const urlParams = new URLSearchParams(window.location.search)
      const urlQuery = urlParams.get('query')
      const urlNumResults = urlParams.get('num_results')
      const urlOffset = urlParams.get('offset')
      
      if (urlQuery) {
        const numResults = urlNumResults ? parseInt(urlNumResults, 10) : 10
        const offset = urlOffset ? parseInt(urlOffset, 10) : 0
        const page = Math.floor(offset / numResults) + 1
        
        setQuery(urlQuery)
        setResultsPerPage(numResults)
        setCurrentPage(page)
        
        // Trigger search
        performSearch(urlQuery, page, numResults)
      }
    }
  }, [])

  const handleSearch = (page: number = 1) => {
    performSearch(query, page, resultsPerPage)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handleSearch(1)
  }

  const totalPages = Math.ceil(total / resultsPerPage)

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-blue-600 bg-blue-50'
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50'
    return 'text-gray-600 bg-gray-50'
  }

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Search Files</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex gap-3">
            <div className="flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query... (e.g., 'database configuration', 'API endpoints')"
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-sm"
                disabled={loading}
              />
            </div>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Search
            </button>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              Results per page:
              <select
                value={resultsPerPage}
                onChange={(e) => {
                  setResultsPerPage(Number(e.target.value))
                  if (hasSearched) handleSearch(1)
                }}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                disabled={loading}
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </label>
          </div>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center p-12">
          <Loader className="w-8 h-8 text-blue-600 animate-spin" />
        </div>
      )}

      {/* Results */}
      {!loading && hasSearched && (
        <>
          {files.length === 0 ? (
            <div className="text-center p-12 bg-gray-50 rounded-lg">
              <File className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">No files found matching your query</p>
              <p className="text-sm text-gray-500 mt-2">Try adjusting your search terms</p>
            </div>
          ) : (
            <>
              {/* Results Header */}
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-700">
                  Showing {(currentPage - 1) * resultsPerPage + 1} to{' '}
                  {Math.min(currentPage * resultsPerPage, total)} of {total} results
                </p>
              </div>

              {/* Files List */}
              <div className="space-y-3">
                {files.map((file, index) => (
                  <div
                    key={`${file.source}-${index}`}
                    className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <File className="w-5 h-5 text-blue-600 flex-shrink-0" />
                          <h3 className="text-sm font-semibold text-gray-900 truncate">
                            {file.filename}
                          </h3>
                        </div>
                        <p className="text-xs text-gray-500 mb-2 break-all">
                          {file.source}
                        </p>
                        <div className="text-sm text-gray-700 prose prose-sm max-w-none line-clamp-6">
                          <ReactMarkdown>{file.text_preview}</ReactMarkdown>
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        <span
                          className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${getScoreColor(
                            file.score
                          )}`}
                        >
                          {(file.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                  <div className="text-sm text-gray-700">
                    Page {currentPage} of {totalPages}
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleSearch(currentPage - 1)}
                      disabled={currentPage === 1 || loading}
                      className="p-2 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="px-4 py-2 text-sm text-gray-700">
                      {currentPage} / {totalPages}
                    </span>
                    <button
                      onClick={() => handleSearch(currentPage + 1)}
                      disabled={currentPage === totalPages || loading}
                      className="p-2 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}
