import React, { useEffect, useState } from 'react'
import { RefreshCw, ChevronLeft, ChevronRight, AlertCircle, CheckCircle, Loader, Clock } from 'lucide-react'
import { api, Message, QueueStats } from '../api'
import { apiV2, Job } from '../api_v2'

const QueueTab = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [stats, setStats] = useState<QueueStats | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [state, setState] = useState<string>('all')
  const [rowsPerPage, setRowsPerPage] = useState<number>(50)
  const [currentPage, setCurrentPage] = useState<number>(1)

  // Pagination
  const [total, setTotal] = useState<number>(0)

  const fetchJobs = async () => {
    try {
      const jobsData = await apiV2.listJobs({ limit: 1000 })
      setJobs(jobsData.jobs)
      
      // Auto-select the first processing/running job if none selected
      if (!selectedJobId && jobsData.jobs.length > 0) {
        const processingJob = jobsData.jobs.find(
          job => job.status === 'running' || job.status === 'pending'
        )
        if (processingJob) {
          setSelectedJobId(processingJob.id)
        } else {
          // If no processing job, select the first one
          setSelectedJobId(jobsData.jobs[0].id)
        }
      }
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
    }
  }

  const fetchData = async () => {
    // Don't fetch if no job is selected
    if (!selectedJobId) {
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const offset = (currentPage - 1) * rowsPerPage
      const limit = rowsPerPage === -1 ? 10000 : rowsPerPage // "All" option

      const [messagesData, statsData] = await Promise.all([
        api.getMessages({ job_id: selectedJobId, state, limit, offset }),
        api.getStats(),
      ])

      setMessages(messagesData.messages)
      setTotal(messagesData.total)
      setStats(statsData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    // Fetch jobs less frequently
    const jobsInterval = setInterval(fetchJobs, 30000)
    return () => clearInterval(jobsInterval)
  }, [])

  useEffect(() => {
    fetchData()
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [state, selectedJobId, rowsPerPage, currentPage])

  const totalPages = rowsPerPage === -1 ? 1 : Math.ceil(total / rowsPerPage)

  // Get all unique pipeline steps across all messages, maintaining order
  const allPipelineSteps = React.useMemo(() => {
    const stepsSet = new Set<string>()
    const stepsOrder: string[] = []
    
    messages.forEach((message) => {
      message.pipeline_steps.forEach((step) => {
        if (!stepsSet.has(step)) {
          stepsSet.add(step)
          stepsOrder.push(step)
        }
      })
    })
    
    return stepsOrder
  }, [messages])

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-600" />
      case 'processing':
        return <Loader className="w-4 h-4 text-blue-600 animate-spin" />
      case 'pending':
        return <Clock className="w-4 h-4 text-gray-400" />
      default:
        return <div className="w-4 h-4 rounded-full bg-gray-200" />
    }
  }

  const getStepStatusColor = (status: string, isCurrent: boolean) => {
    if (isCurrent) {
      return 'bg-blue-100 border-blue-300'
    }
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200'
      case 'failed':
        return 'bg-red-50 border-red-200'
      case 'processing':
        return 'bg-blue-50 border-blue-200'
      case 'pending':
        return 'bg-gray-50 border-gray-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const getStepDisplayName = (stepName: string) => {
    // Convert technical step names to user-friendly display names
    const displayNames: Record<string, string> = {
      'ParsingTransformation': 'Parsing',
      'ExtractKeywords': 'Keywords',
      'ChunkingTransformation': 'Chunking',
      'sink:document_store': 'Doc Store',
      'sink:vector_index': 'Vector Index',
    }
    
    return displayNames[stepName] || stepName.replace('sink:', '')
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="text-sm font-medium text-blue-600">Total Messages</div>
            <div className="text-2xl font-bold text-blue-900">{stats.total_messages}</div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="text-sm font-medium text-green-600">Done</div>
            <div className="text-2xl font-bold text-green-900">
              {stats.by_state.done || 0}
            </div>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="text-sm font-medium text-yellow-600">Pending</div>
            <div className="text-2xl font-bold text-yellow-900">
              {stats.by_state.ready || 0}
            </div>
          </div>
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="text-sm font-medium text-red-600">Failed</div>
            <div className="text-2xl font-bold text-red-900">{stats.failed_messages}</div>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="text-sm font-medium text-purple-600">Errors</div>
            <div className="text-2xl font-bold text-purple-900">
              {stats.by_queue['ingest.error'] || 0}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {/* Job Filter */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Job:</label>
            <select
              value={selectedJobId}
              onChange={(e) => {
                setSelectedJobId(e.target.value)
                setCurrentPage(1)
              }}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 max-w-xs"
            >
              {!selectedJobId && <option value="">&lt;Select Job&gt;</option>}
              {jobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.name} ({job.status})
                </option>
              ))}
            </select>
          </div>

          {/* State Filter */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Status:</label>
            <select
              value={state}
              onChange={(e) => {
                setState(e.target.value)
                setCurrentPage(1)
              }}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* Rows Per Page */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Rows:</label>
            <select
              value={rowsPerPage}
              onChange={(e) => {
                setRowsPerPage(Number(e.target.value))
                setCurrentPage(1)
              }}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={-1}>All</option>
            </select>
          </div>
        </div>

        {/* Refresh Button */}
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Messages Table */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Job
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Document
                </th>
                {allPipelineSteps.map((step) => (
                  <th
                    key={step}
                    className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    {getStepDisplayName(step)}
                  </th>
                ))}
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {!selectedJobId ? (
                <tr>
                  <td colSpan={allPipelineSteps.length + 3} className="px-6 py-12 text-center text-gray-500">
                    <div className="text-lg font-medium mb-2">Please select a job</div>
                    <div className="text-sm">Choose a job from the dropdown above to view its documents</div>
                  </td>
                </tr>
              ) : loading && messages.length === 0 ? (
                <tr>
                  <td colSpan={allPipelineSteps.length + 3} className="px-6 py-12 text-center text-gray-500">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    Loading messages...
                  </td>
                </tr>
              ) : messages.length === 0 ? (
                <tr>
                  <td colSpan={allPipelineSteps.length + 3} className="px-6 py-12 text-center text-gray-500">
                    No messages found
                  </td>
                </tr>
              ) : (
                messages.map((message) => (
                  <tr key={message.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{message.job_name}</div>
                      <div className="text-xs text-gray-500">ID: {message.job_id.substring(0, 8)}...</div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{message.source}</div>
                      {message.error_message && (
                        <div className="text-xs text-red-600 mt-1">
                          Error: {message.error_message}
                        </div>
                      )}
                      {message.retry_count > 0 && (
                        <div className="text-xs text-orange-600 mt-1">
                          Retries: {message.retry_count}
                        </div>
                      )}
                    </td>
                    {allPipelineSteps.map((step) => {
                      const stepStatus = message.step_statuses[step] || 'pending'
                      const isCurrent = message.current_step === step && message.status === 'processing'
                      const hasStep = message.pipeline_steps.includes(step)
                      
                      return (
                        <td key={step} className="px-4 py-3">
                          {hasStep ? (
                            <div
                              className={`
                                flex items-center justify-center gap-1 px-2 py-1 rounded border
                                ${getStepStatusColor(stepStatus, isCurrent)}
                              `}
                            >
                              {getStepStatusIcon(stepStatus)}
                              {isCurrent && (
                                <span className="text-xs font-medium text-blue-700">
                                  Current
                                </span>
                              )}
                            </div>
                          ) : (
                            <div className="flex items-center justify-center">
                              <span className="text-gray-300">-</span>
                            </div>
                          )}
                        </td>
                      )
                    })}
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span
                        className={`
                          px-2 py-1 text-xs font-medium rounded-full
                          ${
                            message.status === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : message.status === 'failed'
                              ? 'bg-red-100 text-red-800'
                              : message.status === 'processing'
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-gray-100 text-gray-800'
                          }
                        `}
                      >
                        {message.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {(currentPage - 1) * rowsPerPage + 1} to{' '}
            {Math.min(currentPage * rowsPerPage, total)} of {total} results
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-2 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-4 py-2 text-sm text-gray-700">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-2 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default QueueTab
