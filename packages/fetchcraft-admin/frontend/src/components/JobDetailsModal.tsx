import { useState, useEffect } from 'react'
import { X, RefreshCw, AlertCircle, CheckCircle, Loader, Clock, RotateCcw } from 'lucide-react'
import { apiV2, Job, Document } from '../api_v2'

interface JobDetailsModalProps {
  job: Job
  onClose: () => void
  onJobUpdated: () => void
}

export default function JobDetailsModal({ job, onClose, onJobUpdated }: JobDetailsModalProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retrying, setRetrying] = useState(false)
  const [filterStatus, setFilterStatus] = useState<string>('all')

  const fetchDocuments = async () => {
    try {
      setLoading(true)
      const status = filterStatus === 'all' ? undefined : filterStatus
      const response = await apiV2.listJobDocuments(job.id, { status, limit: 1000 })
      setDocuments(response.documents)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
    const interval = setInterval(fetchDocuments, 3000) // Refresh every 3 seconds
    return () => clearInterval(interval)
  }, [job.id, filterStatus])

  const handleRetry = async () => {
    try {
      setRetrying(true)
      const response = await apiV2.retryFailedDocuments(job.id)
      alert(`Retrying ${response.retried_count} failed documents`)
      fetchDocuments()
      onJobUpdated()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to retry documents')
    } finally {
      setRetrying(false)
    }
  }

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

  const statusCounts = documents.reduce((acc, doc) => {
    acc[doc.status] = (acc[doc.status] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const failedCount = statusCounts.failed || 0

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-7xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{job.name}</h2>
            <p className="text-sm text-gray-600 mt-1">
              Source: {job.source_path} • Status: <span className="font-medium">{job.status}</span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 border-b border-gray-200">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{documents.length}</div>
            <div className="text-xs text-gray-600">Total</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{statusCounts.completed || 0}</div>
            <div className="text-xs text-gray-600">Completed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{statusCounts.processing || 0}</div>
            <div className="text-xs text-gray-600">Processing</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{statusCounts.failed || 0}</div>
            <div className="text-xs text-gray-600">Failed</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex gap-2">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Documents</option>
              <option value="pending">Pending</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          <div className="flex gap-2">
            {failedCount > 0 && (
              <button
                onClick={handleRetry}
                disabled={retrying}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-orange-600 rounded-md hover:bg-orange-700 disabled:opacity-50"
              >
                <RotateCcw className={`w-4 h-4 ${retrying ? 'animate-spin' : ''}`} />
                Retry Failed ({failedCount})
              </button>
            )}
            <button
              onClick={fetchDocuments}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mx-4 mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {/* Documents Table */}
        <div className="flex-1 overflow-auto">
          {loading && documents.length === 0 ? (
            <div className="flex items-center justify-center p-8">
              <Loader className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center p-8 text-gray-500">
              No documents found
            </div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Document
                  </th>
                  {job.pipeline_steps.map((step) => (
                    <th
                      key={step}
                      className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase"
                    >
                      {step.replace('sink:', '')}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {documents.map((doc) => (
                  <tr key={doc.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{doc.source}</div>
                      {doc.error_message && (
                        <div className="text-xs text-red-600 mt-1">
                          Error: {doc.error_message}
                        </div>
                      )}
                      {doc.retry_count > 0 && (
                        <div className="text-xs text-orange-600 mt-1">
                          Retries: {doc.retry_count}
                        </div>
                      )}
                    </td>
                    {job.pipeline_steps.map((step) => {
                      const stepStatus = doc.step_statuses[step] || 'pending'
                      const isCurrent = doc.current_step === step
                      return (
                        <td key={step} className="px-4 py-3">
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
                        </td>
                      )
                    })}
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span
                        className={`
                          px-2 py-1 text-xs font-medium rounded-full
                          ${
                            doc.status === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : doc.status === 'failed'
                              ? 'bg-red-100 text-red-800'
                              : doc.status === 'processing'
                              ? 'bg-blue-100 text-blue-800'
                              : 'bg-gray-100 text-gray-800'
                          }
                        `}
                      >
                        {doc.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end p-4 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
