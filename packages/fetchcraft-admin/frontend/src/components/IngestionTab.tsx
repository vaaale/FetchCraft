import { useEffect, useState } from 'react'
import { Play, Square, RefreshCw, AlertCircle, CheckCircle, XCircle } from 'lucide-react'
import { api, IngestionStatus } from '../api'

const IngestionTab = () => {
  const [status, setStatus] = useState<IngestionStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actionLoading, setActionLoading] = useState(false)

  const fetchStatus = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getIngestionStatus()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    // Auto-refresh every 2 seconds
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleStart = async () => {
    setActionLoading(true)
    setError(null)
    try {
      const data = await api.startIngestion()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start ingestion')
    } finally {
      setActionLoading(false)
    }
  }

  const handleStop = async () => {
    setActionLoading(true)
    setError(null)
    try {
      const data = await api.stopIngestion()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop ingestion')
    } finally {
      setActionLoading(false)
    }
  }

  const isRunning = status?.status === 'running'
  const isError = status?.status === 'error'

  const getStatusIcon = () => {
    if (isRunning) return <CheckCircle className="w-8 h-8 text-green-600" />
    if (isError) return <XCircle className="w-8 h-8 text-red-600" />
    return <Square className="w-8 h-8 text-gray-400" />
  }

  const getStatusColor = () => {
    if (isRunning) return 'border-green-200 bg-green-50'
    if (isError) return 'border-red-200 bg-red-50'
    return 'border-gray-200 bg-gray-50'
  }

  const getStatusTextColor = () => {
    if (isRunning) return 'text-green-900'
    if (isError) return 'text-red-900'
    return 'text-gray-900'
  }

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <div className={`border rounded-lg p-6 ${getStatusColor()}`}>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            {getStatusIcon()}
            <div>
              <h2 className={`text-xl font-bold ${getStatusTextColor()}`}>
                Ingestion Job Status
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                {loading ? (
                  <span className="flex items-center">
                    <RefreshCw className="w-3 h-3 animate-spin mr-1" />
                    Checking status...
                  </span>
                ) : (
                  <>
                    Status: <span className="font-semibold">{status?.status || 'Unknown'}</span>
                    {status?.pid && (
                      <span className="ml-2">
                        | PID: <span className="font-mono">{status.pid}</span>
                      </span>
                    )}
                  </>
                )}
              </p>
            </div>
          </div>
          <button
            onClick={fetchStatus}
            disabled={loading}
            className="p-2 border border-gray-300 bg-white rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Refresh status"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
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

      {/* Control Panel */}
      <div className="border border-gray-200 rounded-lg p-6 bg-white">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Job Control</h3>
        <div className="flex space-x-4">
          <button
            onClick={handleStart}
            disabled={isRunning || actionLoading}
            className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            {actionLoading && !isRunning ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            <span className="font-medium">Start Ingestion</span>
          </button>
          <button
            onClick={handleStop}
            disabled={!isRunning || actionLoading}
            className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            {actionLoading && isRunning ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Square className="w-5 h-5" />
            )}
            <span className="font-medium">Stop Ingestion</span>
          </button>
        </div>
      </div>

      {/* Information */}
      <div className="border border-blue-200 rounded-lg p-6 bg-blue-50">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">About Ingestion Jobs</h3>
        <div className="text-sm text-blue-800 space-y-2">
          <p>
            <strong>Start:</strong> Begins processing documents from the configured path and adds
            them to the vector database. The job runs in a separate process and will continue until
            all documents are processed.
          </p>
          <p>
            <strong>Stop:</strong> Gracefully stops the ingestion job. Any documents currently being
            processed may not be completed.
          </p>
          <p className="text-xs text-blue-700 mt-3">
            <strong>Note:</strong> Make sure your database and document sources are properly
            configured before starting an ingestion job.
          </p>
        </div>
      </div>

      {/* Status Details */}
      {status && (
        <div className="border border-gray-200 rounded-lg p-6 bg-white">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Status Details</h3>
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <dt className="text-sm font-medium text-gray-500">Current Status</dt>
              <dd className="mt-1 text-sm text-gray-900">
                <span
                  className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                    isRunning
                      ? 'bg-green-100 text-green-800'
                      : isError
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {status.status.toUpperCase()}
                </span>
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Process ID</dt>
              <dd className="mt-1 text-sm text-gray-900 font-mono">
                {status.pid || 'N/A'}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  )
}

export default IngestionTab
