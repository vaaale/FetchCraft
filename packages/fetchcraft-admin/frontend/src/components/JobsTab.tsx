import { useState, useEffect } from 'react'
import { Folder, Plus, RefreshCw, CheckCircle, XCircle, Clock, Loader, Trash2, RotateCcw } from 'lucide-react'
import { apiV2, Job } from '../api_v2'
import CreateJobModal from './CreateJobModal'
import JobDetailsModal from './JobDetailsModal'

export default function JobsTab() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [filterStatus, setFilterStatus] = useState<string>('all')

  const fetchJobs = async () => {
    try {
      setLoading(true)
      const status = filterStatus === 'all' ? undefined : filterStatus
      const response = await apiV2.listJobs({ status })
      setJobs(response.jobs)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [filterStatus])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
      case 'running':
        return <Loader className="w-5 h-5 text-blue-600 animate-spin" />
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-600" />
      default:
        return <Clock className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const baseClasses = 'px-2 py-1 text-xs font-medium rounded-full'
    switch (status) {
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-800`
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-800`
      case 'running':
        return `${baseClasses} bg-blue-100 text-blue-800`
      case 'pending':
        return `${baseClasses} bg-yellow-100 text-yellow-800`
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`
    }
  }

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString()
  }

  const handleDeleteJob = async (job: Job) => {
    if (!confirm(`Are you sure you want to delete job "${job.name}"? This will also delete all associated documents.`)) {
      return
    }

    try {
      await apiV2.deleteJob(job.id, true)
      fetchJobs()
    } catch (err) {
      alert(`Failed to delete job: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  const handleRestartJob = async (job: Job) => {
    if (!confirm(`Are you sure you want to restart job "${job.name}"? This will create a new job and re-process all documents.`)) {
      return
    }

    try {
      await apiV2.restartJob(job.id)
      fetchJobs()
    } catch (err) {
      alert(`Failed to restart job: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  return (
    <div className="space-y-4">
      {/* Header with Actions */}
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-900">Ingestion Jobs</h2>
        <div className="flex gap-2">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <button
            onClick={fetchJobs}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
          >
            <Plus className="w-4 h-4" />
            New Job
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Jobs Table */}
      {loading && jobs.length === 0 ? (
        <div className="flex items-center justify-center p-8">
          <Loader className="w-8 h-8 text-blue-600 animate-spin" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="text-center p-8 bg-gray-50 rounded-lg">
          <Folder className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">No jobs found</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="mt-4 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            Create your first job
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto bg-white rounded-lg border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Source Path
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {jobs.map((job) => {
                const duration = job.started_at && job.completed_at
                  ? `${Math.round((new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000)}s`
                  : job.started_at
                  ? 'Running...'
                  : 'N/A'

                return (
                  <tr key={job.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(job.status)}
                        <span className={getStatusBadge(job.status)}>
                          {job.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">{job.name}</div>
                      <div className="text-xs text-gray-500">ID: {job.id.slice(0, 8)}</div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-900">{job.source_path}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(job.created_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {duration}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setSelectedJob(job)}
                          className="text-blue-600 hover:text-blue-900 font-medium"
                        >
                          View Details
                        </button>
                        {(job.status === 'completed' || job.status === 'failed') && (
                          <>
                            <button
                              onClick={() => handleRestartJob(job)}
                              className="p-1 text-green-600 hover:text-green-900 hover:bg-green-50 rounded"
                              title="Restart job"
                            >
                              <RotateCcw className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleDeleteJob(job)}
                              className="p-1 text-red-600 hover:text-red-900 hover:bg-red-50 rounded"
                              title="Delete job"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Modals */}
      {showCreateModal && (
        <CreateJobModal
          onClose={() => setShowCreateModal(false)}
          onJobCreated={() => {
            setShowCreateModal(false)
            fetchJobs()
          }}
        />
      )}

      {selectedJob && (
        <JobDetailsModal
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
          onJobUpdated={fetchJobs}
        />
      )}
    </div>
  )
}
