import { useState, useEffect } from 'react'
import { X, Folder, FolderOpen, Loader } from 'lucide-react'
import { apiV2, DirectoryItem } from '../api_v2'

interface CreateJobModalProps {
  onClose: () => void
  onJobCreated: () => void
}

export default function CreateJobModal({ onClose, onJobCreated }: CreateJobModalProps) {
  const [jobName, setJobName] = useState('')
  const [currentPath, setCurrentPath] = useState('')
  const [directories, setDirectories] = useState<DirectoryItem[]>([])
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [browsing, setBrowsing] = useState(false)

  const loadDirectories = async (path: string = '') => {
    try {
      setBrowsing(true)
      const response = await apiV2.listDirectories(path)
      setDirectories(response.items)
      setCurrentPath(response.current_path)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load directories')
    } finally {
      setBrowsing(false)
    }
  }

  useEffect(() => {
    loadDirectories()
  }, [])

  const handleNavigate = (item: DirectoryItem) => {
    if (item.is_directory) {
      loadDirectories(item.path)
      
      // Auto-fill job name if empty
      if (!jobName.trim()) {
        setJobName(toCamelCase(item.name))
      }
    }
  }

  const toCamelCase = (str: string): string => {
    return str
      .split(/[-_\s]+/)
      .map((word, index) => 
        index === 0 
          ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
          : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
      )
      .join(' ')
  }

  const handleSelectPath = () => {
    setSelectedPath(currentPath)
    
    // Auto-fill job name if empty
    if (!jobName.trim() && currentPath) {
      const folderName = currentPath.split('/').pop() || ''
      if (folderName) {
        setJobName(toCamelCase(folderName))
      }
    }
  }

  const handleGoUp = () => {
    const parts = currentPath.split('/')
    parts.pop()
    const parentPath = parts.join('/')
    loadDirectories(parentPath)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!jobName.trim()) {
      setError('Job name is required')
      return
    }
    
    if (selectedPath === null) {
      setError('Please select a source path')
      return
    }

    try {
      setLoading(true)
      setError(null)
      await apiV2.createJob({
        name: jobName,
        source_path: selectedPath,
      })
      onJobCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create job')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Create Ingestion Job</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Job Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Job Name
              </label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g., Weekly Document Ingestion"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>

            {/* Source Path Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Source Path
              </label>
              
              {selectedPath !== null ? (
                <div className="bg-blue-50 border border-blue-200 rounded-md p-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Folder className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-900">
                      {selectedPath || '(root)'}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedPath(null)}
                    className="text-xs text-blue-600 hover:text-blue-800"
                  >
                    Change
                  </button>
                </div>
              ) : (
                <div className="border border-gray-300 rounded-md overflow-hidden">
                  {/* Current Path */}
                  <div className="bg-gray-50 px-4 py-2 border-b border-gray-200 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-gray-700">
                      <FolderOpen className="w-4 h-4" />
                      <span className="font-medium">
                        {currentPath || '(root)'}
                      </span>
                    </div>
                    <div className="flex gap-2">
                      {currentPath && (
                        <button
                          type="button"
                          onClick={handleGoUp}
                          className="text-xs text-gray-600 hover:text-gray-900"
                        >
                          ← Up
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={handleSelectPath}
                        className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                      >
                        Select This Folder
                      </button>
                    </div>
                  </div>

                  {/* Directory Listing */}
                  <div className="max-h-64 overflow-y-auto">
                    {browsing ? (
                      <div className="flex items-center justify-center p-8">
                        <Loader className="w-6 h-6 text-blue-600 animate-spin" />
                      </div>
                    ) : directories.length === 0 ? (
                      <div className="p-8 text-center text-gray-500 text-sm">
                        No subdirectories found
                      </div>
                    ) : (
                      directories.map((item) => (
                        <button
                          key={item.path}
                          type="button"
                          onClick={() => handleNavigate(item)}
                          className={`
                            w-full px-4 py-2 flex items-center gap-2 hover:bg-gray-50 transition-colors
                            ${item.is_directory ? 'cursor-pointer' : 'cursor-default opacity-50'}
                          `}
                          disabled={!item.is_directory}
                        >
                          {item.is_directory ? (
                            <Folder className="w-4 h-4 text-blue-500" />
                          ) : (
                            <div className="w-4 h-4" />
                          )}
                          <span className="text-sm text-gray-700">{item.name}</span>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || selectedPath === null}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading && <Loader className="w-4 h-4 animate-spin" />}
              Create Job
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
