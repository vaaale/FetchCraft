import { useState } from 'react'
import { Database, Briefcase } from 'lucide-react'
import QueueTab from './components/QueueTab'
import JobsTab from './components/JobsTab'

type Tab = 'jobs' | 'queue'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('jobs')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Database className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Fetchcraft Admin</h1>
                <p className="text-sm text-gray-600">Document Ingestion Administration</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('jobs')}
                className={`
                  flex items-center px-6 py-3 text-sm font-medium border-b-2 transition-colors
                  ${
                    activeTab === 'jobs'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Briefcase className="w-4 h-4 mr-2" />
                Jobs
              </button>
              <button
                onClick={() => setActiveTab('queue')}
                className={`
                  flex items-center px-6 py-3 text-sm font-medium border-b-2 transition-colors
                  ${
                    activeTab === 'queue'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Database className="w-4 h-4 mr-2" />
                Queue Messages
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'jobs' && <JobsTab />}
            {activeTab === 'queue' && <QueueTab />}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
