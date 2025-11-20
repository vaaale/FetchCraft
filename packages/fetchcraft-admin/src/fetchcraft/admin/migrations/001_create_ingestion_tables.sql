-- Migration: Create ingestion job and document tracking tables
-- Description: Tables for tracking ingestion jobs and document processing status

-- Create ingestion_jobs table
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    document_root TEXT NOT NULL,
    status TEXT NOT NULL,
    pipeline_steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for ingestion_jobs
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status 
    ON ingestion_jobs(status);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_pipeline_name 
    ON ingestion_jobs(pipeline_name);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at 
    ON ingestion_jobs(created_at DESC);

-- Create ingestion_documents table
CREATE TABLE IF NOT EXISTS ingestion_documents (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    current_step TEXT,
    step_statuses JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    error_message TEXT,
    error_step TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for ingestion_documents
CREATE INDEX IF NOT EXISTS idx_ingestion_documents_job_id 
    ON ingestion_documents(job_id);

CREATE INDEX IF NOT EXISTS idx_ingestion_documents_status 
    ON ingestion_documents(status);

CREATE INDEX IF NOT EXISTS idx_ingestion_documents_job_status 
    ON ingestion_documents(job_id, status);

CREATE INDEX IF NOT EXISTS idx_ingestion_documents_source 
    ON ingestion_documents(job_id, source);

-- Comments
COMMENT ON TABLE ingestion_jobs IS 'Tracks ingestion jobs and their overall status';
COMMENT ON TABLE ingestion_documents IS 'Tracks individual documents within ingestion jobs';
