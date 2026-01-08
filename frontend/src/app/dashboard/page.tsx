'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
    Upload,
    Clock,
    CheckCircle,
    XCircle,
    Loader2,
    FileText,
    Trash2,
    RefreshCw
} from 'lucide-react';

interface Job {
    job_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    filename?: string;
    created_at: string;
    completed_at?: string;
}

export default function DashboardPage() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    const fetchJobs = async () => {
        try {
            const response = await fetch('/api/v1/jobs?limit=20');
            if (response.ok) {
                const data = await response.json();
                setJobs(data.jobs || []);
            }
        } catch (err) {
            console.error('Failed to fetch jobs:', err);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchJobs();
    }, []);

    const handleRefresh = () => {
        setRefreshing(true);
        fetchJobs();
    };

    const handleDelete = async (jobId: string) => {
        if (!confirm('Are you sure you want to delete this job?')) return;

        try {
            const response = await fetch(`/api/v1/extract/${jobId}`, {
                method: 'DELETE'
            });
            if (response.ok) {
                setJobs(jobs.filter(j => j.job_id !== jobId));
            }
        } catch (err) {
            console.error('Failed to delete job:', err);
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed':
                return <CheckCircle size={18} color="var(--color-success)" />;
            case 'failed':
                return <XCircle size={18} color="var(--color-error)" />;
            case 'processing':
                return <Loader2 size={18} color="var(--color-info)" className="spinner" />;
            default:
                return <Clock size={18} color="var(--color-text-tertiary)" />;
        }
    };

    const getStatusBadge = (status: string) => {
        const classes: Record<string, string> = {
            completed: 'badge-success',
            failed: 'badge-error',
            processing: 'badge-info',
            pending: 'badge-warning'
        };
        return `badge ${classes[status] || ''}`;
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    return (
        <div className="page">
            <div className="container">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 'var(--space-2xl)',
                        flexWrap: 'wrap',
                        gap: 'var(--space-md)'
                    }}
                >
                    <div>
                        <h1>Dashboard</h1>
                        <p style={{ marginTop: 'var(--space-sm)' }}>
                            View and manage your extraction jobs
                        </p>
                    </div>

                    <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                        <button
                            onClick={handleRefresh}
                            className="btn btn-secondary"
                            disabled={refreshing}
                        >
                            <RefreshCw size={18} className={refreshing ? 'spinner' : ''} />
                            Refresh
                        </button>
                        <Link href="/upload" className="btn btn-primary">
                            <Upload size={18} />
                            New Extraction
                        </Link>
                    </div>
                </motion.div>

                {/* Stats Cards */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                        gap: 'var(--space-md)',
                        marginBottom: 'var(--space-2xl)'
                    }}
                >
                    <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                        <div className="data-label">Total Jobs</div>
                        <div className="data-value">{jobs.length}</div>
                    </div>
                    <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                        <div className="data-label">Completed</div>
                        <div className="data-value" style={{ color: 'var(--color-success)' }}>
                            {jobs.filter(j => j.status === 'completed').length}
                        </div>
                    </div>
                    <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                        <div className="data-label">Processing</div>
                        <div className="data-value" style={{ color: 'var(--color-info)' }}>
                            {jobs.filter(j => j.status === 'processing').length}
                        </div>
                    </div>
                    <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                        <div className="data-label">Failed</div>
                        <div className="data-value" style={{ color: 'var(--color-error)' }}>
                            {jobs.filter(j => j.status === 'failed').length}
                        </div>
                    </div>
                </motion.div>

                {/* Jobs List */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card"
                >
                    <h3 style={{ marginBottom: 'var(--space-lg)' }}>Recent Jobs</h3>

                    {loading ? (
                        <div style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
                            <Loader2 size={32} className="spinner" />
                            <p style={{ marginTop: 'var(--space-md)' }}>Loading jobs...</p>
                        </div>
                    ) : jobs.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: 'var(--space-2xl)' }}>
                            <FileText size={48} color="var(--color-text-tertiary)" style={{ marginBottom: 'var(--space-md)' }} />
                            <h4>No jobs yet</h4>
                            <p style={{ marginBottom: 'var(--space-lg)' }}>
                                Upload your first UCS graph to get started
                            </p>
                            <Link href="/upload" className="btn btn-primary">
                                <Upload size={18} />
                                Upload Graph
                            </Link>
                        </div>
                    ) : (
                        <div style={{ overflowX: 'auto' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>Status</th>
                                        <th>Filename</th>
                                        <th>Job ID</th>
                                        <th>Created</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {jobs.map((job, index) => (
                                        <motion.tr
                                            key={job.job_id}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: index * 0.05 }}
                                        >
                                            <td>
                                                <span className={getStatusBadge(job.status)}>
                                                    {getStatusIcon(job.status)}
                                                    {job.status}
                                                </span>
                                            </td>
                                            <td>{job.filename || 'Unknown'}</td>
                                            <td>
                                                <code style={{
                                                    fontSize: '0.75rem',
                                                    background: 'var(--color-surface)',
                                                    padding: '2px 6px',
                                                    borderRadius: 'var(--radius-sm)'
                                                }}>
                                                    {job.job_id.slice(0, 8)}...
                                                </code>
                                            </td>
                                            <td>{formatDate(job.created_at)}</td>
                                            <td>
                                                <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                                                    {job.status === 'completed' && (
                                                        <Link
                                                            href={`/results/${job.job_id}`}
                                                            className="btn btn-ghost"
                                                            style={{ padding: 'var(--space-xs) var(--space-sm)' }}
                                                        >
                                                            View
                                                        </Link>
                                                    )}
                                                    <button
                                                        onClick={() => handleDelete(job.job_id)}
                                                        className="btn btn-ghost"
                                                        style={{
                                                            padding: 'var(--space-xs) var(--space-sm)',
                                                            color: 'var(--color-error)'
                                                        }}
                                                    >
                                                        <Trash2 size={16} />
                                                    </button>
                                                </div>
                                            </td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    );
}
