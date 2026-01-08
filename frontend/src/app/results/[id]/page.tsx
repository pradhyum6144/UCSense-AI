'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceDot,
    Legend
} from 'recharts';
import {
    Download,
    ArrowLeft,
    Loader2,
    AlertCircle,
    Table,
    BarChart3,
    Info
} from 'lucide-react';

interface DataPoint {
    strain: number;
    stress: number;
    pixel_x?: number;
    pixel_y?: number;
}

interface ExtractionResult {
    job_id: string;
    status: string;
    data_points: DataPoint[];
    features?: {
        peak_stress: number;
        failure_strain: number;
        initial_modulus?: number;
        secant_modulus_50?: number;
        energy_to_peak?: number;
        post_peak_detected?: boolean;
    };
    confidence?: {
        overall_score: number;
        grade: string;
        factors: {
            ocr_confidence: number;
            curve_smoothness: number;
            axis_detection: number;
            image_quality: number;
            data_validity: number;
            extraction_method: number;
        };
        warnings: string[];
        recommendations: string[];
    };
    x_axis?: { label?: string; unit?: string; min_value?: number; max_value?: number };
    y_axis?: { label?: string; unit?: string; min_value?: number; max_value?: number };
    extraction_method?: string;
    processing_time_ms?: number;
}

export default function ResultsPage() {
    const params = useParams();
    const jobId = params.id as string;

    const [result, setResult] = useState<ExtractionResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'chart' | 'table'>('chart');

    useEffect(() => {
        const fetchResult = async () => {
            try {
                const response = await fetch(`/api/v1/extract/${jobId}`);
                if (!response.ok) {
                    throw new Error('Failed to fetch results');
                }
                const data = await response.json();
                setResult(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred');
            } finally {
                setLoading(false);
            }
        };

        fetchResult();
    }, [jobId]);

    const downloadCSV = async () => {
        window.open(`/api/v1/extract/${jobId}/csv`, '_blank');
    };

    const getGradeClass = (grade: string) => `grade-${grade.toLowerCase()}`;

    if (loading) {
        return (
            <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                    <Loader2 size={48} className="spinner" style={{ marginBottom: 'var(--space-md)' }} />
                    <p>Loading results...</p>
                </div>
            </div>
        );
    }

    if (error || !result) {
        return (
            <div className="page">
                <div className="container" style={{ maxWidth: '600px', textAlign: 'center' }}>
                    <AlertCircle size={48} color="var(--color-error)" style={{ marginBottom: 'var(--space-md)' }} />
                    <h2>Error Loading Results</h2>
                    <p style={{ marginBottom: 'var(--space-xl)' }}>{error || 'Results not found'}</p>
                    <Link href="/upload" className="btn btn-primary">
                        <ArrowLeft size={20} />
                        Try Again
                    </Link>
                </div>
            </div>
        );
    }

    // Prepare chart data with peak annotation
    const chartData = result.data_points.map(p => ({
        strain: p.strain,
        stress: p.stress
    }));

    const peakPoint = result.features ? {
        strain: result.features.failure_strain,
        stress: result.features.peak_stress
    } : null;

    return (
        <div className="page">
            <div className="container">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ marginBottom: 'var(--space-2xl)' }}
                >
                    <Link
                        href="/upload"
                        style={{
                            color: 'var(--color-text-tertiary)',
                            fontSize: '0.875rem',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 'var(--space-xs)',
                            marginBottom: 'var(--space-md)'
                        }}
                    >
                        <ArrowLeft size={16} />
                        Back to Upload
                    </Link>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
                        <div>
                            <h1>Extraction Results</h1>
                            <p style={{ marginTop: 'var(--space-sm)' }}>
                                Job ID: <code style={{
                                    background: 'var(--color-surface)',
                                    padding: '2px 8px',
                                    borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.875rem'
                                }}>{jobId}</code>
                            </p>
                        </div>

                        <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                            <button onClick={downloadCSV} className="btn btn-primary">
                                <Download size={18} />
                                Download CSV
                            </button>
                        </div>
                    </div>
                </motion.div>

                {/* Summary Cards */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                        gap: 'var(--space-md)',
                        marginBottom: 'var(--space-xl)'
                    }}
                >
                    <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                        <div className="data-label">Data Points</div>
                        <div className="data-value">{result.data_points.length}</div>
                    </div>

                    {result.features && (
                        <>
                            <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                                <div className="data-label">Peak UCS</div>
                                <div className="data-value">
                                    {result.features.peak_stress.toFixed(1)}
                                    <span className="data-unit">kN/m²</span>
                                </div>
                            </div>

                            <div className="glass-card" style={{ padding: 'var(--space-lg)' }}>
                                <div className="data-label">Failure Strain</div>
                                <div className="data-value">
                                    {result.features.failure_strain.toFixed(3)}
                                    <span className="data-unit">%</span>
                                </div>
                            </div>
                        </>
                    )}

                    {result.confidence && (
                        <div className="glass-card" style={{
                            padding: 'var(--space-lg)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 'var(--space-md)'
                        }}>
                            <div className={`grade-display ${getGradeClass(result.confidence.grade)}`} style={{ width: '60px', height: '60px', fontSize: '1.5rem' }}>
                                {result.confidence.grade}
                            </div>
                            <div>
                                <div className="data-label">Confidence</div>
                                <div className="data-value" style={{ fontSize: '1.25rem' }}>
                                    {(result.confidence.overall_score * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    )}
                </motion.div>

                {/* Chart/Table Toggle */}
                <div style={{
                    display: 'flex',
                    gap: 'var(--space-sm)',
                    marginBottom: 'var(--space-lg)'
                }}>
                    <button
                        onClick={() => setActiveTab('chart')}
                        className={`btn ${activeTab === 'chart' ? 'btn-primary' : 'btn-secondary'}`}
                    >
                        <BarChart3 size={18} />
                        Chart
                    </button>
                    <button
                        onClick={() => setActiveTab('table')}
                        className={`btn ${activeTab === 'table' ? 'btn-primary' : 'btn-secondary'}`}
                    >
                        <Table size={18} />
                        Data Table
                    </button>
                </div>

                {/* Chart View */}
                {activeTab === 'chart' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="glass-card"
                        style={{ marginBottom: 'var(--space-xl)' }}
                    >
                        <h3 style={{ marginBottom: 'var(--space-lg)' }}>Stress-Strain Curve</h3>
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                                    <XAxis
                                        dataKey="strain"
                                        name="Strain"
                                        unit="%"
                                        stroke="var(--color-text-tertiary)"
                                        tickFormatter={(v) => v.toFixed(2)}
                                        label={{
                                            value: 'Strain (%)',
                                            position: 'bottom',
                                            fill: 'var(--color-text-secondary)'
                                        }}
                                    />
                                    <YAxis
                                        name="Stress"
                                        unit=" kN/m²"
                                        stroke="var(--color-text-tertiary)"
                                        label={{
                                            value: 'Stress (kN/m²)',
                                            angle: -90,
                                            position: 'insideLeft',
                                            fill: 'var(--color-text-secondary)'
                                        }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: 'var(--color-bg-elevated)',
                                            border: '1px solid var(--color-border)',
                                            borderRadius: 'var(--radius-md)'
                                        }}
                                        formatter={(value: number, name: string) => [
                                            `${value.toFixed(2)} ${name === 'stress' ? 'kN/m²' : '%'}`,
                                            name === 'stress' ? 'Stress' : 'Strain'
                                        ]}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="stress"
                                        stroke="var(--chart-primary)"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Stress"
                                    />
                                    {peakPoint && (
                                        <ReferenceDot
                                            x={peakPoint.strain}
                                            y={peakPoint.stress}
                                            r={8}
                                            fill="var(--color-error)"
                                            stroke="white"
                                            strokeWidth={2}
                                        />
                                    )}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        {peakPoint && (
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--space-sm)',
                                marginTop: 'var(--space-md)',
                                padding: 'var(--space-sm) var(--space-md)',
                                background: 'var(--color-surface)',
                                borderRadius: 'var(--radius-md)',
                                fontSize: '0.875rem'
                            }}>
                                <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--color-error)' }} />
                                <span>Peak UCS: {peakPoint.stress.toFixed(1)} kN/m² at {peakPoint.strain.toFixed(3)}% strain</span>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Table View */}
                {activeTab === 'table' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="glass-card"
                        style={{ marginBottom: 'var(--space-xl)', overflowX: 'auto' }}
                    >
                        <h3 style={{ marginBottom: 'var(--space-lg)' }}>Data Table</h3>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Strain (%)</th>
                                    <th>Stress (kN/m²)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.data_points.slice(0, 100).map((point, index) => (
                                    <tr key={index}>
                                        <td>{index + 1}</td>
                                        <td>{point.strain.toFixed(4)}</td>
                                        <td>{point.stress.toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {result.data_points.length > 100 && (
                            <p style={{
                                marginTop: 'var(--space-md)',
                                fontSize: '0.875rem',
                                color: 'var(--color-text-tertiary)'
                            }}>
                                Showing first 100 of {result.data_points.length} points. Download CSV for full data.
                            </p>
                        )}
                    </motion.div>
                )}

                {/* Additional Features */}
                {result.features && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="glass-card"
                        style={{ marginBottom: 'var(--space-xl)' }}
                    >
                        <h3 style={{ marginBottom: 'var(--space-lg)' }}>
                            <Info size={20} style={{ marginRight: 'var(--space-sm)', verticalAlign: 'middle' }} />
                            Material Properties
                        </h3>
                        <div className="data-grid">
                            <div className="data-item">
                                <div className="data-label">Peak UCS</div>
                                <div className="data-value">
                                    {result.features.peak_stress.toFixed(1)}
                                    <span className="data-unit">kN/m²</span>
                                </div>
                            </div>
                            <div className="data-item">
                                <div className="data-label">Failure Strain</div>
                                <div className="data-value">
                                    {result.features.failure_strain.toFixed(4)}
                                    <span className="data-unit">%</span>
                                </div>
                            </div>
                            {result.features.initial_modulus && (
                                <div className="data-item">
                                    <div className="data-label">Initial Tangent Modulus</div>
                                    <div className="data-value">
                                        {result.features.initial_modulus.toFixed(0)}
                                        <span className="data-unit">kN/m²/%</span>
                                    </div>
                                </div>
                            )}
                            {result.features.secant_modulus_50 && (
                                <div className="data-item">
                                    <div className="data-label">Secant Modulus (50%)</div>
                                    <div className="data-value">
                                        {result.features.secant_modulus_50.toFixed(0)}
                                        <span className="data-unit">kN/m²/%</span>
                                    </div>
                                </div>
                            )}
                            {result.features.energy_to_peak && (
                                <div className="data-item">
                                    <div className="data-label">Energy to Peak</div>
                                    <div className="data-value">
                                        {result.features.energy_to_peak.toFixed(2)}
                                        <span className="data-unit">kJ/m³</span>
                                    </div>
                                </div>
                            )}
                            <div className="data-item">
                                <div className="data-label">Post-Peak Behavior</div>
                                <div className="data-value" style={{ fontSize: '1rem' }}>
                                    {result.features.post_peak_detected ? 'Captured' : 'Not Captured'}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* Confidence Details */}
                {result.confidence && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="glass-card"
                    >
                        <h3 style={{ marginBottom: 'var(--space-lg)' }}>Confidence Breakdown</h3>

                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                            gap: 'var(--space-md)',
                            marginBottom: 'var(--space-xl)'
                        }}>
                            {Object.entries(result.confidence.factors).map(([key, value]) => (
                                <div key={key} className="data-item">
                                    <div className="data-label">{key.replace(/_/g, ' ')}</div>
                                    <div style={{ marginTop: 'var(--space-xs)' }}>
                                        <div className="progress-bar" style={{ height: '6px' }}>
                                            <div
                                                className="progress-bar-fill"
                                                style={{ width: `${value * 100}%` }}
                                            />
                                        </div>
                                        <div style={{ fontSize: '0.875rem', marginTop: '4px' }}>
                                            {(value * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {(result.confidence.warnings.length > 0 || result.confidence.recommendations.length > 0) && (
                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                                gap: 'var(--space-lg)'
                            }}>
                                {result.confidence.warnings.length > 0 && (
                                    <div>
                                        <h5 style={{ color: 'var(--color-warning)', marginBottom: 'var(--space-sm)' }}>
                                            ⚠️ Warnings
                                        </h5>
                                        <ul style={{ paddingLeft: 'var(--space-lg)', fontSize: '0.875rem' }}>
                                            {result.confidence.warnings.map((w, i) => (
                                                <li key={i} style={{ marginBottom: 'var(--space-xs)' }}>{w}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {result.confidence.recommendations.length > 0 && (
                                    <div>
                                        <h5 style={{ color: 'var(--color-info)', marginBottom: 'var(--space-sm)' }}>
                                            💡 Recommendations
                                        </h5>
                                        <ul style={{ paddingLeft: 'var(--space-lg)', fontSize: '0.875rem' }}>
                                            {result.confidence.recommendations.map((r, i) => (
                                                <li key={i} style={{ marginBottom: 'var(--space-xs)' }}>{r}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        )}
                    </motion.div>
                )}
            </div>
        </div>
    );
}
