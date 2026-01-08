'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
    Upload,
    FileImage,
    X,
    CheckCircle,
    AlertCircle,
    Loader2,
    ArrowRight,
    Settings,
    Zap
} from 'lucide-react';

interface ExtractionResult {
    job_id: string;
    status: string;
    data_points: Array<{ strain: number; stress: number }>;
    features?: {
        peak_stress: number;
        failure_strain: number;
        initial_modulus?: number;
        secant_modulus_50?: number;
    };
    confidence?: {
        overall_score: number;
        grade: string;
        warnings: string[];
        recommendations: string[];
    };
    extraction_method?: string;
    processing_time_ms?: number;
}

export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<ExtractionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);

    // Settings
    const [method, setMethod] = useState('auto');
    const [applySmoothing, setApplySmoothing] = useState(true);
    const [showSettings, setShowSettings] = useState(false);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const uploadedFile = acceptedFiles[0];
        if (uploadedFile) {
            setFile(uploadedFile);
            setPreview(URL.createObjectURL(uploadedFile));
            setResult(null);
            setError(null);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/png': ['.png'],
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/tiff': ['.tiff', '.tif'],
            'image/bmp': ['.bmp']
        },
        maxFiles: 1,
        maxSize: 50 * 1024 * 1024 // 50MB
    });

    const processImage = async () => {
        if (!file) return;

        setIsProcessing(true);
        setError(null);
        setProgress(0);

        const progressInterval = setInterval(() => {
            setProgress(p => Math.min(p + 5, 90));
        }, 200);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(
                `/api/v1/extract?method=${method}&apply_smoothing=${applySmoothing}`,
                {
                    method: 'POST',
                    body: formData,
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Extraction failed');
            }

            const data = await response.json();
            setResult(data);
            setProgress(100);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            clearInterval(progressInterval);
            setIsProcessing(false);
        }
    };

    const clearFile = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        setProgress(0);
    };

    const getGradeClass = (grade: string) => {
        return `grade-${grade.toLowerCase()}`;
    };

    return (
        <div className="page">
            <div className="container" style={{ maxWidth: '900px' }}>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ marginBottom: 'var(--space-2xl)' }}
                >
                    <Link
                        href="/"
                        style={{
                            color: 'var(--color-text-tertiary)',
                            fontSize: '0.875rem',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 'var(--space-xs)',
                            marginBottom: 'var(--space-md)'
                        }}
                    >
                        ← Back to Home
                    </Link>
                    <h1>Upload UCS Graph</h1>
                    <p style={{ marginTop: 'var(--space-sm)' }}>
                        Upload a UCS test graph image to extract stress-strain data
                    </p>
                </motion.div>

                {/* Upload Zone */}
                <AnimatePresence mode="wait">
                    {!file ? (
                        <motion.div
                            key="dropzone"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                        >
                            <div
                                {...getRootProps()}
                                className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
                            >
                                <input {...getInputProps()} />
                                <div className="upload-icon">
                                    <Upload size={32} />
                                </div>
                                <h3 style={{ marginBottom: 'var(--space-sm)' }}>
                                    {isDragActive ? 'Drop your image here' : 'Drag & drop your UCS graph'}
                                </h3>
                                <p style={{ marginBottom: 'var(--space-md)' }}>
                                    or click to browse files
                                </p>
                                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-tertiary)' }}>
                                    Supports PNG, JPG, TIFF • Max 50MB
                                </p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="preview"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="glass-card"
                        >
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'flex-start',
                                marginBottom: 'var(--space-lg)'
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
                                    <div style={{
                                        width: '48px',
                                        height: '48px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        background: 'var(--color-surface)',
                                        borderRadius: 'var(--radius-md)'
                                    }}>
                                        <FileImage size={24} />
                                    </div>
                                    <div>
                                        <h4>{file.name}</h4>
                                        <p style={{ fontSize: '0.875rem' }}>
                                            {(file.size / 1024 / 1024).toFixed(2)} MB
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={clearFile}
                                    className="btn btn-ghost"
                                    disabled={isProcessing}
                                >
                                    <X size={20} />
                                </button>
                            </div>

                            {/* Image Preview */}
                            <div style={{
                                position: 'relative',
                                width: '100%',
                                height: '300px',
                                background: 'var(--color-bg-tertiary)',
                                borderRadius: 'var(--radius-md)',
                                overflow: 'hidden',
                                marginBottom: 'var(--space-lg)'
                            }}>
                                {preview && (
                                    <img
                                        src={preview}
                                        alt="Preview"
                                        style={{
                                            width: '100%',
                                            height: '100%',
                                            objectFit: 'contain'
                                        }}
                                    />
                                )}
                            </div>

                            {/* Settings Toggle */}
                            <div style={{ marginBottom: 'var(--space-lg)' }}>
                                <button
                                    onClick={() => setShowSettings(!showSettings)}
                                    className="btn btn-ghost"
                                    style={{ marginBottom: 'var(--space-md)' }}
                                >
                                    <Settings size={16} />
                                    {showSettings ? 'Hide Settings' : 'Show Settings'}
                                </button>

                                <AnimatePresence>
                                    {showSettings && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            style={{
                                                background: 'var(--color-surface)',
                                                borderRadius: 'var(--radius-md)',
                                                padding: 'var(--space-lg)',
                                                overflow: 'hidden'
                                            }}
                                        >
                                            <div style={{ display: 'grid', gap: 'var(--space-md)' }}>
                                                <div>
                                                    <label style={{
                                                        display: 'block',
                                                        marginBottom: 'var(--space-xs)',
                                                        fontSize: '0.875rem',
                                                        color: 'var(--color-text-secondary)'
                                                    }}>
                                                        Extraction Method
                                                    </label>
                                                    <select
                                                        value={method}
                                                        onChange={(e) => setMethod(e.target.value)}
                                                        style={{
                                                            width: '100%',
                                                            padding: 'var(--space-sm) var(--space-md)',
                                                            background: 'var(--color-bg-tertiary)',
                                                            border: '1px solid var(--color-border)',
                                                            borderRadius: 'var(--radius-md)',
                                                            color: 'var(--color-text-primary)',
                                                            fontSize: '0.875rem'
                                                        }}
                                                    >
                                                        <option value="auto">Auto (Recommended)</option>
                                                        <option value="contour">Contour Tracing</option>
                                                        <option value="unet">U-Net Segmentation</option>
                                                        <option value="hybrid">Hybrid</option>
                                                    </select>
                                                </div>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                                                    <input
                                                        type="checkbox"
                                                        id="smoothing"
                                                        checked={applySmoothing}
                                                        onChange={(e) => setApplySmoothing(e.target.checked)}
                                                        style={{ width: '16px', height: '16px' }}
                                                    />
                                                    <label htmlFor="smoothing" style={{ fontSize: '0.875rem' }}>
                                                        Apply Savitzky-Golay Smoothing
                                                    </label>
                                                </div>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            {/* Progress Bar */}
                            {isProcessing && (
                                <div style={{ marginBottom: 'var(--space-lg)' }}>
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        marginBottom: 'var(--space-xs)'
                                    }}>
                                        <span style={{ fontSize: '0.875rem' }}>Processing...</span>
                                        <span style={{ fontSize: '0.875rem' }}>{progress}%</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div
                                            className="progress-bar-fill"
                                            style={{ width: `${progress}%` }}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Error Message */}
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 'var(--space-sm)',
                                        padding: 'var(--space-md)',
                                        background: 'rgba(239, 68, 68, 0.1)',
                                        border: '1px solid rgba(239, 68, 68, 0.3)',
                                        borderRadius: 'var(--radius-md)',
                                        marginBottom: 'var(--space-lg)'
                                    }}
                                >
                                    <AlertCircle size={20} color="var(--color-error)" />
                                    <span style={{ color: 'var(--color-error)' }}>{error}</span>
                                </motion.div>
                            )}

                            {/* Action Buttons */}
                            <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
                                <button
                                    onClick={processImage}
                                    disabled={isProcessing}
                                    className="btn btn-primary btn-lg"
                                    style={{ flex: 1 }}
                                >
                                    {isProcessing ? (
                                        <>
                                            <Loader2 size={20} className="animate-spin" />
                                            Processing...
                                        </>
                                    ) : (
                                        <>
                                            <Zap size={20} />
                                            Extract Data
                                        </>
                                    )}
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results Section */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            style={{ marginTop: 'var(--space-xl)' }}
                        >
                            {/* Success Header */}
                            <div
                                className="glass-card"
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    marginBottom: 'var(--space-lg)'
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
                                    <div style={{
                                        width: '48px',
                                        height: '48px',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        background: 'rgba(16, 185, 129, 0.1)',
                                        borderRadius: 'var(--radius-full)'
                                    }}>
                                        <CheckCircle size={24} color="var(--color-success)" />
                                    </div>
                                    <div>
                                        <h3>Extraction Complete</h3>
                                        <p style={{ fontSize: '0.875rem' }}>
                                            {result.data_points.length} data points extracted in{' '}
                                            {result.processing_time_ms?.toFixed(0)}ms
                                        </p>
                                    </div>
                                </div>

                                {result.confidence && (
                                    <div
                                        className={`grade-display ${getGradeClass(result.confidence.grade)}`}
                                    >
                                        {result.confidence.grade}
                                    </div>
                                )}
                            </div>

                            {/* Features Grid */}
                            {result.features && (
                                <div className="glass-card" style={{ marginBottom: 'var(--space-lg)' }}>
                                    <h4 style={{ marginBottom: 'var(--space-lg)' }}>Extracted Features</h4>
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
                                                {result.features.failure_strain.toFixed(3)}
                                                <span className="data-unit">%</span>
                                            </div>
                                        </div>
                                        {result.features.initial_modulus && (
                                            <div className="data-item">
                                                <div className="data-label">Initial Modulus</div>
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
                                    </div>
                                </div>
                            )}

                            {/* Confidence Details */}
                            {result.confidence && (
                                <div className="glass-card" style={{ marginBottom: 'var(--space-lg)' }}>
                                    <h4 style={{ marginBottom: 'var(--space-lg)' }}>Confidence Analysis</h4>

                                    <div style={{ marginBottom: 'var(--space-lg)' }}>
                                        <div style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            marginBottom: 'var(--space-xs)'
                                        }}>
                                            <span>Overall Confidence</span>
                                            <span>{(result.confidence.overall_score * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="progress-bar">
                                            <div
                                                className="progress-bar-fill"
                                                style={{ width: `${result.confidence.overall_score * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    {result.confidence.warnings.length > 0 && (
                                        <div style={{ marginBottom: 'var(--space-md)' }}>
                                            <h5 style={{ marginBottom: 'var(--space-sm)', color: 'var(--color-warning)' }}>
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
                                            <h5 style={{ marginBottom: 'var(--space-sm)', color: 'var(--color-info)' }}>
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

                            {/* View Full Results Button */}
                            <Link
                                href={`/results/${result.job_id}`}
                                className="btn btn-primary btn-lg"
                                style={{ width: '100%' }}
                            >
                                View Full Results & Download CSV
                                <ArrowRight size={20} />
                            </Link>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
