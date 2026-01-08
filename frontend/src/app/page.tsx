'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Upload, BarChart3, Zap, Target, Shield, Download } from 'lucide-react';

export default function HomePage() {
    const features = [
        {
            icon: <Upload size={24} />,
            title: 'Smart Upload',
            description: 'Drag & drop UCS graph images. Supports PNG, JPG, TIFF formats up to 50MB.'
        },
        {
            icon: <Zap size={24} />,
            title: 'AI Extraction',
            description: 'Hybrid OpenCV + U-Net pipeline automatically selects the best method.'
        },
        {
            icon: <Target size={24} />,
            title: '95%+ Accuracy',
            description: 'Confidence scoring with Savitzky-Golay smoothing ensures high precision.'
        },
        {
            icon: <BarChart3 size={24} />,
            title: 'Feature Detection',
            description: 'Automatic peak UCS, failure strain, and modulus calculation.'
        },
        {
            icon: <Shield size={24} />,
            title: 'Validation',
            description: 'Compare against ground truth CSV with MAE and R² metrics.'
        },
        {
            icon: <Download size={24} />,
            title: 'Export',
            description: 'Download results as CSV or JSON for further analysis.'
        }
    ];

    return (
        <div className="page">
            <div className="container">
                {/* Hero Section */}
                <motion.section
                    className="hero"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    style={{ textAlign: 'center', padding: 'var(--space-3xl) 0' }}
                >
                    <motion.div
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.5 }}
                        style={{
                            display: 'inline-flex',
                            padding: 'var(--space-xs) var(--space-md)',
                            background: 'rgba(99, 102, 241, 0.1)',
                            borderRadius: 'var(--radius-full)',
                            marginBottom: 'var(--space-lg)',
                            border: '1px solid rgba(99, 102, 241, 0.3)'
                        }}
                    >
                        <span style={{ color: 'var(--color-accent-primary)', fontSize: '0.875rem' }}>
                            ✨ Powered by AI & Computer Vision
                        </span>
                    </motion.div>

                    <h1 style={{
                        fontSize: 'clamp(2rem, 5vw, 3.5rem)',
                        marginBottom: 'var(--space-md)',
                        background: 'var(--color-accent-gradient)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text'
                    }}>
                        UCSense-AI
                    </h1>

                    <p style={{
                        fontSize: '1.25rem',
                        color: 'var(--color-text-secondary)',
                        maxWidth: '600px',
                        margin: '0 auto var(--space-xl)'
                    }}>
                        Transform geotechnical UCS graph images into precise digital data
                        with our intelligent extraction pipeline.
                    </p>

                    <div style={{ display: 'flex', gap: 'var(--space-md)', justifyContent: 'center', flexWrap: 'wrap' }}>
                        <Link href="/upload" className="btn btn-primary btn-lg">
                            <Upload size={20} />
                            Start Extraction
                        </Link>
                        <Link href="/dashboard" className="btn btn-secondary btn-lg">
                            <BarChart3 size={20} />
                            View Dashboard
                        </Link>
                    </div>
                </motion.section>

                {/* Features Grid */}
                <section style={{ padding: 'var(--space-3xl) 0' }}>
                    <motion.h2
                        style={{ textAlign: 'center', marginBottom: 'var(--space-2xl)' }}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        Powerful Features
                    </motion.h2>

                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                        gap: 'var(--space-lg)'
                    }}>
                        {features.map((feature, index) => (
                            <motion.div
                                key={feature.title}
                                className="glass-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 * index }}
                            >
                                <div style={{
                                    width: '48px',
                                    height: '48px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    background: 'var(--color-accent-gradient)',
                                    borderRadius: 'var(--radius-md)',
                                    marginBottom: 'var(--space-md)',
                                    color: 'white'
                                }}>
                                    {feature.icon}
                                </div>
                                <h4 style={{ marginBottom: 'var(--space-sm)' }}>{feature.title}</h4>
                                <p style={{ fontSize: '0.875rem' }}>{feature.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* How It Works */}
                <section style={{ padding: 'var(--space-3xl) 0' }}>
                    <motion.h2
                        style={{ textAlign: 'center', marginBottom: 'var(--space-2xl)' }}
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                    >
                        How It Works
                    </motion.h2>

                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 'var(--space-lg)',
                        maxWidth: '800px',
                        margin: '0 auto'
                    }}>
                        {[
                            { step: '01', title: 'Upload Image', desc: 'Upload your UCS graph image (scan or photo)' },
                            { step: '02', title: 'AI Processing', desc: 'Our hybrid pipeline rectifies, filters, and extracts data' },
                            { step: '03', title: 'OCR & Mapping', desc: 'Tesseract reads axis values and maps coordinates' },
                            { step: '04', title: 'Feature Analysis', desc: 'Automatic peak detection and feature extraction' },
                            { step: '05', title: 'Export Results', desc: 'Download precise stress-strain data as CSV' }
                        ].map((item, index) => (
                            <motion.div
                                key={item.step}
                                style={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    gap: 'var(--space-lg)'
                                }}
                                initial={{ opacity: 0, x: -20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <div style={{
                                    flexShrink: 0,
                                    width: '48px',
                                    height: '48px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    background: 'var(--color-surface)',
                                    borderRadius: 'var(--radius-full)',
                                    border: '2px solid var(--color-accent-primary)',
                                    fontWeight: 600,
                                    color: 'var(--color-accent-primary)'
                                }}>
                                    {item.step}
                                </div>
                                <div>
                                    <h4 style={{ marginBottom: 'var(--space-xs)' }}>{item.title}</h4>
                                    <p style={{ fontSize: '0.875rem' }}>{item.desc}</p>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* CTA Section */}
                <motion.section
                    className="glass-card"
                    style={{
                        textAlign: 'center',
                        margin: 'var(--space-3xl) 0',
                        background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05))'
                    }}
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                >
                    <h2 style={{ marginBottom: 'var(--space-md)' }}>
                        Ready to Digitize Your Data?
                    </h2>
                    <p style={{ marginBottom: 'var(--space-xl)', maxWidth: '500px', margin: '0 auto var(--space-xl)' }}>
                        Start extracting precise stress-strain data from your UCS graphs today.
                    </p>
                    <Link href="/upload" className="btn btn-primary btn-lg">
                        <Upload size={20} />
                        Upload Your First Graph
                    </Link>
                </motion.section>
            </div>
        </div>
    );
}
