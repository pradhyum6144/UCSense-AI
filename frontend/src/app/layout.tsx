import type { Metadata } from 'next';
import '@/styles/globals.css';

export const metadata: Metadata = {
    title: 'UCSense-AI | Graph-to-Data Pipeline',
    description: 'AI-powered extraction of stress-strain data from UCS test graphs with 95%+ accuracy',
    keywords: ['UCS', 'geotechnical', 'stress-strain', 'digitization', 'AI', 'machine learning'],
    authors: [{ name: 'UCSense-AI Team' }],
    openGraph: {
        title: 'UCSense-AI | Graph-to-Data Pipeline',
        description: 'AI-powered extraction of stress-strain data from UCS test graphs',
        type: 'website',
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
                    rel="stylesheet"
                />
            </head>
            <body>
                <main>{children}</main>
            </body>
        </html>
    );
}
