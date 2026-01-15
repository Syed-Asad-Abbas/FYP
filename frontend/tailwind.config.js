/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Deep slate backgrounds
                slate: {
                    950: '#0B0F14',
                    900: '#111820',
                    850: '#161d27',
                    800: '#1a2332',
                },
                // Accent colors
                cyan: {
                    400: '#38BDF8',
                    500: '#0EA5E9',
                },
                emerald: {
                    400: '#4ADE80',
                    500: '#22C55E',
                },
                rose: {
                    400: '#F87171',
                    500: '#EF4444',
                },
                amber: {
                    400: '#FBBF24',
                }
            },
            fontFamily: {
                sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'SF Pro Display', 'Segoe UI', 'sans-serif'],
            },
            fontSize: {
                '2xs': '0.625rem',
            },
            borderRadius: {
                'xl': '12px',
                '2xl': '16px',
                '3xl': '24px',
            },
            boxShadow: {
                'glow': '0 0 40px rgba(56, 189, 248, 0.08)',
                'glow-lg': '0 0 60px rgba(56, 189, 248, 0.12)',
                'card': '0 4px 24px rgba(0, 0, 0, 0.2)',
            },
            backdropBlur: {
                'xs': '2px',
            },
            animation: {
                'shimmer': 'shimmer 2s infinite linear',
                'fade-in': 'fadeIn 0.3s ease-out',
                'slide-up': 'slideUp 0.3s ease-out',
            },
            keyframes: {
                shimmer: {
                    '0%': { backgroundPosition: '-200% 0' },
                    '100%': { backgroundPosition: '200% 0' },
                },
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { opacity: '0', transform: 'translateY(8px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
            },
        },
    },
    plugins: [],
}
