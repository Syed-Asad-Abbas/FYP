import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useAuth } from './hooks/useAuth'

// Layout
import Navbar from './components/Navbar'
import ProtectedRoute from './components/ProtectedRoute'

// Pages
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import DashboardPage from './pages/DashboardPage'
import ScanPage from './pages/ScanPage'
import ResultPage from './pages/ResultPage'
import HistoryPage from './pages/HistoryPage'
import AdminDashboardPage from './pages/AdminDashboardPage'

function App() {
    const { user, loading } = useAuth()

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
            </div>
        )
    }

    return (
        <div className="min-h-screen flex flex-col">
            {/* Skip link for accessibility */}
            <a href="#main-content" className="skip-link">
                Skip to main content
            </a>

            {user && <Navbar />}

            <main id="main-content" className="flex-1">
                <AnimatePresence mode="wait">
                    <Routes>
                        {/* Public routes */}
                        <Route
                            path="/login"
                            element={user ? <Navigate to="/dashboard" replace /> : <LoginPage />}
                        />
                        <Route
                            path="/signup"
                            element={user ? <Navigate to="/dashboard" replace /> : <SignupPage />}
                        />

                        {/* Protected routes */}
                        <Route element={<ProtectedRoute />}>
                            <Route path="/dashboard" element={<DashboardPage />} />
                            <Route path="/scan" element={<ScanPage />} />
                            <Route path="/result/:id" element={<ResultPage />} />
                            <Route path="/history" element={<HistoryPage />} />
                        </Route>

                        {/* Admin routes */}
                        <Route element={<ProtectedRoute requiredRole="admin" />}>
                            <Route path="/admin" element={<AdminDashboardPage />} />
                        </Route>

                        {/* Default redirect */}
                        <Route
                            path="*"
                            element={<Navigate to={user ? "/dashboard" : "/login"} replace />}
                        />
                    </Routes>
                </AnimatePresence>
            </main>
        </div>
    )
}

export default App
