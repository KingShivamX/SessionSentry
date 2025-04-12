import {
    BrowserRouter as Router,
    Routes,
    Route,
    Navigate,
} from "react-router-dom"
import Sidebar from "./components/Sidebar"
import Header from "./components/Header"
import Users from "./components/Users"
import UserDashboard from "./components/UserDashboard"

function App() {
    return (
        <Router>
            <div className="flex h-screen bg-gray-50">
                <Sidebar />

                <div className="flex flex-col flex-1 overflow-hidden">
                    <Header username="SessionSentry Admin" status="Online" />

                    <main className="flex-1 overflow-y-auto">
                        <Routes>
                            <Route path="/" element={<Users />} />
                            <Route
                                path="/user/:username"
                                element={<UserDashboard />}
                            />
                            <Route
                                path="*"
                                element={<Navigate to="/" replace />}
                            />
                        </Routes>
                    </main>
                </div>
            </div>
        </Router>
    )
}

export default App
