import { createContext, useState } from "react"
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
import Sessions from "./components/Sessions"
import Alert from "./components/Alert"

// Create context for sharing user data
export const UserContext = createContext({
    activeUser: null,
    activeUserStatus: "Offline",
    setActiveUser: () => {},
    setActiveUserStatus: () => {},
})

function App() {
    const [activeUser, setActiveUser] = useState("SessionSentry Admin")
    const [activeUserStatus, setActiveUserStatus] = useState("Online")

    const userContextValue = {
        activeUser,
        activeUserStatus,
        setActiveUser,
        setActiveUserStatus,
    }

    return (
        <UserContext.Provider value={userContextValue}>
            <Router>
                <div className="flex h-screen bg-gray-50">
                    <Sidebar />

                    <div className="flex flex-col flex-1 overflow-hidden">
                        <Header
                            username={activeUser}
                            status={activeUserStatus}
                        />

                        <main className="flex-1 overflow-y-auto">
                            <Routes>
                                <Route path="/" element={<Users />} />
                                <Route
                                    path="/user/:computer_name"
                                    element={<UserDashboard />}
                                />
                                <Route
                                    path="/sessions"
                                    element={<Sessions />}
                                />
                                <Route path="/alerts" element={<Alert />} />
                                <Route
                                    path="*"
                                    element={<Navigate to="/" replace />}
                                />
                            </Routes>
                        </main>
                    </div>
                </div>
            </Router>
        </UserContext.Provider>
    )
}

export default App
