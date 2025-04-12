import { 
  differenceInMinutes, 
  parseISO, 
  isToday, 
  isThisWeek, 
  isThisMonth, 
  format, 
    subDays,
} from "date-fns"

/**
 * Parses a date string from the format used in events
 * @param {string} dateString - Date string in format "YYYY-MM-DD HH:MM" or ISO format
 * @returns {Date} Parsed date object
 */
export const parseEventDate = (dateString) => {
    if (!dateString) return new Date()

    try {
        // Handle ISO format directly
        if (dateString.includes("T") || dateString.includes("Z")) {
            return parseISO(dateString)
        }

        // Handle older format "YYYY-MM-DD HH:MM"
        const formattedString = dateString.replace(" ", "T") + ":00"
        return parseISO(formattedString)
    } catch (error) {
        console.error("Error parsing date:", error)
        return new Date()
    }
}

/**
 * Determines if an event is a login event
 * @param {Object} event - Event object
 * @returns {boolean} True if this is a login event
 */
const isLoginEvent = (event) => {
    if (!event || !event.event_type) {
        console.warn("Invalid event passed to isLoginEvent:", event)
        return false
    }

    const result =
        event.event_type === "Login" ||
        event.event_type === "login" ||
        (event.event_type === "login_attempt" && event.status === "success")

    return result
}

/**
 * Determines if an event is a logout event
 * @param {Object} event - Event object
 * @returns {boolean} True if this is a logout event
 */
const isLogoutEvent = (event) => {
    if (!event || !event.event_type) {
        console.warn("Invalid event passed to isLogoutEvent:", event)
        return false
    }

    return event.event_type === "Logout" || event.event_type === "logout"
}

/**
 * Calculates total session hours from login/logout event pairs
 * @param {Array} events - Array of login/logout events
 * @returns {Object} Session metrics including today, weekly, and monthly hours
 */
export const calculateSessionMetrics = (events) => {
    console.log(
        "Starting session metrics calculation with",
        events?.length || 0,
        "events"
    )

  if (!events || events.length === 0) {
        console.log("No events provided, returning empty metrics")
    return {
      todayHours: 0,
      weeklyHours: 0,
      monthlyHours: 0,
            currentStatus: "Offline",
            sessionHistory: [],
        }
    }

    // Validate input events - ensure they all have required fields
    const validEvents = events.filter((event) => {
        if (!event || !event.time || !event.event_type) {
            console.warn("Invalid event found, skipping:", event)
            return false
        }
        return true
    })

    if (validEvents.length < events.length) {
        console.warn(
            `Filtered out ${events.length - validEvents.length} invalid events`
        )
  }

  // Sort events by time
    console.log("Sorting events by time")
    const sortedEvents = [...validEvents].sort((a, b) => {
        return parseEventDate(a.time) - parseEventDate(b.time)
    })

    // Group events by user_name to handle multiple users on same machine
    // This helps with proper pairing of login/logout events
    const userGroups = {}

    sortedEvents.forEach((event) => {
        const userName = event.user_name || "unknown_user"
        if (!userGroups[userName]) {
            userGroups[userName] = []
        }
        userGroups[userName].push(event)
    })

    console.log(`Found ${Object.keys(userGroups).length} different users`)

    // Process sessions for each user separately
    const allSessions = []
    let totalTodayMinutes = 0
    let totalWeeklyMinutes = 0
    let totalMonthlyMinutes = 0

    Object.entries(userGroups).forEach(([userName, userEvents]) => {
        console.log(
            `Processing events for user: ${userName} (${userEvents.length} events)`
        )

        // Filter out system events for this user
        const filteredUserEvents = userEvents.filter(
            (event) =>
                !userName.includes("SYSTEM") &&
                !userName.includes("LOCAL SERVICE") &&
                !userName.includes("NETWORK SERVICE")
        )

        if (filteredUserEvents.length === 0) {
            console.log(`Skipping system user: ${userName}`)
            return
        }

        // Pair login and logout events for this user
        const userSessions = []
        let currentLogin = null
        let userTodayMinutes = 0
        let userWeeklyMinutes = 0
        let userMonthlyMinutes = 0

        // Process each event for this user
        for (let i = 0; i < filteredUserEvents.length; i++) {
            const event = filteredUserEvents[i]
            const eventDate = parseEventDate(event.time)

            if (isLoginEvent(event)) {
                if (currentLogin) {
                    console.log(
                        `Login event found but already have an active login from ${currentLogin.startTime}. Will close previous login.`
                    )
                    // If we already have an active login, close it implicitly before starting a new one
                    const implicitDuration = differenceInMinutes(
                        eventDate, // Use current event time as implicit logout time
                        currentLogin.start
                    )

                    if (implicitDuration > 0) {
                        const session = {
                            start: currentLogin.startTime,
                            end: event.time,
                            duration: implicitDuration,
                            durationText: formatDuration(implicitDuration),
                            ip: currentLogin.ip,
                            user: userName,
                            implicitLogout: true,
                        }

                        userSessions.push(session)

                        // Update metrics
                        if (isToday(eventDate)) {
                            userTodayMinutes += implicitDuration
                        }

                        if (isThisWeek(eventDate, { weekStartsOn: 1 })) {
                            userWeeklyMinutes += implicitDuration
                        }

                        if (isThisMonth(eventDate)) {
                            userMonthlyMinutes += implicitDuration
                        }
                    }
                }

                console.log(`Login event at ${event.time} for ${userName}`)
      currentLogin = {
        start: eventDate,
        startTime: event.time,
                    ip: event.ip_address,
                }
            } else if (isLogoutEvent(event) && currentLogin) {
                console.log(`Logout event at ${event.time} for ${userName}`)
      // Calculate session duration
                const durationMinutes = differenceInMinutes(
                    eventDate,
                    currentLogin.start
                )
      
      if (durationMinutes > 0) {
                    console.log(`Session duration: ${durationMinutes} minutes`)
        const session = {
          start: currentLogin.startTime,
          end: event.time,
          duration: durationMinutes,
          durationText: formatDuration(durationMinutes),
                        ip: currentLogin.ip || event.ip_address,
                        user: userName,
                    }
        
                    userSessions.push(session)
        
        // Update metrics
        if (isToday(eventDate)) {
                        userTodayMinutes += durationMinutes
        }
        
        if (isThisWeek(eventDate, { weekStartsOn: 1 })) {
                        userWeeklyMinutes += durationMinutes
        }
        
        if (isThisMonth(eventDate)) {
                        userMonthlyMinutes += durationMinutes
        }
                } else {
                    console.warn(
                        `Invalid session duration: ${durationMinutes} minutes. Ignoring session.`
                    )
      }
      
      // Reset current login
                currentLogin = null
            } else {
                console.log(
                    `Skipping event: ${event.event_type} at ${event.time} for ${userName}`
                )
            }
        }
  
  // If user is currently logged in, add the ongoing session
        if (currentLogin) {
            console.log(
                `User ${userName} is currently logged in, adding ongoing session`
            )
            const now = new Date()
            const ongoingDuration = differenceInMinutes(now, currentLogin.start)
            console.log(`Ongoing session duration: ${ongoingDuration} minutes`)
    
    // Add ongoing session to today's, weekly, and monthly totals
            if (isToday(currentLogin.start)) {
                userTodayMinutes += ongoingDuration
    }
    
            if (isThisWeek(currentLogin.start, { weekStartsOn: 1 })) {
                userWeeklyMinutes += ongoingDuration
    }
    
            if (isThisMonth(currentLogin.start)) {
                userMonthlyMinutes += ongoingDuration
    }
    
    // Add ongoing session to sessions array
            userSessions.push({
                start: currentLogin.startTime,
                end: "Current",
      duration: ongoingDuration,
      durationText: formatDuration(ongoingDuration),
                ip: currentLogin.ip,
                user: userName,
                active: true,
            })
        }

        // Add this user's sessions to the complete list
        allSessions.push(...userSessions)

        // Add this user's time to the total metrics
        totalTodayMinutes += userTodayMinutes
        totalWeeklyMinutes += userWeeklyMinutes
        totalMonthlyMinutes += userMonthlyMinutes

        console.log(
            `Completed processing for user ${userName}. Found ${userSessions.length} sessions.`
        )
    })

    // Sort all sessions by start time (newest first)
    allSessions.sort(
        (a, b) => parseEventDate(b.start) - parseEventDate(a.start)
    )

    // Special handling for unknown session patterns
    if (allSessions.length === 0 && sortedEvents.length > 0) {
        console.log(
            "No sessions could be created from events, creating fallback sessions"
        )

        // Check if we only have logout events for this user (a common edge case)
        const onlyLogoutEvents = sortedEvents.every((event) =>
            isLogoutEvent(event)
        )

        if (onlyLogoutEvents) {
            console.log(
                "Special case: User has only logout events and no login events"
            )

            // Create a single session for each logout event, assuming they were logged in
            // for a reasonable time (1 hour) before each logout
            sortedEvents.forEach((event, index) => {
                const eventDate = parseEventDate(event.time)
                // Create an artificial login time 1 hour before the logout
                const artificialLoginTime = new Date(eventDate)
                artificialLoginTime.setHours(artificialLoginTime.getHours() - 1)

                const session = {
                    start: artificialLoginTime.toISOString(),
                    end: event.time,
                    duration: 60, // 60 minutes (1 hour)
                    durationText: "1h",
                    ip: event.ip_address,
                    user: event.user_name || "Unknown User",
                    artificialLogin: true, // Mark this as having an artificial login time
                    fallbackSession: true,
                }

                allSessions.push(session)

                // Update metrics
                if (isToday(eventDate)) {
                    totalTodayMinutes += 60
                }
                if (isThisWeek(eventDate, { weekStartsOn: 1 })) {
                    totalWeeklyMinutes += 60
                }
                if (isThisMonth(eventDate)) {
                    totalMonthlyMinutes += 60
                }
            })
        } else {
            // If we couldn't pair login/logout events but have events, create sessions based on
            // individual events as a fallback
            let lastEventDate = null
            let lastEventTime = null

            sortedEvents.forEach((event, index) => {
                const eventDate = parseEventDate(event.time)

                // If this is a login or logout, create a session
                if (isLoginEvent(event) || isLogoutEvent(event)) {
                    if (lastEventDate) {
                        // Calculate time since last event
                        const durationMinutes = differenceInMinutes(
                            eventDate,
                            lastEventDate
                        )

                        if (durationMinutes > 0) {
                            const session = {
                                start: lastEventTime,
                                end: event.time,
                                duration: durationMinutes,
                                durationText: formatDuration(durationMinutes),
                                ip: event.ip_address,
                                user: event.user_name || "Unknown User",
                                fallbackSession: true,
                            }

                            allSessions.push(session)

                            // Update metrics
                            if (isToday(eventDate)) {
                                totalTodayMinutes += durationMinutes
                            }
                            if (isThisWeek(eventDate, { weekStartsOn: 1 })) {
                                totalWeeklyMinutes += durationMinutes
                            }
                            if (isThisMonth(eventDate)) {
                                totalMonthlyMinutes += durationMinutes
                            }
                        }
                    }

                    lastEventDate = eventDate
                    lastEventTime = event.time
                }
            })

            // Add one final session from the last event to now if needed
            if (lastEventDate) {
                const now = new Date()
                const durationMinutes = differenceInMinutes(now, lastEventDate)

                if (durationMinutes > 0) {
                    const lastEvent = sortedEvents[sortedEvents.length - 1]

                    allSessions.push({
                        start: lastEventTime,
                        end: "Current",
                        duration: durationMinutes,
                        durationText: formatDuration(durationMinutes),
                        ip: lastEvent.ip_address,
                        user: lastEvent.user_name || "Unknown User",
                        active: true,
                        fallbackSession: true,
                    })

                    // Update metrics
                    if (isToday(lastEventDate)) {
                        totalTodayMinutes += durationMinutes
                    }
                    if (isThisWeek(lastEventDate, { weekStartsOn: 1 })) {
                        totalWeeklyMinutes += durationMinutes
                    }
                    if (isThisMonth(lastEventDate)) {
                        totalMonthlyMinutes += durationMinutes
                    }
                }
            }
        }

        // Sort fallback sessions as well
        allSessions.sort(
            (a, b) => parseEventDate(b.start) - parseEventDate(a.start)
        )
  }
  
  // Generate daily sessions for the past 7 days
    const dailySessions = generateDailySessions(allSessions)

    console.log(
        `Calculation complete. Found ${allSessions.length} total sessions.`
    )
  
  return {
        todayHours: parseFloat((totalTodayMinutes / 60).toFixed(1)),
        weeklyHours: parseFloat((totalWeeklyMinutes / 60).toFixed(1)),
        monthlyHours: parseFloat((totalMonthlyMinutes / 60).toFixed(1)),
        currentStatus: getCurrentUserStatus(sortedEvents),
        sessionHistory: allSessions, // Most recent first
        dailySessions,
    }
}

/**
 * Determine if user is currently online based on last event
 * @param {Array} events - Sorted events array
 * @returns {string} "Online" or "Offline"
 */
const getCurrentUserStatus = (events) => {
    if (!events || events.length === 0) return "Offline"

    // Get the most recent event
    const lastEvent = events[events.length - 1]

    const isOnline =
        isLoginEvent(lastEvent) &&
        !events
            .slice()
            .reverse()
            .some(
                (e) =>
                    e.user_name === lastEvent.user_name &&
                    isLogoutEvent(e) &&
                    parseEventDate(e.time) > parseEventDate(lastEvent.time)
            )

    return isOnline ? "Online" : "Offline"
}

/**
 * Generate daily session data for the past 7 days
 * @param {Array} sessions - Array of session objects
 * @returns {Array} Daily session data for the past 7 days
 */
const generateDailySessions = (sessions) => {
    const days = []
    const today = new Date()
  
  // Create entries for the past 7 days
  for (let i = 6; i >= 0; i--) {
        const day = subDays(today, i)
        const dayFormat = format(day, "yyyy-MM-dd")
        const displayFormat = format(day, "EEE")
    
    days.push({
      date: dayFormat,
      day: displayFormat,
            hours: 0,
        })
  }
  
  // Calculate hours for each day
    sessions.forEach((session) => {
        const startDate = parseEventDate(session.start)
        const startDay = format(startDate, "yyyy-MM-dd")

        const dayEntry = days.find((d) => d.date === startDay)
    if (dayEntry) {
            dayEntry.hours += session.duration / 60
    }
    })
  
  // Round hours to 1 decimal place
    days.forEach((day) => {
        day.hours = parseFloat(day.hours.toFixed(1))
    })

    return days
}

/**
 * Format minutes duration into a readable string (e.g., "2h 30m")
 * @param {number} minutes - Duration in minutes
 * @returns {string} Formatted duration string
 */
const formatDuration = (minutes) => {
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = Math.round(minutes % 60)
  
  if (hours === 0) {
        return `${remainingMinutes}m`
  } else if (remainingMinutes === 0) {
        return `${hours}h`
  } else {
        return `${hours}h ${remainingMinutes}m`
    }
  }
