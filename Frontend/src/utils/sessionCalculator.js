import { 
  differenceInMinutes, 
  parseISO, 
  isToday, 
  isThisWeek, 
  isThisMonth, 
  format, 
  subDays 
} from 'date-fns';

/**
 * Parses a date string from the format used in events (YYYY-MM-DD HH:MM)
 * @param {string} dateString - Date string in format "YYYY-MM-DD HH:MM" 
 * @returns {Date} Parsed date object
 */
export const parseEventDate = (dateString) => {
  const formattedString = dateString.replace(' ', 'T') + ':00';
  return parseISO(formattedString);
};

/**
 * Calculates total session hours from login/logout event pairs
 * @param {Array} events - Array of login/logout events
 * @returns {Object} Session metrics including today, weekly, and monthly hours
 */
export const calculateSessionMetrics = (events) => {
  if (!events || events.length === 0) {
    return {
      todayHours: 0,
      weeklyHours: 0,
      monthlyHours: 0,
      currentStatus: 'Offline',
      sessionHistory: []
    };
  }

  // Sort events by time
  const sortedEvents = [...events].sort((a, b) => {
    return parseEventDate(a.time) - parseEventDate(b.time);
  });

  // Pair login and logout events to calculate session durations
  const sessions = [];
  let currentLogin = null;
  let todayMinutes = 0;
  let weeklyMinutes = 0;
  let monthlyMinutes = 0;
  
  // Process each event
  for (let i = 0; i < sortedEvents.length; i++) {
    const event = sortedEvents[i];
    const eventDate = parseEventDate(event.time);
    
    if (event.event_type === 'Login') {
      currentLogin = {
        start: eventDate,
        startTime: event.time,
        ip: event.ip_address
      };
    } else if (event.event_type === 'Logout' && currentLogin) {
      // Calculate session duration
      const durationMinutes = differenceInMinutes(eventDate, currentLogin.start);
      
      if (durationMinutes > 0) {
        const session = {
          start: currentLogin.startTime,
          end: event.time,
          duration: durationMinutes,
          durationText: formatDuration(durationMinutes),
          ip: currentLogin.ip || event.ip_address
        };
        
        sessions.push(session);
        
        // Update metrics
        if (isToday(eventDate)) {
          todayMinutes += durationMinutes;
        }
        
        if (isThisWeek(eventDate, { weekStartsOn: 1 })) {
          weeklyMinutes += durationMinutes;
        }
        
        if (isThisMonth(eventDate)) {
          monthlyMinutes += durationMinutes;
        }
      }
      
      // Reset current login
      currentLogin = null;
    }
  }
  
  // Check if the user is currently logged in
  const currentStatus = currentLogin ? 'Online' : 'Offline';
  
  // If user is currently logged in, add the ongoing session
  if (currentLogin) {
    const now = new Date();
    const ongoingDuration = differenceInMinutes(now, currentLogin.start);
    
    // Add ongoing session to today's, weekly, and monthly totals
    if (isToday(currentLogin.start)) {
      todayMinutes += ongoingDuration;
    }
    
    if (isThisWeek(currentLogin.start, { weekStartsOn: 1 })) {
      weeklyMinutes += ongoingDuration;
    }
    
    if (isThisMonth(currentLogin.start)) {
      monthlyMinutes += ongoingDuration;
    }
    
    // Add ongoing session to the sessions array
    sessions.push({
      start: currentLogin.startTime,
      end: 'Current Session',
      duration: ongoingDuration,
      durationText: formatDuration(ongoingDuration),
      ip: currentLogin.ip,
      ongoing: true
    });
  }
  
  // Generate daily sessions for the past 7 days
  const dailySessions = generateDailySessions(sessions);
  
  return {
    todayHours: parseFloat((todayMinutes / 60).toFixed(1)),
    weeklyHours: parseFloat((weeklyMinutes / 60).toFixed(1)),
    monthlyHours: parseFloat((monthlyMinutes / 60).toFixed(1)),
    currentStatus,
    sessionHistory: sessions.reverse(), // Most recent first
    dailySessions
  };
};

/**
 * Generate daily session data for the past 7 days
 * @param {Array} sessions - Array of session objects
 * @returns {Array} Daily session data for the past 7 days
 */
const generateDailySessions = (sessions) => {
  const days = [];
  const today = new Date();
  
  // Create entries for the past 7 days
  for (let i = 6; i >= 0; i--) {
    const day = subDays(today, i);
    const dayFormat = format(day, 'yyyy-MM-dd');
    const displayFormat = format(day, 'EEE');
    
    days.push({
      date: dayFormat,
      day: displayFormat,
      hours: 0
    });
  }
  
  // Calculate hours for each day
  sessions.forEach(session => {
    const startDate = parseEventDate(session.start);
    const startDay = format(startDate, 'yyyy-MM-dd');
    
    const dayEntry = days.find(d => d.date === startDay);
    if (dayEntry) {
      dayEntry.hours += session.duration / 60;
    }
  });
  
  // Round hours to 1 decimal place
  days.forEach(day => {
    day.hours = parseFloat(day.hours.toFixed(1));
  });
  
  return days;
};

/**
 * Format minutes duration into a readable string (e.g., "2h 30m")
 * @param {number} minutes - Duration in minutes
 * @returns {string} Formatted duration string
 */
const formatDuration = (minutes) => {
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = Math.round(minutes % 60);
  
  if (hours === 0) {
    return `${remainingMinutes}m`;
  } else if (remainingMinutes === 0) {
    return `${hours}h`;
  } else {
    return `${hours}h ${remainingMinutes}m`;
  }
};
