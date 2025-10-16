const STORAGE_KEYS = {
  QUIZ_SCORES: 'ml_study_guide_quiz_scores',
  QUIZ_ANSWERS: 'ml_study_guide_quiz_answers',
  COMPLETED_UNITS: 'ml_study_guide_completed_units',
  LAST_VISITED_UNIT: 'ml_study_guide_last_unit',
  USER_PREFERENCES: 'ml_study_guide_preferences'
};

// Check if localStorage is available
const isLocalStorageAvailable = () => {
  try {
    const test = '__localStorage_test__';
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch (e) {
    return false;
  }
};

// Generic get from localStorage
export const getFromStorage = (key, defaultValue = null) => {
  if (!isLocalStorageAvailable()) return defaultValue;

  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error(`Error reading from localStorage (${key}):`, error);
    return defaultValue;
  }
};

// Generic set to localStorage
export const setToStorage = (key, value) => {
  if (!isLocalStorageAvailable()) return false;

  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Error writing to localStorage (${key}):`, error);
    return false;
  }
};

// Remove from localStorage
export const removeFromStorage = (key) => {
  if (!isLocalStorageAvailable()) return false;

  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error(`Error removing from localStorage (${key}):`, error);
    return false;
  }
};

// Clear all app data
export const clearAllStorage = () => {
  if (!isLocalStorageAvailable()) return false;

  try {
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
    return true;
  } catch (error) {
    console.error('Error clearing localStorage:', error);
    return false;
  }
};

// Quiz Scores
export const saveQuizScores = (scores) => {
  return setToStorage(STORAGE_KEYS.QUIZ_SCORES, scores);
};

export const loadQuizScores = () => {
  return getFromStorage(STORAGE_KEYS.QUIZ_SCORES, {});
};

// Quiz Answers (for current session)
export const saveQuizAnswers = (unitId, answers) => {
  const allAnswers = getFromStorage(STORAGE_KEYS.QUIZ_ANSWERS, {});
  allAnswers[unitId] = answers;
  return setToStorage(STORAGE_KEYS.QUIZ_ANSWERS, allAnswers);
};

export const loadQuizAnswers = (unitId) => {
  const allAnswers = getFromStorage(STORAGE_KEYS.QUIZ_ANSWERS, {});
  return allAnswers[unitId] || {};
};

export const clearQuizAnswers = (unitId) => {
  const allAnswers = getFromStorage(STORAGE_KEYS.QUIZ_ANSWERS, {});
  delete allAnswers[unitId];
  return setToStorage(STORAGE_KEYS.QUIZ_ANSWERS, allAnswers);
};

// Completed Units
export const saveCompletedUnits = (completedUnits) => {
  return setToStorage(STORAGE_KEYS.COMPLETED_UNITS, Array.from(completedUnits));
};

export const loadCompletedUnits = () => {
  const completed = getFromStorage(STORAGE_KEYS.COMPLETED_UNITS, []);
  return new Set(completed);
};

export const markUnitAsCompleted = (unitId) => {
  const completed = loadCompletedUnits();
  completed.add(unitId);
  return saveCompletedUnits(completed);
};

// Last Visited Unit
export const saveLastVisitedUnit = (unitId) => {
  return setToStorage(STORAGE_KEYS.LAST_VISITED_UNIT, unitId);
};

export const loadLastVisitedUnit = () => {
  return getFromStorage(STORAGE_KEYS.LAST_VISITED_UNIT, null);
};

// User Preferences
export const saveUserPreferences = (preferences) => {
  return setToStorage(STORAGE_KEYS.USER_PREFERENCES, preferences);
};

export const loadUserPreferences = () => {
  return getFromStorage(STORAGE_KEYS.USER_PREFERENCES, {
    speechRate: 1.0,
    speechPitch: 1.0,
    speechVolume: 1.0,
    selectedVoiceIndex: null
  });
};

// Export storage keys for reference
export { STORAGE_KEYS };
