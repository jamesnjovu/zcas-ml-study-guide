import { useState, useCallback } from 'react';
import { VIEWS } from '../constants';

export const useNavigation = () => {
  const [currentView, setCurrentView] = useState(VIEWS.HOME);
  const [selectedUnit, setSelectedUnit] = useState(null);

  const goToHome = useCallback(() => {
    setCurrentView(VIEWS.HOME);
    setSelectedUnit(null);
  }, []);

  const goToQuizList = useCallback(() => {
    setCurrentView(VIEWS.QUIZ_LIST);
  }, []);

  const goToUnit = useCallback((unit) => {
    setSelectedUnit(unit);
    setCurrentView(VIEWS.UNIT);
  }, []);

  const goToQuiz = useCallback(() => {
    setCurrentView(VIEWS.QUIZ);
  }, []);

  const goBackToUnit = useCallback(() => {
    setCurrentView(VIEWS.UNIT);
  }, []);

  const goToNextUnit = useCallback((units) => {
    if (!selectedUnit) return;

    const currentIndex = units.findIndex(u => u.id === selectedUnit.id);
    if (currentIndex < units.length - 1) {
      setSelectedUnit(units[currentIndex + 1]);
      setCurrentView(VIEWS.UNIT);
    }
  }, [selectedUnit]);

  return {
    currentView,
    selectedUnit,
    goToHome,
    goToQuizList,
    goToUnit,
    goToQuiz,
    goBackToUnit,
    goToNextUnit
  };
};
