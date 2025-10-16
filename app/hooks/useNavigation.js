import { useState, useCallback } from 'react';
import { VIEWS } from '../constants';

export const useNavigation = () => {
  const [currentView, setCurrentView] = useState(VIEWS.HOME);
  const [selectedUnit, setSelectedUnit] = useState(null);
  const [selectedPastPaper, setSelectedPastPaper] = useState(null);

  const goToHome = useCallback(() => {
    setCurrentView(VIEWS.HOME);
    setSelectedUnit(null);
    setSelectedPastPaper(null);
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

  const goToPastPapers = useCallback(() => {
    setCurrentView(VIEWS.PAST_PAPERS);
    setSelectedPastPaper(null);
  }, []);

  const goToPastPaperDetail = useCallback((paper) => {
    setSelectedPastPaper(paper);
    setCurrentView(VIEWS.PAST_PAPER_DETAIL);
  }, []);

  const goBackToPastPapers = useCallback(() => {
    setCurrentView(VIEWS.PAST_PAPERS);
  }, []);

  return {
    currentView,
    selectedUnit,
    selectedPastPaper,
    goToHome,
    goToQuizList,
    goToUnit,
    goToQuiz,
    goBackToUnit,
    goToNextUnit,
    goToPastPapers,
    goToPastPaperDetail,
    goBackToPastPapers
  };
};
