import { useState, useCallback, useEffect } from 'react';
import {
  saveQuizScores,
  loadQuizScores,
  saveQuizAnswers,
  loadQuizAnswers,
  clearQuizAnswers
} from '../utils/localStorage';

export const useQuiz = () => {
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [quizScores, setQuizScores] = useState({});
  const [currentUnitId, setCurrentUnitId] = useState(null);

  // Load quiz scores from localStorage on mount
  useEffect(() => {
    const savedScores = loadQuizScores();
    if (Object.keys(savedScores).length > 0) {
      setQuizScores(savedScores);
    }
  }, []);

  const selectAnswer = useCallback((questionIndex, optionIndex) => {
    if (!quizSubmitted) {
      setQuizAnswers(prev => {
        const updated = { ...prev, [questionIndex]: optionIndex };
        // Save answers to localStorage for current unit
        if (currentUnitId) {
          saveQuizAnswers(currentUnitId, updated);
        }
        return updated;
      });
    }
  }, [quizSubmitted, currentUnitId]);

  const submitQuiz = useCallback((unit) => {
    setQuizSubmitted(true);
    const correct = unit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
    const newScores = {
      ...quizScores,
      [unit.id]: {
        score: correct,
        total: unit.quiz.length,
        date: new Date().toISOString(),
        percentage: Math.round((correct / unit.quiz.length) * 100)
      }
    };
    setQuizScores(newScores);
    // Save scores to localStorage
    saveQuizScores(newScores);
    // Clear saved answers after submission
    clearQuizAnswers(unit.id);
  }, [quizAnswers, quizScores]);

  const resetQuiz = useCallback((unitId = null) => {
    setQuizAnswers({});
    setQuizSubmitted(false);

    if (unitId) {
      setCurrentUnitId(unitId);
      // Load any saved answers for this unit
      const savedAnswers = loadQuizAnswers(unitId);
      if (Object.keys(savedAnswers).length > 0) {
        setQuizAnswers(savedAnswers);
      }
    }
  }, []);

  const calculateScore = useCallback((unit) => {
    if (!quizSubmitted) return 0;
    return unit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
  }, [quizAnswers, quizSubmitted]);

  const clearAllData = useCallback(() => {
    setQuizAnswers({});
    setQuizSubmitted(false);
    setQuizScores({});
    setCurrentUnitId(null);
    saveQuizScores({});
  }, []);

  return {
    quizAnswers,
    quizSubmitted,
    quizScores,
    selectAnswer,
    submitQuiz,
    resetQuiz,
    calculateScore,
    clearAllData
  };
};
