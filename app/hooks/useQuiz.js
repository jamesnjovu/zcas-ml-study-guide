import { useState, useCallback } from 'react';

export const useQuiz = () => {
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [quizScores, setQuizScores] = useState({});

  const selectAnswer = useCallback((questionIndex, optionIndex) => {
    if (!quizSubmitted) {
      setQuizAnswers(prev => ({ ...prev, [questionIndex]: optionIndex }));
    }
  }, [quizSubmitted]);

  const submitQuiz = useCallback((unit) => {
    setQuizSubmitted(true);
    const correct = unit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
    const newScores = {
      ...quizScores,
      [unit.id]: {
        score: correct,
        total: unit.quiz.length
      }
    };
    setQuizScores(newScores);
  }, [quizAnswers, quizScores]);

  const resetQuiz = useCallback(() => {
    setQuizAnswers({});
    setQuizSubmitted(false);
  }, []);

  const calculateScore = useCallback((unit) => {
    if (!quizSubmitted) return 0;
    return unit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
  }, [quizAnswers, quizSubmitted]);

  return {
    quizAnswers,
    quizSubmitted,
    quizScores,
    selectAnswer,
    submitQuiz,
    resetQuiz,
    calculateScore
  };
};
