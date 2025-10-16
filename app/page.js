'use client'

import React from 'react';
import { units } from './data/units';
import { VIEWS } from './constants';
import { useNavigation } from './hooks/useNavigation';
import { useQuiz } from './hooks/useQuiz';
import { HomeView } from './views/HomeView';
import { QuizListView } from './views/QuizListView';
import { UnitView } from './views/UnitView';
import { QuizView } from './views/QuizView';

const MLStudyApp = () => {
  const {
    currentView,
    selectedUnit,
    goToHome,
    goToQuizList,
    goToUnit,
    goToQuiz,
    goBackToUnit,
    goToNextUnit
  } = useNavigation();

  const {
    quizAnswers,
    quizSubmitted,
    quizScores,
    selectAnswer,
    submitQuiz,
    resetQuiz,
    calculateScore
  } = useQuiz();

  const handleSelectUnit = (unit) => {
    goToUnit(unit);
    resetQuiz();
  };

  const handleStartQuiz = () => {
    resetQuiz();
    goToQuiz();
  };

  const handleNextUnit = () => {
    goToNextUnit(units);
    resetQuiz();
  };

  const handleSubmitQuiz = () => {
    submitQuiz(selectedUnit);
  };

  // Render appropriate view based on current state
  switch (currentView) {
    case VIEWS.HOME:
      return (
        <HomeView
          units={units}
          quizScores={quizScores}
          onSelectUnit={handleSelectUnit}
          onViewQuizList={goToQuizList}
        />
      );

    case VIEWS.QUIZ_LIST:
      return (
        <QuizListView
          units={units}
          quizScores={quizScores}
          onSelectUnit={handleSelectUnit}
          onGoHome={goToHome}
        />
      );

    case VIEWS.UNIT:
      return selectedUnit ? (
        <UnitView
          unit={selectedUnit}
          units={units}
          onStartQuiz={handleStartQuiz}
          onNextUnit={handleNextUnit}
          onGoHome={goToHome}
        />
      ) : null;

    case VIEWS.QUIZ:
      return selectedUnit ? (
        <QuizView
          unit={selectedUnit}
          quizAnswers={quizAnswers}
          isSubmitted={quizSubmitted}
          score={calculateScore(selectedUnit)}
          onSelectAnswer={selectAnswer}
          onSubmitQuiz={handleSubmitQuiz}
          onBackToUnit={goBackToUnit}
          onGoHome={goToHome}
        />
      ) : null;

    default:
      return null;
  }
};

export default MLStudyApp;