import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { X, Award, ChevronLeft, ChevronRight, CheckCircle } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { QuizQuestion } from '../components/QuizQuestion';
import { useTextToSpeech } from '../hooks/useTextToSpeech';

export const QuizView = ({
  unit,
  quizAnswers,
  isSubmitted,
  score,
  onSelectAnswer,
  onSubmitQuiz,
  onBackToUnit,
  onGoHome
}) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const allAnswered = Object.keys(quizAnswers).length === unit.quiz.length;
  const percentageScore = Math.round((score / unit.quiz.length) * 100);
  const currentQuestion = unit.quiz[currentQuestionIndex];
  const isLastQuestion = currentQuestionIndex === unit.quiz.length - 1;
  const isFirstQuestion = currentQuestionIndex === 0;

  const { speak } = useTextToSpeech();

  const goToNextQuestion = () => {
    if (currentQuestionIndex < unit.quiz.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const goToPreviousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSelectAnswer = (questionIndex, optionIndex) => {
    onSelectAnswer(questionIndex, optionIndex);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <Button onClick={onBackToUnit} variant="secondary" icon={X} size="sm">
            Back to Unit
          </Button>
          <Badge variant="green">Unit {unit.id} Quiz</Badge>
        </div>

        <Card className="mb-6">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            {unit.title} - Quiz
          </h2>

          {!isSubmitted ? (
            <>
              {/* Progress Indicator */}
              <div className="mb-6 bg-blue-50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-semibold text-gray-700">
                    Question {currentQuestionIndex + 1} of {unit.quiz.length}
                  </p>
                  <p className="text-sm text-gray-600">
                    {Object.keys(quizAnswers).length}/{unit.quiz.length} answered
                  </p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${((currentQuestionIndex + 1) / unit.quiz.length) * 100}%`
                    }}
                  />
                </div>
              </div>

              {/* Question Navigation Dots */}
              <div className="flex flex-wrap gap-2 mb-6 justify-center">
                {unit.quiz.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentQuestionIndex(index)}
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold transition-all ${
                      index === currentQuestionIndex
                        ? 'bg-blue-600 text-white scale-110'
                        : quizAnswers[index] !== undefined
                        ? 'bg-green-500 text-white hover:bg-green-600'
                        : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
                    }`}
                    title={`Question ${index + 1}${quizAnswers[index] !== undefined ? ' (answered)' : ''}`}
                  >
                    {quizAnswers[index] !== undefined ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : (
                      index + 1
                    )}
                  </button>
                ))}
              </div>

              {/* Current Question */}
              <QuizQuestion
                question={currentQuestion}
                questionIndex={currentQuestionIndex}
                selectedAnswer={quizAnswers[currentQuestionIndex]}
                onSelectAnswer={handleSelectAnswer}
                isSubmitted={false}
                onReadQuestion={speak}
              />

              {/* Navigation Buttons */}
              <div className="flex gap-4 mt-8">
                <Button
                  onClick={goToPreviousQuestion}
                  disabled={isFirstQuestion}
                  variant="secondary"
                  size="lg"
                  icon={ChevronLeft}
                  className="flex-1"
                >
                  Previous
                </Button>

                {!isLastQuestion ? (
                  <Button
                    onClick={goToNextQuestion}
                    variant="primary"
                    size="lg"
                    icon={ChevronRight}
                    className="flex-1"
                  >
                    Next
                  </Button>
                ) : (
                  <Button
                    onClick={onSubmitQuiz}
                    disabled={!allAnswered}
                    variant="success"
                    size="lg"
                    className="flex-1"
                  >
                    Submit Quiz
                  </Button>
                )}
              </div>

              {!allAnswered && isLastQuestion && (
                <p className="text-center text-sm text-orange-600 mt-4">
                  Please answer all questions before submitting
                </p>
              )}
            </>
          ) : (
            <>
              {/* Results View */}
              <div className="mb-6 bg-green-50 p-6 rounded-lg border-2 border-green-300">
                <div className="flex items-center gap-3 mb-2">
                  <Award className="w-8 h-8 text-yellow-500" />
                  <h3 className="text-2xl font-bold text-gray-800">Quiz Complete!</h3>
                </div>
                <p className="text-xl text-gray-700 mb-4">
                  Your Score:{' '}
                  <span className="font-bold text-green-600">
                    {score}/{unit.quiz.length}
                  </span>{' '}
                  ({percentageScore}%)
                </p>
                <p className="text-sm text-gray-600">
                  Review your answers below to see what you got right and wrong.
                </p>
              </div>

              {/* All Questions with Results */}
              <div className="space-y-6 mb-8">
                {unit.quiz.map((question, qIndex) => (
                  <QuizQuestion
                    key={qIndex}
                    question={question}
                    questionIndex={qIndex}
                    selectedAnswer={quizAnswers[qIndex]}
                    onSelectAnswer={handleSelectAnswer}
                    isSubmitted={true}
                    onReadQuestion={speak}
                  />
                ))}
              </div>

              <div className="flex gap-4">
                <Button onClick={onBackToUnit} variant="primary" size="lg" className="flex-1">
                  Back to Unit
                </Button>
                <Button onClick={onGoHome} variant="secondary" size="lg" className="flex-1">
                  Home
                </Button>
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
};

QuizView.propTypes = {
  unit: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    quiz: PropTypes.arrayOf(PropTypes.object).isRequired
  }).isRequired,
  quizAnswers: PropTypes.object.isRequired,
  isSubmitted: PropTypes.bool.isRequired,
  score: PropTypes.number.isRequired,
  onSelectAnswer: PropTypes.func.isRequired,
  onSubmitQuiz: PropTypes.func.isRequired,
  onBackToUnit: PropTypes.func.isRequired,
  onGoHome: PropTypes.func.isRequired
};
