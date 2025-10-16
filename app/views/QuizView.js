import React from 'react';
import PropTypes from 'prop-types';
import { X, Award } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { QuizQuestion } from '../components/QuizQuestion';

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
  const allAnswered = Object.keys(quizAnswers).length === unit.quiz.length;
  const percentageScore = Math.round((score / unit.quiz.length) * 100);

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

          {!isSubmitted && (
            <div className="mb-6 bg-blue-50 p-4 rounded-lg">
              <p className="text-sm text-gray-700">
                Progress: {Object.keys(quizAnswers).length}/{unit.quiz.length} questions
                answered
              </p>
            </div>
          )}

          {isSubmitted && (
            <div className="mb-6 bg-green-50 p-6 rounded-lg border-2 border-green-300">
              <div className="flex items-center gap-3 mb-2">
                <Award className="w-8 h-8 text-yellow-500" />
                <h3 className="text-2xl font-bold text-gray-800">Quiz Complete!</h3>
              </div>
              <p className="text-xl text-gray-700">
                Your Score:{' '}
                <span className="font-bold text-green-600">
                  {score}/{unit.quiz.length}
                </span>{' '}
                ({percentageScore}%)
              </p>
            </div>
          )}

          <div className="space-y-6">
            {unit.quiz.map((question, qIndex) => (
              <QuizQuestion
                key={qIndex}
                question={question}
                questionIndex={qIndex}
                selectedAnswer={quizAnswers[qIndex]}
                onSelectAnswer={onSelectAnswer}
                isSubmitted={isSubmitted}
              />
            ))}
          </div>

          {!isSubmitted && (
            <Button
              onClick={onSubmitQuiz}
              disabled={!allAnswered}
              variant="success"
              size="lg"
              className="w-full mt-8"
            >
              Submit Quiz
            </Button>
          )}

          {isSubmitted && (
            <div className="flex gap-4 mt-8">
              <Button onClick={onBackToUnit} variant="primary" size="lg" className="flex-1">
                Back to Unit
              </Button>
              <Button onClick={onGoHome} variant="secondary" size="lg" className="flex-1">
                Home
              </Button>
            </div>
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
