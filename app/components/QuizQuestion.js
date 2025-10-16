import React from 'react';
import PropTypes from 'prop-types';
import { CheckCircle, X } from 'lucide-react';

export const QuizQuestion = ({
  question,
  questionIndex,
  selectedAnswer,
  onSelectAnswer,
  isSubmitted
}) => {
  const getOptionStyles = (optionIndex) => {
    const isSelected = selectedAnswer === optionIndex;
    const isCorrect = question.correct === optionIndex;
    const showResult = isSubmitted;

    let styles = 'bg-white hover:bg-gray-50 border-gray-300 text-gray-900';

    if (showResult) {
      if (isCorrect) {
        styles = 'bg-green-50 border-green-600 border-2 text-gray-900 font-medium';
      } else if (isSelected && !isCorrect) {
        styles = 'bg-red-50 border-red-600 border-2 text-gray-900 font-medium';
      } else {
        styles = 'bg-white border-gray-300 text-gray-700';
      }
    } else if (isSelected) {
      styles = 'bg-blue-50 border-blue-600 border-2 text-gray-900 font-medium';
    }

    return styles;
  };

  const showIcon = (optionIndex) => {
    if (!isSubmitted) return null;

    const isCorrect = question.correct === optionIndex;
    const isSelected = selectedAnswer === optionIndex;

    if (isCorrect) {
      return <CheckCircle className="w-5 h-5 text-green-600" />;
    }
    if (isSelected && !isCorrect) {
      return <X className="w-5 h-5 text-red-600" />;
    }
    return null;
  };

  return (
    <div className="border-2 border-gray-200 rounded-lg p-6">
      <h4 className="font-semibold text-gray-800 mb-4">
        {questionIndex + 1}. {question.question}
      </h4>
      <div className="space-y-2">
        {question.options.map((option, optionIndex) => (
          <button
            key={optionIndex}
            onClick={() => onSelectAnswer(questionIndex, optionIndex)}
            disabled={isSubmitted}
            className={`w-full text-left p-4 rounded-lg border transition-all ${getOptionStyles(
              optionIndex
            )} ${isSubmitted ? 'cursor-default' : 'cursor-pointer hover:shadow-none'} ${!isSubmitted && 'hover:shadow-md'}`}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="flex-1 leading-relaxed">{option}</span>
              {showIcon(optionIndex)}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

QuizQuestion.propTypes = {
  question: PropTypes.shape({
    question: PropTypes.string.isRequired,
    options: PropTypes.arrayOf(PropTypes.string).isRequired,
    correct: PropTypes.number.isRequired
  }).isRequired,
  questionIndex: PropTypes.number.isRequired,
  selectedAnswer: PropTypes.number,
  onSelectAnswer: PropTypes.func.isRequired,
  isSubmitted: PropTypes.bool.isRequired
};
