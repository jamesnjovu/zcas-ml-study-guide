import React from 'react';
import PropTypes from 'prop-types';
import { CheckCircle, X, Volume2 } from 'lucide-react';
import { renderMarkdown } from '../utils/markdownRenderer';
import { stripMarkdown } from '../utils/stripMarkdown';

export const QuizQuestion = ({
  question,
  questionIndex,
  selectedAnswer,
  onSelectAnswer,
  isSubmitted,
  onReadQuestion
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

  const readQuestion = () => {
    const cleanQuestion = stripMarkdown(question.question);
    const optionsText = question.options
      .map((opt, idx) => `Option ${idx + 1}: ${stripMarkdown(opt)}`)
      .join('. ');
    const fullText = `Question ${questionIndex + 1}: ${cleanQuestion}. ${optionsText}`;
    onReadQuestion(fullText);
  };

  return (
    <div className="border-2 border-gray-200 rounded-lg p-6">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="font-semibold text-gray-800 flex-1">
          <span>{questionIndex + 1}. </span>
          {renderMarkdown(question.question)}
        </div>
        {onReadQuestion && (
          <button
            onClick={readQuestion}
            className="flex-shrink-0 p-2 rounded-lg bg-blue-100 hover:bg-blue-200 text-blue-700 transition-colors"
            title="Read question aloud"
          >
            <Volume2 className="w-5 h-5" />
          </button>
        )}
      </div>
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
              <div className="flex-1 leading-relaxed">{renderMarkdown(option)}</div>
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
  isSubmitted: PropTypes.bool.isRequired,
  onReadQuestion: PropTypes.func
};
