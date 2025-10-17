import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { Volume2, ChevronDown, ChevronUp } from 'lucide-react';
import { renderMarkdown } from '../utils/markdownRenderer';
import { stripMarkdown } from '../utils/stripMarkdown';

export const PastPaperQuestion = ({ question, onSpeak }) => {
  const [showAnswer, setShowAnswer] = useState(false);

  const handleSpeak = () => {
    if (onSpeak) {
      let textToSpeak = `Question: ${stripMarkdown(question.question)}`;
      if (question.sampleAnswer) {
        textToSpeak += `. Sample Answer: ${stripMarkdown(question.sampleAnswer)}`;
      }
      onSpeak(textToSpeak);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-4 border border-gray-200">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
              {question.marks} marks
            </span>
          </div>
          <div className="text-lg text-gray-800 font-medium leading-relaxed">
            {renderMarkdown(question.question)}
          </div>
        </div>
        {onSpeak && (
          <button
            onClick={handleSpeak}
            className="flex-shrink-0 p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
            title="Read question aloud"
          >
            <Volume2 className="w-5 h-5" />
          </button>
        )}
      </div>

      {question.sampleAnswer && (
        <>
          <button
            onClick={() => setShowAnswer(!showAnswer)}
            className="flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium transition-colors"
          >
            {showAnswer ? (
              <>
                <ChevronUp className="w-5 h-5" />
                Hide Sample Answer
              </>
            ) : (
              <>
                <ChevronDown className="w-5 h-5" />
                Show Sample Answer
              </>
            )}
          </button>

          {showAnswer && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">Sample Answer:</h4>
              <div className="text-gray-700 leading-relaxed">
                {renderMarkdown(question.sampleAnswer)}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

PastPaperQuestion.propTypes = {
  question: PropTypes.shape({
    id: PropTypes.string.isRequired,
    question: PropTypes.string.isRequired,
    sampleAnswer: PropTypes.string.isRequired,
    marks: PropTypes.number.isRequired
  }).isRequired,
  onSpeak: PropTypes.func
};
