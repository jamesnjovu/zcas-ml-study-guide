import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { Home, ArrowLeft, FileText, Calendar } from 'lucide-react';
import { Button } from '../components/Button';
import { PastPaperQuestion } from '../components/PastPaperQuestion';
import { TextToSpeechControls } from '../components/TextToSpeechControls';
import { useTextToSpeech } from '../hooks/useTextToSpeech';

export const PastPaperDetailView = ({ paper, onGoBack, onGoHome }) => {
  const { speak, pause, resume, stop, isSpeaking, isPaused } = useTextToSpeech();

  const fullText = useMemo(() => {
    if (!paper) return '';

    let text = `${paper.year} Examination Paper. Date: ${paper.date}. `;

    paper.sections.forEach((section) => {
      text += `${section.name}. `;
      if (section.mandatory) {
        text += 'This section is mandatory. ';
      }

      section.questions.forEach((question, index) => {
        text += `Question ${index + 1}: ${question.question}. `;
        text += `Sample Answer: ${question.sampleAnswer}. `;
      });
    });

    return text;
  }, [paper]);

  if (!paper) return null;

  const totalQuestions = paper.sections.reduce(
    (total, section) => total + section.questions.length,
    0
  );

  const totalMarks = paper.sections.reduce(
    (total, section) =>
      total + section.questions.reduce((sum, q) => sum + q.marks, 0),
    0
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8">
          <div className="flex gap-3 mb-6">
            <Button onClick={onGoBack} variant="secondary" icon={ArrowLeft}>
              Back to Papers
            </Button>
            <Button onClick={onGoHome} variant="secondary" icon={Home}>
              Home
            </Button>
          </div>

          <div className="bg-white rounded-xl shadow-md p-6 mb-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <FileText className="w-8 h-8 text-green-600" />
                  <h1 className="text-3xl font-bold text-gray-800">
                    {paper.year} Examination Paper
                  </h1>
                </div>
                <div className="flex items-center gap-2 text-gray-600">
                  <Calendar className="w-4 h-4" />
                  <span>{paper.date}</span>
                </div>
              </div>

              <div className="text-right">
                <div className="text-sm text-gray-500 mb-1">Total</div>
                <div className="text-2xl font-bold text-green-600">
                  {totalQuestions} Questions
                </div>
                <div className="text-lg text-gray-600">
                  {totalMarks} Marks
                </div>
              </div>
            </div>

            <TextToSpeechControls
              onSpeak={() => speak(fullText)}
              onPause={pause}
              onResume={resume}
              onStop={stop}
              isSpeaking={isSpeaking}
              isPaused={isPaused}
            />
          </div>
        </header>

        <div className="space-y-8">
          {paper.sections.map((section) => (
            <div key={section.id} className="bg-white rounded-xl shadow-md p-6">
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-2xl font-bold text-gray-800">
                    {section.name}
                  </h2>
                  {section.mandatory && (
                    <span className="px-3 py-1 bg-red-100 text-red-700 text-sm font-semibold rounded-full">
                      Mandatory
                    </span>
                  )}
                </div>
                <p className="text-gray-600">
                  {section.questions.length} question{section.questions.length !== 1 ? 's' : ''} • {' '}
                  {section.questions.reduce((sum, q) => sum + q.marks, 0)} marks total
                </p>
              </div>

              {section.introText && (
                <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                  <h3 className="font-semibold text-blue-900 mb-2">Question 1</h3>
                  <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                    {section.introText}
                  </p>
                </div>
              )}

              <div className="space-y-4">
                {section.questions.map((question) => (
                  <div key={question.id}>
                    <div className="text-sm font-semibold text-gray-500 mb-2">
                      {question.questionNumber || `Question ${question.id}`}
                    </div>
                    <PastPaperQuestion
                      question={question}
                      onSpeak={speak}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

PastPaperDetailView.propTypes = {
  paper: PropTypes.shape({
    id: PropTypes.number.isRequired,
    year: PropTypes.number.isRequired,
    date: PropTypes.string.isRequired,
    sections: PropTypes.arrayOf(PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired,
      mandatory: PropTypes.bool,
      questions: PropTypes.arrayOf(PropTypes.object).isRequired
    })).isRequired
  }),
  onGoBack: PropTypes.func.isRequired,
  onGoHome: PropTypes.func.isRequired
};
