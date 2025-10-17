import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { Home, ArrowLeft, FileText, Calendar, Volume2 } from 'lucide-react';
import { Button } from '../components/Button';
import { PastPaperQuestion } from '../components/PastPaperQuestion';
import { TextToSpeechControls } from '../components/TextToSpeechControls';
import { useTextToSpeech } from '../hooks/useTextToSpeech';
import { renderMarkdown } from '../utils/markdownRenderer';
import { stripMarkdown } from '../utils/stripMarkdown';

export const PastPaperDetailView = ({ paper, onGoBack, onGoHome }) => {
  const { speak, pause, resume, stop, isSpeaking, isPaused, isSupported } = useTextToSpeech();

  const fullText = useMemo(() => {
    if (!paper) return '';

    let text = `${paper.year} Examination Paper. Date: ${paper.date}. `;

    paper.sections.forEach((section) => {
      text += `${section.name}. `;
      if (section.mandatory) {
        text += 'This section is mandatory. ';
      }
      if (section.introText) {
        text += `${stripMarkdown(section.introText)}. `;
      }

      section.questions.forEach((question) => {
        if (question.isParentQuestion) {
          if (question.question) {
            text += `Question ${question.questionNumber}: ${stripMarkdown(question.question)}. `;
          }
          if (question.subQuestions) {
            question.subQuestions.forEach((subQuestion) => {
              text += `Part ${subQuestion.questionNumber}: ${stripMarkdown(subQuestion.question)}. `;
              if (subQuestion.sampleAnswer) {
                text += `Sample Answer: ${stripMarkdown(subQuestion.sampleAnswer)}. `;
              }
            });
          }
        } else {
          text += `Question ${question.questionNumber}: ${stripMarkdown(question.question)}. `;
          if (question.sampleAnswer) {
            text += `Sample Answer: ${stripMarkdown(question.sampleAnswer)}. `;
          }
        }
      });
    });

    return text;
  }, [paper]);

  if (!paper) return null;

  const totalQuestions = paper.sections.reduce((total, section) => {
    return total + section.questions.reduce((questionTotal, q) => {
      if (q.subQuestions) {
        return questionTotal + q.subQuestions.length;
      }
      return questionTotal + 1;
    }, 0);
  }, 0);

  const totalMarks = paper.sections.reduce((total, section) => {
    return total + section.questions.reduce((sum, q) => {
      if (q.subQuestions) {
        return sum + q.subQuestions.reduce((subSum, subQ) => subSum + (subQ.marks || 0), 0);
      }
      return sum + (q.marks || 0);
    }, 0);
  }, 0);

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
              isSupported={isSupported}
              text={fullText}
              label="Read Entire Paper"
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
                <div className="mb-6 p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
                  <h3 className="font-semibold text-yellow-900 mb-2">Instructions</h3>
                  <div className="text-gray-700 leading-relaxed">
                    {renderMarkdown(section.introText)}
                  </div>
                </div>
              )}

              <div className="space-y-6">
                {section.questions.map((question) => (
                  <div key={question.id}>
                    {question.isParentQuestion ? (
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
                        {question.question && (
                          <div className="mb-4">
                            <div className="flex items-start justify-between gap-4 mb-3">
                              <h3 className="text-xl font-bold text-blue-900">
                                Question {question.questionNumber}
                              </h3>
                              <button
                                onClick={() => {
                                  let text = `Question ${question.questionNumber}: ${stripMarkdown(question.question)}. `;
                                  if (question.subQuestions) {
                                    question.subQuestions.forEach((subQ) => {
                                      text += `Part ${subQ.questionNumber}: ${stripMarkdown(subQ.question)}. `;
                                      if (subQ.sampleAnswer) {
                                        text += `Sample Answer: ${stripMarkdown(subQ.sampleAnswer)}. `;
                                      }
                                    });
                                  }
                                  speak(text);
                                }}
                                className="flex-shrink-0 p-2 text-purple-600 hover:bg-purple-100 rounded-lg transition-colors"
                                title="Read entire question with all sub-questions and answers"
                              >
                                <Volume2 className="w-5 h-5" />
                              </button>
                            </div>
                            <div className="text-gray-700 leading-relaxed">
                              {renderMarkdown(question.question)}
                            </div>
                          </div>
                        )}
                        {!question.question && (
                          <div className="flex items-start justify-between gap-4 mb-4">
                            <h3 className="text-xl font-bold text-blue-900">
                              Question {question.questionNumber}
                            </h3>
                            <button
                              onClick={() => {
                                let text = `Question ${question.questionNumber}. `;
                                if (question.subQuestions) {
                                  question.subQuestions.forEach((subQ) => {
                                    text += `Part ${subQ.questionNumber}: ${stripMarkdown(subQ.question)}. `;
                                    if (subQ.sampleAnswer) {
                                      text += `Sample Answer: ${stripMarkdown(subQ.sampleAnswer)}. `;
                                    }
                                  });
                                }
                                speak(text);
                              }}
                              className="flex-shrink-0 p-2 text-purple-600 hover:bg-purple-100 rounded-lg transition-colors"
                              title="Read all sub-questions and answers"
                            >
                              <Volume2 className="w-5 h-5" />
                            </button>
                          </div>
                        )}

                        {question.subQuestions && (
                          <div className="space-y-4 mt-6">
                            {question.subQuestions.map((subQuestion) => (
                              <div key={subQuestion.id} className="bg-white rounded-lg p-4 shadow-sm">
                                <div className="text-sm font-semibold text-indigo-600 mb-2">
                                  {question.questionNumber}({subQuestion.questionNumber})
                                </div>
                                <PastPaperQuestion
                                  question={subQuestion}
                                  onSpeak={speak}
                                />
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>
                        <div className="text-sm font-semibold text-gray-500 mb-2">
                          Question {question.questionNumber}
                        </div>
                        <PastPaperQuestion
                          question={question}
                          onSpeak={speak}
                        />
                      </div>
                    )}
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
