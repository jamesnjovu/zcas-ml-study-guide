import React from 'react';
import PropTypes from 'prop-types';
import { Home, FileText, Calendar } from 'lucide-react';
import { Button } from '../components/Button';

export const PastPapersView = ({ pastPapers, onSelectPaper, onGoHome }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8">
          <Button onClick={onGoHome} variant="secondary" icon={Home} className="mb-6">
            Back to Home
          </Button>

          <h1 className="text-4xl font-bold text-gray-800 mb-3">
            Past Examination Papers
          </h1>
          <p className="text-lg text-gray-600">
            Practice with previous years&apos; exam questions and sample answers
          </p>
        </header>

        <div className="space-y-4">
          {pastPapers.map((paper) => (
            <div
              key={paper.id}
              className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 p-6 border border-gray-200 cursor-pointer"
              onClick={() => onSelectPaper(paper)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-3">
                    <FileText className="w-6 h-6 text-green-600" />
                    <h2 className="text-2xl font-bold text-gray-800">
                      {paper.year} Examination Paper
                    </h2>
                  </div>

                  <div className="flex items-center gap-2 text-gray-600 mb-4">
                    <Calendar className="w-4 h-4" />
                    <span className="text-sm">{paper.date}</span>
                  </div>

                  <div className="space-y-2">
                    {paper.sections.map((section) => (
                      <div key={section.id} className="flex items-center gap-2">
                        <span className="inline-block w-2 h-2 bg-green-500 rounded-full"></span>
                        <span className="text-sm text-gray-700">
                          {section.name}: {section.questions.length} question{section.questions.length !== 1 ? 's' : ''}
                          {section.mandatory && (
                            <span className="ml-2 text-xs font-semibold text-red-600">
                              (Mandatory)
                            </span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm text-gray-500 mb-2">Total Questions</div>
                  <div className="text-3xl font-bold text-green-600">
                    {paper.sections.reduce((total, section) => total + section.questions.length, 0)}
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200">
                <Button
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectPaper(paper);
                  }}
                  variant="primary"
                  className="w-full sm:w-auto"
                >
                  View Paper
                </Button>
              </div>
            </div>
          ))}
        </div>

        {pastPapers.length === 0 && (
          <div className="text-center py-12">
            <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">No past papers available yet.</p>
          </div>
        )}
      </div>
    </div>
  );
};

PastPapersView.propTypes = {
  pastPapers: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.number.isRequired,
    year: PropTypes.number.isRequired,
    date: PropTypes.string.isRequired,
    sections: PropTypes.arrayOf(PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired,
      mandatory: PropTypes.bool,
      questions: PropTypes.array.isRequired
    })).isRequired
  })).isRequired,
  onSelectPaper: PropTypes.func.isRequired,
  onGoHome: PropTypes.func.isRequired
};
