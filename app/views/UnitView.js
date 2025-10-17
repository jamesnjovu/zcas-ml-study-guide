'use client';

import React, { useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { Home, FileText, Award, ArrowRight, BookOpen, ChevronLeft, ChevronRight } from 'lucide-react';
import dynamic from 'next/dynamic';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { TextToSpeechControls } from '../components/TextToSpeechControls';
import { useTextToSpeech } from '../hooks/useTextToSpeech';
import { renderMarkdown } from '../utils/markdownRenderer';
import { stripMarkdown } from '../utils/stripMarkdown';

// Dynamically import PDF viewer with no SSR
const PDFViewer = dynamic(
  () => import('../components/PDFViewer'),
  { ssr: false, loading: () => <div className="text-gray-600 p-8">Loading PDF viewer...</div> }
);

export const UnitView = ({ unit, units, onStartQuiz, onNextUnit, onGoHome }) => {
  const currentIndex = units.findIndex((u) => u.id === unit.id);
  const hasNextUnit = currentIndex < units.length - 1;
  const [showPDF, setShowPDF] = useState(false);
  const [numPages, setNumPages] = useState(null);

  const { speak, pause, resume, stop, isSpeaking, isPaused, isSupported } = useTextToSpeech();

  // Prepare the full text for reading
  const fullText = useMemo(() => {
    const cleanSummary = stripMarkdown(unit.summary);
    const takeawaysText = unit.keyTakeaways
      .map((takeaway, index) => `Takeaway ${index + 1}: ${stripMarkdown(takeaway)}`)
      .join('. ');

    return `${unit.title}. ${cleanSummary}. Key Takeaways: ${takeawaysText}`;
  }, [unit]);

  // Parse page range to get start and end pages
  const getPageRange = (pageRange) => {
    if (!pageRange) return { start: 1, end: 1 };

    // Handle "All" pages - we'll need to get the total from the PDF
    if (pageRange.toLowerCase() === 'all' || pageRange.toLowerCase() === 'cheat sheet') {
      return { start: 1, end: 999 }; // Large number, will be limited by actual PDF pages
    }

    // Extract numbers from formats like "1-5", "10-15", "20"
    const matches = pageRange.match(/\d+/g);
    if (!matches) return { start: 1, end: 1 };

    const start = parseInt(matches[0], 10);
    const end = matches.length > 1 ? parseInt(matches[1], 10) : start;
    return { start, end };
  };

  const pageRange = getPageRange(unit.pages);
  const [currentPage, setCurrentPage] = useState(pageRange.start);

  // Adjust end page based on actual PDF page count
  const actualEndPage = numPages ? Math.min(pageRange.end, numPages) : pageRange.end;

  // Reset current page when unit changes or PDF is shown
  const handleShowPDF = () => {
    setCurrentPage(pageRange.start);
    setShowPDF(!showPDF);
  };

  const goToPreviousPage = () => {
    if (currentPage > pageRange.start) {
      setCurrentPage(currentPage - 1);
    }
  };

  const goToNextPage = () => {
    if (currentPage < actualEndPage) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handleDocumentLoadSuccess = ({ numPages: loadedPages }) => {
    setNumPages(loadedPages);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <Button onClick={onGoHome} variant="secondary" icon={Home} size="sm">
            Home
          </Button>
          <Badge variant="gray">
            Unit {unit.id} of {units.length}
          </Badge>
        </div>

        <Card className="mb-6">
          <div className="mb-6">
            <div className="flex items-center justify-between flex-wrap gap-4 mb-4">
              <Badge variant="blue">
                Unit {unit.id} • Pages {unit.pages}
              </Badge>
              <TextToSpeechControls
                isSpeaking={isSpeaking}
                isPaused={isPaused}
                isSupported={isSupported}
                onSpeak={speak}
                onPause={pause}
                onResume={resume}
                onStop={stop}
                text={fullText}
                label="Read Unit Content"
              />
            </div>
            <h2 className="text-3xl font-bold text-gray-800 mb-4">{unit.title}</h2>
          </div>

          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-gray-700 mb-3 flex items-center">
                <FileText className="w-6 h-6 mr-2 text-blue-500" />
                Summary
              </h3>
              <div className="text-gray-700 leading-relaxed bg-blue-50 p-4 rounded-lg">
                {renderMarkdown(unit.summary)}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-700 mb-4">Key Takeaways</h3>
              <div className="space-y-3">
                {unit.keyTakeaways.map((takeaway, index) => (
                  <div key={index} className="flex items-start bg-purple-50 p-4 rounded-lg">
                    <span className="inline-block bg-purple-500 text-white text-sm font-bold rounded-full w-7 h-7 flex items-center justify-center mr-3 flex-shrink-0">
                      {index + 1}
                    </span>
                    <span className="text-gray-700">{takeaway}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg border-l-4 border-blue-500">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-600">
                  <span className="font-semibold">PDF Reference:</span> Pages{' '}
                  <span className="font-mono bg-white px-2 py-1 rounded">{unit.pages}</span>
                </p>
                <Button
                  onClick={handleShowPDF}
                  variant={showPDF ? "secondary" : "primary"}
                  size="sm"
                  icon={BookOpen}
                >
                  {showPDF ? 'Hide' : 'View'} Lecture Notes
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {showPDF && (
          <Card className="mb-6">
            <div className="mb-4">
              <h3 className="text-xl font-semibold text-gray-700 mb-2 flex items-center">
                <BookOpen className="w-6 h-6 mr-2 text-green-500" />
                Lecture Notes - Pages {unit.pages}
              </h3>
              <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded mb-4">
                <p className="text-sm text-blue-800 font-medium">
                  📖 This unit covers pages {unit.pages} of the lecture notes
                </p>
              </div>

              {/* Page Navigation Controls */}
              <div className="flex items-center justify-between bg-gray-50 p-4 rounded-lg mb-4">
                <Button
                  onClick={goToPreviousPage}
                  variant="secondary"
                  size="sm"
                  icon={ChevronLeft}
                  disabled={currentPage <= pageRange.start}
                >
                  Previous
                </Button>

                <div className="text-center">
                  <div className="text-sm text-gray-600">Page</div>
                  <div className="text-2xl font-bold text-gray-800">
                    {currentPage}
                  </div>
                  <div className="text-xs text-gray-500">
                    {numPages ? `of ${pageRange.start}-${actualEndPage}` : 'Loading...'}
                  </div>
                </div>

                <Button
                  onClick={goToNextPage}
                  variant="secondary"
                  size="sm"
                  icon={ChevronRight}
                  disabled={currentPage >= actualEndPage}
                >
                  Next
                </Button>
              </div>
            </div>

            {/* PDF Viewer */}
            <div className="relative w-full bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center" style={{ minHeight: '800px' }}>
              <PDFViewer
                pageNumber={currentPage}
                pdfFile={unit.pdfFile || 'lecture_notes.pdf'}
                onDocumentLoadSuccess={handleDocumentLoadSuccess}
              />
            </div>

            <div className="mt-4 text-center">
              <a
                href={`/zcas-ml-study-guide/${unit.pdfFile || 'lecture_notes.pdf'}#page=${pageRange.start}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline text-sm"
              >
                Open full PDF in new tab
              </a>
            </div>
          </Card>
        )}

        <div className="flex gap-4">
          <Button
            onClick={onStartQuiz}
            variant="success"
            size="lg"
            icon={Award}
            className="flex-1"
          >
            Take Quiz
          </Button>
          {hasNextUnit && (
            <Button
              onClick={onNextUnit}
              variant="primary"
              size="lg"
              icon={ArrowRight}
              className="flex-1"
            >
              Next Unit
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

UnitView.propTypes = {
  unit: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    pages: PropTypes.string.isRequired,
    summary: PropTypes.string.isRequired,
    keyTakeaways: PropTypes.arrayOf(PropTypes.string).isRequired
  }).isRequired,
  units: PropTypes.arrayOf(PropTypes.object).isRequired,
  onStartQuiz: PropTypes.func.isRequired,
  onNextUnit: PropTypes.func.isRequired,
  onGoHome: PropTypes.func.isRequired
};
