import React from 'react';
import PropTypes from 'prop-types';
import { Home, FileText, Award, ArrowRight } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';

export const UnitView = ({ unit, units, onStartQuiz, onNextUnit, onGoHome }) => {
  const currentIndex = units.findIndex((u) => u.id === unit.id);
  const hasNextUnit = currentIndex < units.length - 1;

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
            <Badge variant="blue" className="mb-4">
              Unit {unit.id} • Pages {unit.pages}
            </Badge>
            <h2 className="text-3xl font-bold text-gray-800 mb-4">{unit.title}</h2>
          </div>

          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-gray-700 mb-3 flex items-center">
                <FileText className="w-6 h-6 mr-2 text-blue-500" />
                Summary
              </h3>
              <p className="text-gray-700 leading-relaxed bg-blue-50 p-4 rounded-lg">
                {unit.summary}
              </p>
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
              <p className="text-sm text-gray-600">
                <span className="font-semibold">PDF Reference:</span> Pages{' '}
                <span className="font-mono bg-white px-2 py-1 rounded">{unit.pages}</span>
              </p>
            </div>
          </div>
        </Card>

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
