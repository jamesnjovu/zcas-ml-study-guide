import React from 'react';
import PropTypes from 'prop-types';
import { Home, ArrowRight, CheckCircle } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';

export const QuizListView = ({ units, quizScores, onSelectUnit, onGoHome }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-4xl font-bold text-gray-800">All Quizzes</h1>
          <Button onClick={onGoHome} variant="secondary" icon={Home}>
            Home
          </Button>
        </div>

        <div className="space-y-4">
          {units.map((unit) => {
            const score = quizScores[unit.id];
            return (
              <Card key={unit.id} className="border-2 border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Badge variant="purple">Unit {unit.id}</Badge>
                      {score && (
                        <div className="flex items-center gap-2 bg-green-100 px-3 py-1 rounded-full">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          <span className="text-sm font-semibold text-green-700">
                            {score.score}/{score.total}
                          </span>
                        </div>
                      )}
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 mb-1">
                      {unit.title}
                    </h3>
                    <p className="text-sm text-gray-600">10 Questions</p>
                  </div>
                  <Button
                    onClick={() => onSelectUnit(unit)}
                    variant="primary"
                    icon={ArrowRight}
                  >
                    {score ? 'Retake' : 'Start'}
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
};

QuizListView.propTypes = {
  units: PropTypes.arrayOf(PropTypes.object).isRequired,
  quizScores: PropTypes.object.isRequired,
  onSelectUnit: PropTypes.func.isRequired,
  onGoHome: PropTypes.func.isRequired
};
