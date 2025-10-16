import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { List, Trash2 } from 'lucide-react';
import { Button } from '../components/Button';
import { UnitCard } from '../components/UnitCard';

export const HomeView = ({ units, quizScores, onSelectUnit, onViewQuizList, onClearData }) => {
  const [showConfirm, setShowConfirm] = useState(false);

  const handleClearData = () => {
    if (onClearData) {
      onClearData();
      setShowConfirm(false);
    }
  };

  const hasData = Object.keys(quizScores).length > 0;
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-800 mb-4">
            Machine Learning Study Guide
          </h1>
          <p className="text-xl text-gray-600">
            Master ML concepts through structured learning units
          </p>
          <div className="mt-6 flex gap-4 justify-center flex-wrap">
            <Button onClick={onViewQuizList} variant="purple" icon={List}>
              View All Quizzes
            </Button>
            {hasData && (
              <Button onClick={() => setShowConfirm(true)} variant="danger" icon={Trash2}>
                Clear All Data
              </Button>
            )}
          </div>

          {showConfirm && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl p-6 max-w-md w-full shadow-2xl">
                <h3 className="text-xl font-bold text-gray-800 mb-4">Clear All Data?</h3>
                <p className="text-gray-600 mb-6">
                  This will permanently delete all your quiz scores and progress. This action cannot be undone.
                </p>
                <div className="flex gap-4">
                  <Button
                    onClick={() => setShowConfirm(false)}
                    variant="secondary"
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleClearData}
                    variant="danger"
                    className="flex-1"
                  >
                    Clear Data
                  </Button>
                </div>
              </div>
            </div>
          )}
        </header>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {units.map((unit) => (
            <UnitCard
              key={unit.id}
              unit={unit}
              onSelect={onSelectUnit}
              quizScore={quizScores[unit.id]}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

HomeView.propTypes = {
  units: PropTypes.arrayOf(PropTypes.object).isRequired,
  quizScores: PropTypes.object.isRequired,
  onSelectUnit: PropTypes.func.isRequired,
  onViewQuizList: PropTypes.func.isRequired,
  onClearData: PropTypes.func
};
