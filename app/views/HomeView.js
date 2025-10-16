import React from 'react';
import PropTypes from 'prop-types';
import { List } from 'lucide-react';
import { Button } from '../components/Button';
import { UnitCard } from '../components/UnitCard';

export const HomeView = ({ units, quizScores, onSelectUnit, onViewQuizList }) => {
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
          <div className="mt-6 flex gap-4 justify-center">
            <Button onClick={onViewQuizList} variant="purple" icon={List}>
              View All Quizzes
            </Button>
          </div>
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
  onViewQuizList: PropTypes.func.isRequired
};
