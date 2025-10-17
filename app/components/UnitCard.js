import React from 'react';
import PropTypes from 'prop-types';
import { Award } from 'lucide-react';
import { Card } from './Card';
import { Badge } from './Badge';
import { renderMarkdown } from '../utils/markdownRenderer';

export const UnitCard = ({ unit, onSelect, quizScore }) => {
  return (
    <Card onClick={() => onSelect(unit)} hoverable>
      <div className="flex items-center justify-between mb-4">
        <Badge variant="blue">Unit {unit.id}</Badge>
        {quizScore && <Award className="w-6 h-6 text-yellow-500" />}
      </div>
      <h3 className="text-xl font-bold text-gray-800 mb-2">{unit.title}</h3>
      <p className="text-sm text-gray-600 mb-2">Pages {unit.pages}</p>
      {unit.summary && (
        <div className="text-sm text-gray-700 mb-4 line-clamp-3">
          {renderMarkdown(unit.summary)}
        </div>
      )}
      {quizScore && (
        <div className="bg-green-50 px-3 py-2 rounded-lg">
          <span className="text-sm font-semibold text-green-700">
            Quiz Score: {quizScore.score}/{quizScore.total}
          </span>
        </div>
      )}
    </Card>
  );
};

UnitCard.propTypes = {
  unit: PropTypes.shape({
    id: PropTypes.number.isRequired,
    title: PropTypes.string.isRequired,
    pages: PropTypes.string.isRequired,
    summary: PropTypes.string,
    keyTakeaways: PropTypes.arrayOf(PropTypes.string),
    quiz: PropTypes.array
  }).isRequired,
  onSelect: PropTypes.func.isRequired,
  quizScore: PropTypes.shape({
    score: PropTypes.number,
    total: PropTypes.number
  })
};
