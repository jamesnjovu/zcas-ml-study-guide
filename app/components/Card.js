import React from 'react';
import PropTypes from 'prop-types';

export const Card = ({
  children,
  onClick,
  className = '',
  hoverable = false
}) => {
  const baseStyles = 'bg-white rounded-xl shadow-lg p-6';
  const hoverStyles = hoverable
    ? 'hover:shadow-xl transition-all cursor-pointer border-2 border-gray-200 hover:border-blue-400'
    : '';

  const classes = `${baseStyles} ${hoverStyles} ${className}`;

  return (
    <div onClick={onClick} className={classes}>
      {children}
    </div>
  );
};

Card.propTypes = {
  children: PropTypes.node.isRequired,
  onClick: PropTypes.func,
  className: PropTypes.string,
  hoverable: PropTypes.bool
};
