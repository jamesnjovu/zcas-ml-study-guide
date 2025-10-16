import React from 'react';
import PropTypes from 'prop-types';

export const Badge = ({ children, variant = 'blue', size = 'md', className = '' }) => {
  const baseStyles = 'rounded-full font-bold inline-block';

  const variantStyles = {
    blue: 'bg-blue-500 text-white',
    purple: 'bg-purple-500 text-white',
    green: 'bg-green-100 text-green-700',
    gray: 'bg-gray-100 text-gray-800'
  };

  const sizeStyles = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const classes = `${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`;

  return <span className={classes}>{children}</span>;
};

Badge.propTypes = {
  children: PropTypes.node.isRequired,
  variant: PropTypes.oneOf(['blue', 'purple', 'green', 'gray']),
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  className: PropTypes.string
};
