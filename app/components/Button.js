import React from 'react';
import PropTypes from 'prop-types';

export const Button = ({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  className = '',
  icon: Icon
}) => {
  const baseStyles = 'rounded-lg font-semibold transition-all flex items-center justify-center gap-2';

  const variantStyles = {
    primary: 'bg-blue-500 text-white hover:bg-blue-600',
    secondary: 'bg-gray-500 text-white hover:bg-gray-600',
    success: 'bg-green-500 text-white hover:bg-green-600',
    danger: 'bg-red-500 text-white hover:bg-red-600',
    purple: 'bg-purple-500 text-white hover:bg-purple-600'
  };

  const sizeStyles = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-6 py-4 text-lg'
  };

  const disabledStyles = disabled
    ? 'bg-gray-300 text-gray-500 cursor-not-allowed hover:bg-gray-300'
    : '';

  const classes = `${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${disabledStyles} ${className}`;

  return (
    <button onClick={onClick} disabled={disabled} className={classes}>
      {Icon && <Icon className="w-5 h-5" />}
      {children}
    </button>
  );
};

Button.propTypes = {
  children: PropTypes.node.isRequired,
  onClick: PropTypes.func,
  variant: PropTypes.oneOf(['primary', 'secondary', 'success', 'danger', 'purple']),
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  disabled: PropTypes.bool,
  className: PropTypes.string,
  icon: PropTypes.elementType
};
