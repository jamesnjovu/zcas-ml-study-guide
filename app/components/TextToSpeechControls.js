import React from 'react';
import PropTypes from 'prop-types';
import { Volume2, Pause, Play, StopCircle } from 'lucide-react';
import { Button } from './Button';

export const TextToSpeechControls = ({
  isSpeaking,
  isPaused,
  isSupported,
  onSpeak,
  onPause,
  onResume,
  onStop,
  text,
  label = 'Read aloud'
}) => {
  if (!isSupported) {
    return null;
  }

  return (
    <div className="flex items-center gap-2">
      {!isSpeaking ? (
        <Button
          onClick={() => onSpeak(text)}
          variant="primary"
          size="sm"
          icon={Volume2}
        >
          {label}
        </Button>
      ) : (
        <>
          {isPaused ? (
            <Button
              onClick={onResume}
              variant="success"
              size="sm"
              icon={Play}
            >
              Resume
            </Button>
          ) : (
            <Button
              onClick={onPause}
              variant="secondary"
              size="sm"
              icon={Pause}
            >
              Pause
            </Button>
          )}
          <Button
            onClick={onStop}
            variant="danger"
            size="sm"
            icon={StopCircle}
          >
            Stop
          </Button>
        </>
      )}
    </div>
  );
};

TextToSpeechControls.propTypes = {
  isSpeaking: PropTypes.bool.isRequired,
  isPaused: PropTypes.bool.isRequired,
  isSupported: PropTypes.bool.isRequired,
  onSpeak: PropTypes.func.isRequired,
  onPause: PropTypes.func.isRequired,
  onResume: PropTypes.func.isRequired,
  onStop: PropTypes.func.isRequired,
  text: PropTypes.string.isRequired,
  label: PropTypes.string
};
