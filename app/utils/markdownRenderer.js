import React from 'react';

// Function to render text with markdown formatting including headings
export const renderMarkdown = (text) => {
  if (!text) return null;

  // Split by code blocks (```...```)
  const parts = text.split(/(```[\s\S]*?```)/g);

  return parts.map((part, index) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      // Extract language and code
      const content = part.slice(3, -3);
      const lines = content.split('\n');
      const language = lines[0].trim();
      const code = lines.slice(1).join('\n');

      return (
        <pre key={index} className="my-3 bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
          {language && (
            <div className="text-xs text-gray-400 mb-2 uppercase">{language}</div>
          )}
          <code className="text-sm font-mono">{code}</code>
        </pre>
      );
    }
    // Process inline markdown for regular text
    return (
      <span key={index} className="whitespace-pre-wrap">
        {renderInlineMarkdown(part)}
      </span>
    );
  });
};

// Function to render inline markdown (bold, italic, inline code, headings)
const renderInlineMarkdown = (text) => {
  if (!text) return null;

  // Split text by lines to handle headings
  const lines = text.split('\n');
  const result = [];

  lines.forEach((line, lineIndex) => {
    // Check for headings (###, ##, #)
    if (line.startsWith('### ')) {
      result.push(
        <h3 key={`h3-${lineIndex}`} className="text-lg font-bold text-gray-800 mt-4 mb-2">
          {processInlineFormatting(line.slice(4))}
        </h3>
      );
    } else if (line.startsWith('## ')) {
      result.push(
        <h2 key={`h2-${lineIndex}`} className="text-xl font-bold text-gray-800 mt-5 mb-3">
          {processInlineFormatting(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith('# ')) {
      result.push(
        <h1 key={`h1-${lineIndex}`} className="text-2xl font-bold text-gray-800 mt-6 mb-3">
          {processInlineFormatting(line.slice(2))}
        </h1>
      );
    } else {
      // Regular line with inline formatting
      const processedLine = processInlineFormatting(line);
      result.push(
        <span key={`line-${lineIndex}`}>
          {processedLine}
          {lineIndex < lines.length - 1 && '\n'}
        </span>
      );
    }
  });

  return result;
};

// Function to process bold, italic, and inline code
const processInlineFormatting = (text) => {
  if (!text) return null;

  const parts = [];
  let currentIndex = 0;
  let key = 0;

  // Regex to match markdown patterns (bold, italic, inline code)
  const markdownRegex = /(\*\*[\s\S]+?\*\*|\*[\s\S]+?\*|`[^`]+`)/g;
  let match;

  while ((match = markdownRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > currentIndex) {
      parts.push(
        <span key={`text-${key++}`}>
          {text.substring(currentIndex, match.index)}
        </span>
      );
    }

    const matchedText = match[0];

    // Bold text (**text**)
    if (matchedText.startsWith('**') && matchedText.endsWith('**')) {
      parts.push(
        <strong key={`bold-${key++}`} className="font-bold">
          {matchedText.slice(2, -2)}
        </strong>
      );
    }
    // Italic text (*text*)
    else if (matchedText.startsWith('*') && matchedText.endsWith('*')) {
      parts.push(
        <em key={`italic-${key++}`} className="italic">
          {matchedText.slice(1, -1)}
        </em>
      );
    }
    // Inline code (`code`)
    else if (matchedText.startsWith('`') && matchedText.endsWith('`')) {
      parts.push(
        <code
          key={`code-${key++}`}
          className="bg-gray-200 text-red-600 px-1.5 py-0.5 rounded text-sm font-mono"
        >
          {matchedText.slice(1, -1)}
        </code>
      );
    }

    currentIndex = match.index + matchedText.length;
  }

  // Add remaining text
  if (currentIndex < text.length) {
    parts.push(
      <span key={`text-${key++}`}>
        {text.substring(currentIndex)}
      </span>
    );
  }

  return parts.length > 0 ? parts : text;
};
