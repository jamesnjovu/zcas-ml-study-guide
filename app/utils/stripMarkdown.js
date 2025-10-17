/**
 * Strip markdown formatting from text for text-to-speech
 * @param {string} text - Text with markdown formatting
 * @returns {string} - Plain text without markdown
 */
export const stripMarkdown = (text) => {
  if (!text) return '';

  let cleanText = text;

  // Remove code blocks (```code```)
  cleanText = cleanText.replace(/```[\s\S]*?```/g, ' code block ');

  // Remove inline code (`code`)
  cleanText = cleanText.replace(/`([^`]+)`/g, '$1');

  // Remove bold (**text** or __text__)
  cleanText = cleanText.replace(/\*\*([^*]+)\*\*/g, '$1');
  cleanText = cleanText.replace(/__([^_]+)__/g, '$1');

  // Remove italic (*text* or _text_)
  cleanText = cleanText.replace(/\*([^*]+)\*/g, '$1');
  cleanText = cleanText.replace(/_([^_]+)_/g, '$1');

  // Remove headings (### text)
  cleanText = cleanText.replace(/^#{1,6}\s+/gm, '');

  // Remove links [text](url)
  cleanText = cleanText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

  // Remove images ![alt](url)
  cleanText = cleanText.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');

  // Remove horizontal rules
  cleanText = cleanText.replace(/^[-*_]{3,}$/gm, '');

  // Remove blockquotes
  cleanText = cleanText.replace(/^>\s+/gm, '');

  // Remove list markers (-, *, +, 1., 2., etc.)
  cleanText = cleanText.replace(/^[\s]*[-*+]\s+/gm, '');
  cleanText = cleanText.replace(/^[\s]*\d+\.\s+/gm, '');

  // Clean up extra whitespace
  cleanText = cleanText.replace(/\s+/g, ' ').trim();

  return cleanText;
};
