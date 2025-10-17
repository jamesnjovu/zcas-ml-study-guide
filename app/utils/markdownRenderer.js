import React from 'react';

// Convert LaTeX symbols to Unicode
const convertLatexToUnicode = (text) => {
  let result = text;

  // Handle \frac{numerator}{denominator}
  result = result.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)');

  // Handle \sqrt{content}
  result = result.replace(/\\sqrt\{([^}]+)\}/g, '√($1)');

  // Handle \hat{x}, \bar{x}, \tilde{x}
  result = result.replace(/\\hat\{([^}]+)\}/g, '$1̂');
  result = result.replace(/\\bar\{([^}]+)\}/g, '$1̄');
  result = result.replace(/\\tilde\{([^}]+)\}/g, '$1̃');

  // Handle subscripts with curly braces _{text}
  result = result.replace(/_\{([^}]+)\}/g, (match, content) => {
    // Convert multi-character subscripts
    const subscriptMap = {
      'res': 'ᵣₑₛ', 'tot': 'ₜₒₜ', 'max': 'ₘₐₓ', 'min': 'ₘᵢₙ',
      'avg': 'ₐᵥ𝓰', 'sum': 'ₛᵤₘ', 'train': 'ₜᵣₐᵢₙ', 'test': 'ₜₑₛₜ'
    };
    if (subscriptMap[content]) {
      return subscriptMap[content];
    }
    // Convert each character to subscript if possible
    return content.split('').map(c => {
      const subs = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
        'v': 'ᵥ', 'x': 'ₓ'
      };
      return subs[c] || c;
    }).join('');
  });

  // Handle superscripts with curly braces ^{text}
  result = result.replace(/\^\{([^}]+)\}/g, (match, content) => {
    return content.split('').map(c => {
      const sups = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        'n': 'ⁿ', 'i': 'ⁱ', 'T': 'ᵀ', '+': '⁺', '-': '⁻',
        '=': '⁼', '(': '⁽', ')': '⁾'
      };
      return sups[c] || c;
    }).join('');
  });

  const replacements = {
    // Greek letters
    '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
    '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ',
    '\\lambda': 'λ', '\\mu': 'μ', '\\pi': 'π', '\\rho': 'ρ',
    '\\sigma': 'σ', '\\tau': 'τ', '\\phi': 'φ', '\\omega': 'ω',
    // Uppercase Greek
    '\\Gamma': 'Γ', '\\Delta': 'Δ', '\\Theta': 'Θ', '\\Lambda': 'Λ',
    '\\Sigma': 'Σ', '\\Phi': 'Φ', '\\Omega': 'Ω',
    // Math operators
    '\\times': '×', '\\div': '÷', '\\pm': '±', '\\mp': '∓',
    '\\leq': '≤', '\\geq': '≥', '\\neq': '≠', '\\approx': '≈',
    '\\equiv': '≡', '\\sum': '∑', '\\prod': '∏', '\\int': '∫',
    '\\partial': '∂', '\\nabla': '∇', '\\infty': '∞',
    '\\in': '∈', '\\notin': '∉', '\\subset': '⊂', '\\supset': '⊃',
    '\\cap': '∩', '\\cup': '∪', '\\forall': '∀', '\\exists': '∃',
    '\\emptyset': '∅', '\\rightarrow': '→', '\\leftarrow': '←',
    '\\Rightarrow': '⇒', '\\Leftarrow': '⇐', '\\leftrightarrow': '↔',
    // Subscripts
    '_0': '₀', '_1': '₁', '_2': '₂', '_3': '₃', '_4': '₄',
    '_5': '₅', '_6': '₆', '_7': '₇', '_8': '₈', '_9': '₉',
    '_i': 'ᵢ', '_j': 'ⱼ', '_n': 'ₙ', '_x': 'ₓ',
    // Superscripts
    '^0': '⁰', '^1': '¹', '^2': '²', '^3': '³', '^4': '⁴',
    '^5': '⁵', '^6': '⁶', '^7': '⁷', '^8': '⁸', '^9': '⁹',
    '^n': 'ⁿ', '^T': 'ᵀ',
    // Additional symbols
    '\\cdot': '⋅', '\\dots': '…', '\\ldots': '…',
  };

  for (const [latex, unicode] of Object.entries(replacements)) {
    result = result.split(latex).join(unicode);
  }

  return result;
};

// Function to render text with markdown formatting including headings, lists, and math
export const renderMarkdown = (text) => {
  if (!text) return null;

  // Split by display math blocks (\[...\]) first - handle multiline blocks
  const mathBlockParts = text.split(/(\\\[\s*[\s\S]*?\s*\\\])/g);

  return mathBlockParts.map((part, partIndex) => {
    // Handle display math blocks \[ ... \]
    if (part.match(/^\\\[\s*[\s\S]*?\s*\\\]$/)) {
      // Extract content and clean up whitespace
      const mathContent = part
        .replace(/^\\\[\s*/, '') // Remove opening \[ and whitespace
        .replace(/\s*\\\]$/, '') // Remove closing \] and whitespace
        .replace(/\s+/g, ' ')    // Normalize whitespace
        .trim();

      const unicodeMath = convertLatexToUnicode(mathContent);
      return (
        <div
          key={`mathblock-${partIndex}`}
          className="my-4 p-4 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg text-center"
        >
          <div className="text-lg text-blue-900 font-mono">
            {unicodeMath}
          </div>
        </div>
      );
    }

    // Split by code blocks (```...```)
    const parts = part.split(/(```[\s\S]*?```)/g);

    return parts.map((subpart, index) => {
      if (subpart.startsWith('```') && subpart.endsWith('```')) {
        const content = subpart.slice(3, -3);
        const lines = content.split('\n');
        const language = lines[0].trim();
        const code = lines.slice(1).join('\n');

        return (
          <pre
            key={`${partIndex}-${index}`}
            className="my-3 bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto"
          >
            {language && (
              <div className="text-xs text-gray-400 mb-2 uppercase">{language}</div>
            )}
            <code className="text-sm font-mono">{code}</code>
          </pre>
        );
      }
      return (
        <span key={`${partIndex}-${index}`} className="whitespace-pre-wrap">
          {renderInlineMarkdown(subpart)}
        </span>
      );
    });
  });
};

// Function to render inline markdown (bold, italic, inline code, headings, and lists)
const renderInlineMarkdown = (text) => {
  if (!text) return null;

  const lines = text.split('\n');
  const result = [];

  lines.forEach((line, lineIndex) => {
    // Horizontal rule (---)
    if (line.trim() === '---' || line.trim() === '***' || line.trim() === '___') {
      result.push(
        <hr
          key={`hr-${lineIndex}`}
          className="my-4 border-t-2 border-gray-300"
        />
      );
    }
    // Headings
    else if (line.startsWith('#### ')) {
      result.push(
        <h4
          key={`h4-${lineIndex}`}
          className="text-base font-bold text-gray-800 mt-3 mb-2"
        >
          {processInlineFormatting(line.slice(5))}
        </h4>
      );
    } else if (line.startsWith('### ')) {
      result.push(
        <h3
          key={`h3-${lineIndex}`}
          className="text-lg font-bold text-gray-800 mt-4 mb-2"
        >
          {processInlineFormatting(line.slice(4))}
        </h3>
      );
    } else if (line.startsWith('## ')) {
      result.push(
        <h2
          key={`h2-${lineIndex}`}
          className="text-xl font-bold text-gray-800 mt-5 mb-3"
        >
          {processInlineFormatting(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith('# ')) {
      result.push(
        <h1
          key={`h1-${lineIndex}`}
          className="text-2xl font-bold text-gray-800 mt-6 mb-3"
        >
          {processInlineFormatting(line.slice(2))}
        </h1>
      );
    }
    // Bulleted lists
    else if (line.trim().startsWith('- ')) {
      result.push(
        <li
          key={`li-${lineIndex}`}
          className="ml-6 list-disc text-gray-700 leading-relaxed"
        >
          {processInlineFormatting(line.trim().slice(2))}
        </li>
      );
    }
    // Inline math (LaTeX-style with \(...\))
    else if (line.match(/\\\(.*?\\\)/g)) {
      const mathParts = line.split(/(\\\([^)]*?\\\))/g);
      result.push(
        <span key={`math-${lineIndex}`} className="text-gray-700">
          {mathParts.map((part, i) => {
            const mathMatch = part.match(/^\\\(([^)]*?)\\\)$/);
            if (mathMatch) {
              const unicodeMath = convertLatexToUnicode(mathMatch[1]);
              return (
                <code
                  key={`mathcode-${i}`}
                  className="bg-blue-50 text-blue-800 font-mono px-2 py-0.5 rounded mx-1 border border-blue-200"
                >
                  {unicodeMath}
                </code>
              );
            }
            return <span key={`text-${i}`}>{processInlineFormatting(part)}</span>;
          })}
        </span>
      );
    }
    // Regular text
    else {
      result.push(
        <p key={`line-${lineIndex}`} className="text-gray-700 leading-relaxed">
          {processInlineFormatting(line)}
        </p>
      );
    }
  });

  return result;
};

// Function to process bold, italic, inline code, and inline math
const processInlineFormatting = (text) => {
  if (!text) return null;

  const parts = [];
  let currentIndex = 0;
  let key = 0;

  // Include inline math \( \) in the regex
  const markdownRegex = /(\*\*[\s\S]+?\*\*|\*[\s\S]+?\*|`[^`]+`|\\\([^)]*?\\\))/g;
  let match;

  while ((match = markdownRegex.exec(text)) !== null) {
    if (match.index > currentIndex) {
      parts.push(
        <span key={`text-${key++}`}>
          {text.substring(currentIndex, match.index)}
        </span>
      );
    }

    const matchedText = match[0];

    // Inline math \( ... \)
    const inlineMathMatch = matchedText.match(/^\\\(([^)]*?)\\\)$/);
    if (inlineMathMatch) {
      const unicodeMath = convertLatexToUnicode(inlineMathMatch[1]);
      parts.push(
        <code
          key={`math-${key++}`}
          className="bg-blue-50 text-blue-800 font-mono px-2 py-0.5 rounded mx-1 border border-blue-200"
        >
          {unicodeMath}
        </code>
      );
    }
    // Bold **text**
    else if (matchedText.startsWith('**') && matchedText.endsWith('**')) {
      parts.push(
        <strong key={`bold-${key++}`} className="font-semibold text-gray-900">
          {matchedText.slice(2, -2)}
        </strong>
      );
    }
    // Italic *text*
    else if (matchedText.startsWith('*') && matchedText.endsWith('*')) {
      parts.push(
        <em key={`italic-${key++}`} className="italic text-gray-700">
          {matchedText.slice(1, -1)}
        </em>
      );
    }
    // Inline code `code`
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

  if (currentIndex < text.length) {
    parts.push(
      <span key={`text-${key++}`}>
        {text.substring(currentIndex)}
      </span>
    );
  }

  return parts;
};
