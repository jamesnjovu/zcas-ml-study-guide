# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Next.js-based machine learning study guide application deployed to GitHub Pages. The app provides interactive study materials, quizzes, and past papers for a machine learning course with 21 units covering topics from ML fundamentals to reinforcement learning, plus Python programming basics.

## Development Commands

```bash
# Development
npm run dev              # Start dev server with Turbopack (http://localhost:3000)

# Production Build & Deploy
npm run build            # Create static export in /out directory
npm start                # Start production server
npm run export           # Build for static export
npm run deploy           # Deploy to GitHub Pages (runs predeploy automatically)

# Code Quality
npm run lint             # Run ESLint
```

## Architecture Overview

### Single-Page Application Structure

The app is a client-side only application (`'use client'` at app/page.js:1) that uses **view-based navigation** instead of Next.js routing. All navigation is managed through custom hooks rather than the App Router.

**Navigation Flow:**
- `app/page.js` - Main application component that renders different views based on state
- `app/hooks/useNavigation.js` - Central navigation state management (currentView, selectedUnit, selectedPastPaper)
- Views defined in `app/constants/index.js` (HOME, QUIZ_LIST, UNIT, QUIZ, PAST_PAPERS, PAST_PAPER_DETAIL)
- Switch statement in app/page.js:68-136 determines which view component to render

### State Management Pattern

The app uses two primary custom hooks for global state:

1. **useNavigation** (`app/hooks/useNavigation.js`)
   - Manages current view and selected content (unit/paper)
   - Provides navigation functions (goToHome, goToUnit, goToQuiz, etc.)
   - All navigation state changes happen through these functions

2. **useQuiz** (`app/hooks/useQuiz.js`)
   - Manages quiz answers, submission state, and scores
   - Automatically syncs with localStorage for persistence
   - Handles score calculation and quiz reset logic

### Data Architecture

**Content Data:** Located in `app/data/`
- `units.js` - 21 ML study units with extensive quiz data (over 2700 lines)
- `pastPapers.js` - Past exam papers with questions and answers
- Each unit has: id, title, pages, summary (markdown with LaTeX), keyTakeaways, quiz array
- Quiz questions have: question (supports markdown/LaTeX), options array (supports markdown/LaTeX), correct answer index
- All markdown content supports rich formatting including mathematical notation

**Local Storage:** Managed through `app/utils/localStorage.js`
- Abstraction layer with error handling and availability checks
- Storage keys prefixed with `ml_study_guide_` to avoid conflicts
- Persists: quiz scores, quiz answers (in-progress), completed units, user preferences
- Functions follow pattern: save*/load*/clear* for consistency

### Component Structure

**Reusable UI Components:** `app/components/`
- `Button.js`, `Badge.js`, `Card.js` - Basic UI primitives
- `UnitCard.js` - Displays unit card with markdown-rendered summary preview (3-line clamp)
- `QuizQuestion.js` - Quiz question with markdown support in questions/options, TTS integration
- `PastPaperQuestion.js` - Past paper question with markdown rendering and TTS
- `TextToSpeechControls.js` - TTS controls using Web Speech API
- `PDFViewer.js` - PDF rendering using react-pdf/pdfjs-dist

**View Components:** `app/views/`
- Each view is a separate component rendered by the main app switch
- Views: HomeView, QuizListView, UnitView, QuizView, PastPapersView, PastPaperDetailView
- Views receive props for data and callbacks, don't manage navigation directly
- **QuizView:** Displays one question at a time with progress bar, navigation dots, and Previous/Next buttons. Results shown only after final submission with full review of all answers.

### Special Features

**Text-to-Speech:**
- Custom hook at `app/hooks/useTextToSpeech.js` wraps Web Speech API
- Handles voice selection, rate, pitch, volume
- Used in UnitView, QuizView, and PastPaperDetailView for reading content aloud
- `app/utils/stripMarkdown.js` cleans all markdown/LaTeX before TTS to prevent reading symbols
- Automatically strips: code blocks, inline code, bold/italic markers, headings, links, lists, math notation
- All TTS implementations properly strip markdown for natural speech

**Markdown Rendering:**
- `app/utils/markdownRenderer.js` converts markdown to React elements with LaTeX math support
- **Supported markdown:**
  - Headers: `#`, `##`, `###`, `####`
  - Horizontal rules: `---`, `***`, `___`
  - Lists: `- item` (bulleted lists)
  - Text formatting: `**bold**`, `*italic*`, `` `code` ``
  - Code blocks: ` ```language ... ``` `
- **LaTeX Math Support:**
  - Inline math: `\( expression \)` - renders with Unicode conversion
  - Display math: `\[ expression \]` - renders in centered blue block
  - **Supported LaTeX commands:**
    - Greek letters: `\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\theta`, `\lambda`, `\mu`, `\pi`, `\sigma`, `\tau`, `\phi`, `\omega`, etc.
    - Math operators: `\times`, `\div`, `\pm`, `\leq`, `\geq`, `\neq`, `\approx`, `\sum`, `\prod`, `\int`, `\partial`, `\nabla`, `\infty`
    - Set operators: `\in`, `\subset`, `\cup`, `\cap`, `\forall`, `\exists`, `\emptyset`
    - Arrows: `\rightarrow`, `\Rightarrow`, `\leftarrow`, `\leftrightarrow`
    - Fractions: `\frac{num}{den}` → (num)/(den)
    - Square root: `\sqrt{x}` → √(x)
    - Accents: `\hat{x}` → x̂, `\bar{x}` → x̄, `\tilde{x}` → x̃
    - Subscripts: `_{text}` or `_i` → converts to Unicode subscripts (₀₁₂...ₐₑᵢₙₒₚᵣₛₜᵤᵥₓ)
    - Superscripts: `^{text}` or `^2` → converts to Unicode superscripts (⁰¹²³⁴...ⁿⁱᵀ)
    - Common subscripts: `SS_{res}`, `SS_{tot}`, `_{max}`, `_{min}`, `_{train}`, `_{test}`
- Used throughout: unit summaries, quiz questions/options, past paper content, key takeaways

**PDF Viewing:**
- `app/components/PDFViewer.js` renders PDF files from public/ directory
- Uses react-pdf with pdfjs-dist worker configuration
- PDF files: lecture_notes.pdf, beginners_python_cheat_sheet_pcc_all.pdf, 2023/2024_past_paper.pdf

## GitHub Pages Deployment

**Configuration in next.config.mjs:**
- `output: 'export'` - Static site generation
- `basePath: '/zcas-ml-study-guide'` - GitHub Pages repository path
- `images: { unoptimized: true }` - Required for static export

**Deployment files:**
- `.nojekyll` file in root prevents Jekyll processing
- `out/` directory created during build contains static assets
- Deploy command uses gh-pages package to push to gh-pages branch

## Key Technical Decisions

1. **View-based navigation** instead of Next.js routing - entire app is single page, all navigation state is React state
2. **All components in app/** directory - not using traditional Next.js pages structure
3. **Extensive data embedded in JavaScript** - units.js is 2700+ lines with all content
4. **localStorage for persistence** - no backend, all data stored client-side
5. **Client-side only** - 'use client' directive on main page, no SSR/SSG per route
6. **Static export to GitHub Pages** - free hosting with basePath configuration

## Common Modifications

**Adding a new unit:**
1. Add unit object to `app/data/units.js` array
2. Include: id, title, pages, summary (markdown with LaTeX support), keyTakeaways array, quiz array
3. Each quiz question needs: question (can use markdown/LaTeX like `### Heading` or `\( math \)`), options (array, also supports markdown), correct (index)
4. Unit automatically appears in HomeView and QuizListView
5. **Markdown/LaTeX tips:**
   - Use `###` for question headings
   - Use `**bold**` and `*italic*` for emphasis
   - Use `` `code` `` for technical terms
   - Use `\( ... \)` for inline math like `\( \beta_0 \)`
   - Use `\[ ... \]` for display math blocks
   - Use `_{subscript}` and `^{superscript}` for notation

**Adding a new view:**
1. Add constant to `app/constants/index.js`
2. Create view component in `app/views/`
3. Add navigation function to `app/hooks/useNavigation.js`
4. Add case to switch statement in `app/page.js`

**Modifying localStorage schema:**
1. Update `STORAGE_KEYS` in `app/utils/localStorage.js`
2. Add corresponding save/load functions
3. Update related hooks (useQuiz, etc.) to use new functions

**Extending LaTeX support:**
1. Edit `app/utils/markdownRenderer.js` - `convertLatexToUnicode()` function
2. Add new LaTeX commands to the `replacements` object or regex handlers
3. For complex commands (like `\frac`, `\sqrt`), add regex-based replacements before the simple replacements
4. Test with both inline `\( \)` and display `\[ \]` math modes
5. Don't forget to update `app/utils/stripMarkdown.js` if needed for TTS

## Dependencies Notes

- **Next.js 15.5.5** with Turbopack dev mode
- **React 19.1.0** - uses new features like `use client` directive
- **Tailwind CSS 4.0** - styling framework with @tailwindcss/postcss
- **react-pdf 10.2.0 + pdfjs-dist 5.4.296** - PDF rendering
- **lucide-react** - Icon library
- **gh-pages** - Deployment to GitHub Pages
