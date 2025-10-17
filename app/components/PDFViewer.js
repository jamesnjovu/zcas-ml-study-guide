import React from 'react';
import PropTypes from 'prop-types';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

const PDFViewer = ({ pageNumber, pdfFile, onDocumentLoadSuccess }) => {
  const pdfPath = `/zcas-ml-study-guide/${pdfFile}`;

  return (
    <Document
      file={pdfPath}
      onLoadSuccess={onDocumentLoadSuccess}
      loading={
        <div className="flex items-center justify-center p-8">
          <div className="text-gray-600">Loading PDF...</div>
        </div>
      }
      error={
        <div className="flex items-center justify-center p-8">
          <div className="text-red-600">Failed to load PDF. Please try again.</div>
        </div>
      }
    >
      <Page
        pageNumber={pageNumber}
        width={800}
        loading={
          <div className="flex items-center justify-center p-8">
            <div className="text-gray-600">Loading page {pageNumber}...</div>
          </div>
        }
        renderTextLayer={false}
        renderAnnotationLayer={false}
      />
    </Document>
  );
};

PDFViewer.propTypes = {
  pageNumber: PropTypes.number.isRequired,
  pdfFile: PropTypes.string.isRequired,
  onDocumentLoadSuccess: PropTypes.func
};

export default PDFViewer;
