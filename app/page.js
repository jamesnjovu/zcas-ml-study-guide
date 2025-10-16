'use client';

import React, { useState } from 'react';
import { ChevronRight, BookOpen, FileText, CheckCircle, ArrowRight, Award, X, Home, List } from 'lucide-react';

const Home = () => {
  const [currentView, setCurrentView] = useState('home');
  const [selectedUnit, setSelectedUnit] = useState(null);
  const [completedUnits, setCompletedUnits] = useState(new Set());
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [quizScores, setQuizScores] = useState({});

  const units = [
    {
      id: 1,
      title: "Introduction to Machine Learning",
      pages: "1-31",
      summary: "This unit introduces the foundations of machine learning, covering key concepts like data types, features, and labels. It explores the main ML paradigms including supervised, unsupervised, and reinforcement learning.",
      keyTakeaways: [
        "ML enables computers to learn from data without explicit programming",
        "Quality and quantity of data are crucial for ML success",
        "Different learning paradigms suit different problem types",
        "Ethical considerations and bias mitigation are essential"
      ],
      quiz: [
        { question: "What is the primary goal of machine learning?", options: ["To program computers explicitly", "To enable computers to learn from data", "To replace human intelligence", "To store data"], correct: 1 },
        { question: "Which is NOT a type of machine learning?", options: ["Supervised Learning", "Unsupervised Learning", "Manual Learning", "Reinforcement Learning"], correct: 2 },
        { question: "What are features in machine learning?", options: ["Output labels", "Input characteristics or attributes", "Training algorithms", "Cost functions"], correct: 1 },
        { question: "What does training data include in supervised learning?", options: ["Only features", "Only labels", "Both features and labels", "Neither"], correct: 2 },
        { question: "What is overfitting?", options: ["Good performance on all data", "Memorizing training data instead of learning", "Model too simple", "Training time too short"], correct: 1 },
        { question: "Common metric for classification tasks?", options: ["Mean Squared Error", "RMSE", "Accuracy and F1-score", "Standard Deviation"], correct: 2 },
        { question: "Role of validation data?", options: ["Train the model", "Evaluate and detect overfitting", "Store results", "Increase dataset size"], correct: 1 },
        { question: "Key ethical concern in ML?", options: ["Processing speed", "Bias and fairness", "Memory usage", "Code complexity"], correct: 1 },
        { question: "What is concept learning?", options: ["Learning programming", "Acquiring and generalizing concepts from examples", "Learning ML tools", "Memorizing algorithms"], correct: 1 },
        { question: "What distinguishes deep learning?", options: ["Uses less data", "Neural networks with multiple layers", "Faster training", "No computers needed"], correct: 1 }
      ]
    },
    {
      id: 2,
      title: "Supervised Learning & Regression",
      pages: "91-104",
      summary: "This unit covers supervised learning with focus on regression techniques. It starts with simple linear regression and progresses to multiple and polynomial regression for handling non-linear relationships.",
      keyTakeaways: [
        "Supervised learning uses labeled data for training",
        "Linear regression models relationships with y = mx + c",
        "Polynomial regression handles non-linear patterns",
        "LWLR adapts to local data patterns for better accuracy"
      ],
      quiz: [
        { question: "Basic equation for simple linear regression?", options: ["y = mx + b", "y = x²", "y = log(x)", "y = 1/x"], correct: 0 },
        { question: "What type of variable does regression predict?", options: ["Categorical", "Binary", "Continuous", "Discrete only"], correct: 2 },
        { question: "In multiple linear regression, 'multiple' refers to?", options: ["Multiple outputs", "Multiple independent variables", "Multiple datasets", "Multiple algorithms"], correct: 1 },
        { question: "When to use polynomial regression?", options: ["Data is linear", "Data shows non-linear patterns", "Always", "Never"], correct: 1 },
        { question: "Main advantage of LWLR?", options: ["Faster training", "Adapts to local patterns", "Uses less memory", "Requires no data"], correct: 1 },
        { question: "What is a residual?", options: ["The slope", "The intercept", "Difference between actual and predicted", "Training time"], correct: 2 },
        { question: "Best-fit line minimizes?", options: ["Number of features", "Sum of squared errors", "Training time", "Data points"], correct: 1 },
        { question: "In y = mx + c, 'm' represents?", options: ["Mean", "Slope", "Median", "Mode"], correct: 1 },
        { question: "What is the dependent variable?", options: ["Input features", "Variable being predicted", "Training data", "Algorithm"], correct: 1 },
        { question: "Key disadvantage of supervised learning?", options: ["Too fast", "Requires labeled data", "Too simple", "Works with all data"], correct: 1 }
      ]
    },
    {
      id: 3,
      title: "Optimization Techniques",
      pages: "167-187",
      summary: "This unit focuses on optimization algorithms used to train machine learning models. It covers gradient descent methods, explaining how models minimize error through iterative parameter updates.",
      keyTakeaways: [
        "Optimization minimizes the cost/error function",
        "Gradient descent updates parameters iteratively",
        "Learning rate controls step size in optimization",
        "Different optimization methods have speed-accuracy tradeoffs"
      ],
      quiz: [
        { question: "What is the purpose of optimization in ML?", options: ["Increase data size", "Minimize cost function", "Add more features", "Speed up prediction"], correct: 1 },
        { question: "Gradient descent moves in which direction?", options: ["Positive gradient", "Negative gradient", "Random direction", "No movement"], correct: 1 },
        { question: "What does learning rate control?", options: ["Model complexity", "Step size in optimization", "Data size", "Number of features"], correct: 1 },
        { question: "What is a cost function?", options: ["Training time", "Measurement of error", "Number of parameters", "Data quality"], correct: 1 },
        { question: "Batch gradient descent uses?", options: ["One sample at a time", "All training data per iteration", "Random samples", "No data"], correct: 1 },
        { question: "Stochastic gradient descent is?", options: ["Slower but more accurate", "Faster but noisier", "Same as batch", "Never used"], correct: 1 },
        { question: "High learning rate can cause?", options: ["Slow convergence", "Overshooting minimum", "Better accuracy", "No effect"], correct: 1 },
        { question: "Goal of gradient descent?", options: ["Maximize error", "Find local/global minimum", "Increase parameters", "Add noise"], correct: 1 },
        { question: "What indicates convergence?", options: ["Error increasing", "Small change in parameters", "More iterations", "Random values"], correct: 1 },
        { question: "Cost function should be?", options: ["Maximized", "Minimized", "Ignored", "Constant"], correct: 1 }
      ]
    },
    {
      id: 4,
      title: "Kernel Methods & SVMs",
      pages: "188-196",
      summary: "This unit introduces kernel methods and Support Vector Machines for classification. It explains how kernels transform data into higher dimensions to make it separable.",
      keyTakeaways: [
        "Kernels map data to higher-dimensional spaces",
        "SVMs find optimal hyperplane for classification",
        "Support vectors are critical data points near boundaries",
        "Different kernels suit different data patterns"
      ],
      quiz: [
        { question: "What do kernel functions do?", options: ["Delete data", "Transform data to higher dimensions", "Reduce dimensions", "Label data"], correct: 1 },
        { question: "What is a hyperplane in SVM?", options: ["Training data", "Decision boundary", "Kernel function", "Cost function"], correct: 1 },
        { question: "Support vectors are?", options: ["All data points", "Points far from boundary", "Points closest to decision boundary", "Labels"], correct: 2 },
        { question: "SVM goal is to?", options: ["Minimize margin", "Maximize margin", "Delete data", "Add noise"], correct: 1 },
        { question: "Which is a kernel type?", options: ["Square kernel", "RBF (Gaussian) kernel", "Triangle kernel", "Circle kernel"], correct: 1 },
        { question: "Linear SVM is used for?", options: ["Non-separable data", "Linearly separable data", "All data types", "No data"], correct: 1 },
        { question: "Non-linear SVM uses?", options: ["Only linear kernels", "Kernel trick", "No kernels", "Simple lines"], correct: 1 },
        { question: "Margin in SVM is?", options: ["Training time", "Distance between classes", "Number of features", "Error rate"], correct: 1 },
        { question: "SVM works well for?", options: ["Only small datasets", "High-dimensional spaces", "Only images", "Time series only"], correct: 1 },
        { question: "What does polynomial kernel add?", options: ["Linear complexity", "Polynomial features", "More data", "Less accuracy"], correct: 1 }
      ]
    },
    {
      id: 5,
      title: "Decision Trees & Ensembles",
      pages: "197-228",
      summary: "This unit covers tree-based learning methods and ensemble techniques. Decision trees make decisions through hierarchical splitting. Ensemble methods like boosting and random forests combine multiple models.",
      keyTakeaways: [
        "Decision trees use if-then rules for classification",
        "Information gain and Gini index guide splitting",
        "Pruning prevents overfitting in trees",
        "Random forests reduce variance through averaging"
      ],
      quiz: [
        { question: "Decision tree starts with?", options: ["Leaf node", "Root node", "Branch", "Random node"], correct: 1 },
        { question: "What is a leaf node?", options: ["Starting point", "Final output node", "Middle node", "Root"], correct: 1 },
        { question: "Information gain measures?", options: ["Tree height", "Reduction in entropy", "Number of nodes", "Training time"], correct: 1 },
        { question: "What is pruning?", options: ["Adding branches", "Removing unnecessary branches", "Training faster", "Adding data"], correct: 1 },
        { question: "Gini index measures?", options: ["Tree size", "Impurity", "Accuracy", "Speed"], correct: 1 },
        { question: "Random forest combines?", options: ["One tree", "Multiple decision trees", "Linear models", "Kernels"], correct: 1 },
        { question: "Boosting focuses on?", options: ["Easy examples", "Misclassified examples", "All examples equally", "Random examples"], correct: 1 },
        { question: "Main advantage of random forest?", options: ["Faster than single tree", "Reduces overfitting", "Simpler", "Uses less memory"], correct: 1 },
        { question: "Decision tree splitting criterion?", options: ["Random", "Maximizes information gain", "Minimizes data", "Adds noise"], correct: 1 },
        { question: "Ensemble learning combines?", options: ["Different datasets", "Multiple models", "One model", "No models"], correct: 1 }
      ]
    }
  ];

  const startQuiz = () => {
    setQuizAnswers({});
    setQuizSubmitted(false);
    setCurrentView('quiz');
  };

  const selectAnswer = (questionIndex, optionIndex) => {
    if (!quizSubmitted) {
      setQuizAnswers({...quizAnswers, [questionIndex]: optionIndex});
    }
  };

  const submitQuiz = () => {
    setQuizSubmitted(true);
    const correct = selectedUnit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
    const newScores = {...quizScores, [selectedUnit.id]: {score: correct, total: selectedUnit.quiz.length}};
    setQuizScores(newScores);
  };

  const nextUnit = () => {
    const currentIndex = units.findIndex(u => u.id === selectedUnit.id);
    if (currentIndex < units.length - 1) {
      setSelectedUnit(units[currentIndex + 1]);
      setCurrentView('unit');
      setQuizAnswers({});
      setQuizSubmitted(false);
    }
  };

  const goToHome = () => {
    setCurrentView('home');
    setSelectedUnit(null);
  };

  const selectUnit = (unit) => {
    setSelectedUnit(unit);
    setCurrentView('unit');
    setQuizAnswers({});
    setQuizSubmitted(false);
  };

  // Home View
  if (currentView === 'home') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
        <div className="max-w-6xl mx-auto">
          <header className="text-center mb-12">
            <h1 className="text-5xl font-bold text-gray-800 mb-4">Machine Learning Study Guide</h1>
            <p className="text-xl text-gray-600">Master ML concepts through structured learning units</p>
            <div className="mt-6 flex gap-4 justify-center">
              <button
                onClick={() => setCurrentView('quizList')}
                className="bg-purple-500 text-white px-6 py-3 rounded-lg hover:bg-purple-600 flex items-center gap-2"
              >
                <List className="w-5 h-5" />
                View All Quizzes
              </button>
            </div>
          </header>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {units.map((unit) => (
              <div
                key={unit.id}
                onClick={() => selectUnit(unit)}
                className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all cursor-pointer border-2 border-gray-200 hover:border-blue-400"
              >
                <div className="flex items-center justify-between mb-4">
                  <span className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                    Unit {unit.id}
                  </span>
                  {quizScores[unit.id] && (
                    <Award className="w-6 h-6 text-yellow-500" />
                  )}
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">{unit.title}</h3>
                <p className="text-sm text-gray-600 mb-4">Pages {unit.pages}</p>
                {quizScores[unit.id] && (
                  <div className="bg-green-50 px-3 py-2 rounded-lg">
                    <span className="text-sm font-semibold text-green-700">
                      Quiz Score: {quizScores[unit.id].score}/{quizScores[unit.id].total}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Quiz List View
  if (currentView === 'quizList') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-4xl font-bold text-gray-800">All Quizzes</h1>
            <button
              onClick={goToHome}
              className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 flex items-center gap-2"
            >
              <Home className="w-5 h-5" />
              Home
            </button>
          </div>

          <div className="space-y-4">
            {units.map((unit) => (
              <div
                key={unit.id}
                className="bg-white rounded-lg p-6 shadow-lg border-2 border-gray-200"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="bg-purple-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                        Unit {unit.id}
                      </span>
                      {quizScores[unit.id] && (
                        <div className="flex items-center gap-2 bg-green-100 px-3 py-1 rounded-full">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          <span className="text-sm font-semibold text-green-700">
                            {quizScores[unit.id].score}/{quizScores[unit.id].total}
                          </span>
                        </div>
                      )}
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 mb-1">{unit.title}</h3>
                    <p className="text-sm text-gray-600">10 Questions</p>
                  </div>
                  <button
                    onClick={() => selectUnit(unit)}
                    className="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600 flex items-center gap-2"
                  >
                    {quizScores[unit.id] ? 'Retake' : 'Start'}
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Unit Content View
  if (currentView === 'unit') {
    const currentIndex = units.findIndex(u => u.id === selectedUnit.id);
    const hasNextUnit = currentIndex < units.length - 1;

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <button
              onClick={goToHome}
              className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 flex items-center gap-2"
            >
              <Home className="w-5 h-5" />
              Home
            </button>
            <span className="bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold">
              Unit {selectedUnit.id} of {units.length}
            </span>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
            <div className="mb-6">
              <span className="inline-block bg-blue-500 text-white px-4 py-2 rounded-full text-sm font-bold mb-4">
                Unit {selectedUnit.id} • Pages {selectedUnit.pages}
              </span>
              <h2 className="text-3xl font-bold text-gray-800 mb-4">{selectedUnit.title}</h2>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-gray-700 mb-3 flex items-center">
                  <FileText className="w-6 h-6 mr-2 text-blue-500" />
                  Summary
                </h3>
                <p className="text-gray-700 leading-relaxed bg-blue-50 p-4 rounded-lg">
                  {selectedUnit.summary}
                </p>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-gray-700 mb-4">Key Takeaways</h3>
                <div className="space-y-3">
                  {selectedUnit.keyTakeaways.map((takeaway, index) => (
                    <div key={index} className="flex items-start bg-purple-50 p-4 rounded-lg">
                      <span className="inline-block bg-purple-500 text-white text-sm font-bold rounded-full w-7 h-7 flex items-center justify-center mr-3 flex-shrink-0">
                        {index + 1}
                      </span>
                      <span className="text-gray-700">{takeaway}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg border-l-4 border-blue-500">
                <p className="text-sm text-gray-600">
                  <span className="font-semibold">PDF Reference:</span> Pages{' '}
                  <span className="font-mono bg-white px-2 py-1 rounded">{selectedUnit.pages}</span>
                </p>
              </div>
            </div>
          </div>

          <div className="flex gap-4">
            <button
              onClick={startQuiz}
              className="flex-1 bg-green-500 text-white px-6 py-4 rounded-lg hover:bg-green-600 font-semibold text-lg flex items-center justify-center gap-2"
            >
              <Award className="w-6 h-6" />
              Take Quiz
            </button>
            {hasNextUnit && (
              <button
                onClick={nextUnit}
                className="flex-1 bg-blue-500 text-white px-6 py-4 rounded-lg hover:bg-blue-600 font-semibold text-lg flex items-center justify-center gap-2"
              >
                Next Unit
                <ArrowRight className="w-6 h-6" />
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Quiz View
  if (currentView === 'quiz') {
    const allAnswered = Object.keys(quizAnswers).length === selectedUnit.quiz.length;
    const score = quizSubmitted ? selectedUnit.quiz.filter((q, i) => quizAnswers[i] === q.correct).length : 0;

    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <button
              onClick={() => setCurrentView('unit')}
              className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 flex items-center gap-2"
            >
              <X className="w-5 h-5" />
              Back to Unit
            </button>
            <span className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-semibold">
              Unit {selectedUnit.id} Quiz
            </span>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">{selectedUnit.title} - Quiz</h2>

            {!quizSubmitted && (
              <div className="mb-6 bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-gray-700">
                  Progress: {Object.keys(quizAnswers).length}/{selectedUnit.quiz.length} questions answered
                </p>
              </div>
            )}

            {quizSubmitted && (
              <div className="mb-6 bg-green-50 p-6 rounded-lg border-2 border-green-300">
                <div className="flex items-center gap-3 mb-2">
                  <Award className="w-8 h-8 text-yellow-500" />
                  <h3 className="text-2xl font-bold text-gray-800">Quiz Complete!</h3>
                </div>
                <p className="text-xl text-gray-700">
                  Your Score: <span className="font-bold text-green-600">{score}/{selectedUnit.quiz.length}</span>
                  {' '}({Math.round((score/selectedUnit.quiz.length) * 100)}%)
                </p>
              </div>
            )}

            <div className="space-y-6">
              {selectedUnit.quiz.map((question, qIndex) => (
                <div key={qIndex} className="border-2 border-gray-200 rounded-lg p-6">
                  <h4 className="font-semibold text-gray-800 mb-4">
                    {qIndex + 1}. {question.question}
                  </h4>
                  <div className="space-y-2">
                    {question.options.map((option, oIndex) => {
                      const isSelected = quizAnswers[qIndex] === oIndex;
                      const isCorrect = question.correct === oIndex;
                      const showResult = quizSubmitted;

                      let bgColor = 'bg-gray-50 hover:bg-gray-100';
                      if (showResult) {
                        if (isCorrect) bgColor = 'bg-green-100 border-green-500';
                        else if (isSelected && !isCorrect) bgColor = 'bg-red-100 border-red-500';
                      } else if (isSelected) {
                        bgColor = 'bg-blue-100 border-blue-500';
                      }

                      return (
                        <button
                          key={oIndex}
                          onClick={() => selectAnswer(qIndex, oIndex)}
                          disabled={quizSubmitted}
                          className={`w-full text-left p-4 rounded-lg border-2 transition-all ${bgColor} ${
                            quizSubmitted ? 'cursor-default' : 'cursor-pointer'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span>{option}</span>
                            {showResult && isCorrect && (
                              <CheckCircle className="w-5 h-5 text-green-600" />
                            )}
                            {showResult && isSelected && !isCorrect && (
                              <X className="w-5 h-5 text-red-600" />
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            {!quizSubmitted && (
              <button
                onClick={submitQuiz}
                disabled={!allAnswered}
                className={`w-full mt-8 px-6 py-4 rounded-lg font-semibold text-lg ${
                  allAnswered
                    ? 'bg-green-500 text-white hover:bg-green-600'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Submit Quiz
              </button>
            )}

            {quizSubmitted && (
              <div className="flex gap-4 mt-8">
                <button
                  onClick={() => setCurrentView('unit')}
                  className="flex-1 bg-blue-500 text-white px-6 py-4 rounded-lg hover:bg-blue-600 font-semibold"
                >
                  Back to Unit
                </button>
                <button
                  onClick={goToHome}
                  className="flex-1 bg-gray-500 text-white px-6 py-4 rounded-lg hover:bg-gray-600 font-semibold"
                >
                  Home
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }
};

export default Home;