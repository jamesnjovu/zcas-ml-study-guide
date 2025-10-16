export const units = [
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
