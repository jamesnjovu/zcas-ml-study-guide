export const units = [
  {
    id: 1,
    title: "Introduction to Machine Learning",
    pages: "1-31",
    summary: "This unit introduces the foundations of machine learning, covering key concepts like data types, features, and labels. It explores the main ML paradigms including supervised, unsupervised, and reinforcement learning, along with perspectives and ethical issues.",
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
      { question: "Instance-based learning stores?", options: ["Only rules", "Specific instances in memory", "Nothing", "Only formulas"], correct: 1 }
    ]
  },
  {
    id: 2,
    title: "Related Areas & Applications of ML",
    pages: "44-90",
    summary: "This unit explores related areas of machine learning including deep learning, NLP, computer vision, and reinforcement learning. It covers diverse applications across industries and introduces popular ML software tools and frameworks.",
    keyTakeaways: [
      "Deep learning uses neural networks with multiple layers",
      "NLP enables computers to understand human language",
      "Computer vision interprets visual information from images/videos",
      "ML has wide-ranging applications from healthcare to autonomous vehicles"
    ],
    quiz: [
      { question: "Deep learning is a subset of?", options: ["Data Science", "Machine Learning", "Statistics", "Programming"], correct: 1 },
      { question: "Which library is NOT mentioned for Python ML?", options: ["TensorFlow", "PyTorch", "scikit-learn", "MATLAB"], correct: 3 },
      { question: "NLP stands for?", options: ["Natural Language Processing", "New Learning Protocol", "Neural Layer Processing", "Numeric Logic Programming"], correct: 0 },
      { question: "Computer vision deals with?", options: ["Audio data", "Text data", "Images and videos", "Time series"], correct: 2 },
      { question: "Which is an ML application in healthcare?", options: ["Gaming", "Medical diagnostics", "Social media", "Shopping"], correct: 1 },
      { question: "Transfer learning involves?", options: ["Creating new models", "Transferring knowledge between tasks", "Deleting old models", "Random training"], correct: 1 },
      { question: "Keras runs on top of?", options: ["PyTorch", "TensorFlow", "R", "Excel"], correct: 1 },
      { question: "Which area uses sentiment analysis?", options: ["Computer Vision", "NLP", "Robotics", "Gaming"], correct: 1 },
      { question: "Reinforcement learning agents learn through?", options: ["Labeled data", "Trial and error", "Memorization", "Random selection"], correct: 1 },
      { question: "Which is used for data visualization?", options: ["NumPy", "Matplotlib", "TensorFlow", "Keras"], correct: 1 }
    ]
  },
  {
    id: 3,
    title: "Supervised Learning & Classification",
    pages: "91-104",
    summary: "This unit covers supervised learning fundamentals, including both regression and classification techniques. It explains how models are trained using labeled data and covers algorithms like decision trees, logistic regression, and SVM.",
    keyTakeaways: [
      "Supervised learning uses well-labeled training data",
      "Regression predicts continuous variables",
      "Classification predicts categorical variables",
      "Training involves multiple steps from data collection to evaluation"
    ],
    quiz: [
      { question: "Supervised learning requires?", options: ["Unlabeled data", "Labeled data", "No data", "Random data"], correct: 1 },
      { question: "Classification is used when output is?", options: ["Continuous", "Categorical", "Infinite", "Unknown"], correct: 1 },
      { question: "Which is a classification algorithm?", options: ["Linear Regression", "Random Forest", "K-means", "PCA"], correct: 1 },
      { question: "First step in supervised learning?", options: ["Test model", "Determine training dataset type", "Deploy model", "Collect money"], correct: 1 },
      { question: "Logistic regression is used for?", options: ["Regression", "Classification", "Clustering", "Dimension reduction"], correct: 1 },
      { question: "What splits data in supervised learning?", options: ["Training, test, validation sets", "Only training", "Only test", "No split needed"], correct: 0 },
      { question: "Support Vector Machines are for?", options: ["Regression only", "Classification", "Clustering only", "Visualization"], correct: 1 },
      { question: "A disadvantage of supervised learning?", options: ["Too simple", "Requires labeled data", "Too fast", "No accuracy"], correct: 1 },
      { question: "Regression Trees predict?", options: ["Categories", "Continuous values", "Nothing", "Only integers"], correct: 1 },
      { question: "Model accuracy is evaluated on?", options: ["Training set", "Test set", "All data", "No data"], correct: 1 }
    ]
  },
  {
    id: 4,
    title: "Linear & Polynomial Regression",
    pages: "105-166",
    summary: "This unit provides deep coverage of regression techniques from simple linear regression to polynomial regression and locally weighted linear regression (LWLR). It includes equation derivations, residual analysis, and handling non-linear relationships.",
    keyTakeaways: [
      "Simple linear regression uses y = mx + c equation",
      "Multiple linear regression handles multiple independent variables",
      "Polynomial regression captures non-linear patterns",
      "LWLR adapts to local data patterns dynamically"
    ],
    quiz: [
      { question: "In y = mx + c, what is 'c'?", options: ["Slope", "Intercept", "Error", "Prediction"], correct: 1 },
      { question: "Dependent variable is denoted by?", options: ["x", "y", "m", "c"], correct: 1 },
      { question: "Multiple linear regression has multiple?", options: ["Outputs", "Independent variables", "Errors", "Models"], correct: 1 },
      { question: "Best-fit line has lowest?", options: ["Slope", "Intercept", "Sum of squared errors", "Number of points"], correct: 2 },
      { question: "Polynomial regression degree refers to?", options: ["Temperature", "Highest power of variables", "Number of samples", "Time"], correct: 1 },
      { question: "When should polynomial regression be used?", options: ["Always", "For non-linear data", "For linear data only", "Never"], correct: 1 },
      { question: "LWLR assigns weights based on?", options: ["Random selection", "Proximity to target point", "Data size", "Color"], correct: 1 },
      { question: "Residual is the difference between?", options: ["x and y", "Actual and predicted values", "Training and test", "Mean and median"], correct: 1 },
      { question: "Polynomial regression can cause?", options: ["Underfitting only", "Overfitting with high degrees", "No issues", "Faster training"], correct: 1 },
      { question: "LWLR is useful for?", options: ["Time series", "Static patterns", "Locally varying patterns", "Constant data"], correct: 2 }
    ]
  },
  {
    id: 5,
    title: "Numerical Optimization & Gradient Descent",
    pages: "167-187",
    summary: "This unit focuses on numerical optimization techniques essential for training ML models. It covers gradient descent variants including batch, stochastic, and mini-batch, explaining cost functions, learning rates, and convergence.",
    keyTakeaways: [
      "Optimization finds best parameters by minimizing cost function",
      "Gradient descent moves opposite to gradient direction",
      "Learning rate controls optimization step size",
      "Different GD variants offer speed vs accuracy tradeoffs"
    ],
    quiz: [
      { question: "Gradient descent minimizes?", options: ["Features", "Cost function", "Data size", "Time"], correct: 1 },
      { question: "Learning rate controls?", options: ["Data quality", "Step size", "Number of features", "Model type"], correct: 1 },
      { question: "Batch GD uses how many samples per update?", options: ["One", "Random", "All training data", "None"], correct: 2 },
      { question: "Stochastic GD uses how many samples?", options: ["All", "One at a time", "Half", "None"], correct: 1 },
      { question: "High learning rate causes?", options: ["Slow convergence", "Overshooting", "No effect", "Better accuracy"], correct: 1 },
      { question: "Cost function measures?", options: ["Speed", "Error/difference", "Data size", "Features"], correct: 1 },
      { question: "Gradient points toward?", options: ["Minimum", "Maximum", "Random direction", "Origin"], correct: 1 },
      { question: "Convergence means?", options: ["Starting training", "Reaching optimal point", "Adding data", "Deleting model"], correct: 1 },
      { question: "SGD advantage over batch GD?", options: ["Always better", "Faster updates", "Always accurate", "Uses more memory"], correct: 1 },
      { question: "Hypothesis in ML represents?", options: ["Data", "Model's prediction function", "Error", "Learning rate"], correct: 1 }
    ]
  },
  {
    id: 6,
    title: "Kernel Methods & Support Vector Machines",
    pages: "188-308",
    summary: "This unit covers kernel methods and Support Vector Machines in depth. It explains kernel functions (linear, polynomial, RBF, Laplace), the kernel trick, hyperplanes, margins, and how SVMs handle both linearly and non-linearly separable data.",
    keyTakeaways: [
      "Kernels transform data to higher-dimensional spaces",
      "SVMs find optimal hyperplane maximizing margin",
      "Support vectors are critical boundary points",
      "Different kernels (linear, polynomial, RBF) suit different patterns"
    ],
    quiz: [
      { question: "Kernel function must satisfy?", options: ["Mercer's condition", "Newton's law", "No condition", "Random properties"], correct: 0 },
      { question: "Linear kernel is?", options: ["Dot product", "Exponential", "Logarithmic", "Square root"], correct: 0 },
      { question: "RBF kernel is also called?", options: ["Linear", "Polynomial", "Gaussian kernel", "Square kernel"], correct: 2 },
      { question: "Hyperplane in SVM is?", options: ["Data point", "Decision boundary", "Kernel", "Training set"], correct: 1 },
      { question: "Margin in SVM is?", options: ["Error rate", "Distance between support vectors and hyperplane", "Number of features", "Training time"], correct: 1 },
      { question: "SVM goal?", options: ["Minimize margin", "Maximize margin", "Ignore margin", "Delete data"], correct: 1 },
      { question: "Non-linear SVM uses?", options: ["No kernels", "Kernel trick", "Only lines", "Circles"], correct: 1 },
      { question: "Support vectors are?", options: ["All points", "Points closest to boundary", "Far points", "Random points"], correct: 1 },
      { question: "Polynomial kernel degree controls?", options: ["Speed", "Complexity of decision boundary", "Data size", "Color"], correct: 1 },
      { question: "SVM works well for?", options: ["Only low dimensions", "High-dimensional data", "No data", "Only text"], correct: 1 }
    ]
  },
  {
    id: 7,
    title: "Decision Trees",
    pages: "197-212",
    summary: "This unit explains decision tree algorithms for classification and regression. It covers tree components (root, leaf, branch nodes), splitting criteria (information gain, Gini index, entropy), pruning techniques, and decision tree advantages and limitations.",
    keyTakeaways: [
      "Decision trees use hierarchical if-then-else rules",
      "Information gain and Gini index guide optimal splits",
      "Pruning removes unnecessary branches to prevent overfitting",
      "Trees are interpretable but can overfit without regularization"
    ],
    quiz: [
      { question: "Root node represents?", options: ["Output", "Entire dataset", "Error", "Leaf"], correct: 1 },
      { question: "Leaf node is?", options: ["Starting point", "Final decision/output", "Middle node", "Root"], correct: 1 },
      { question: "Splitting divides node based on?", options: ["Random choice", "Attribute values", "Time", "Color"], correct: 1 },
      { question: "Information gain measures?", options: ["Time", "Reduction in entropy/uncertainty", "Size", "Speed"], correct: 1 },
      { question: "Entropy measures?", options: ["Speed", "Randomness/impurity", "Size", "Color"], correct: 1 },
      { question: "Gini index measures?", options: ["Speed", "Impurity", "Time", "Size"], correct: 1 },
      { question: "Pruning helps to?", options: ["Grow tree", "Prevent overfitting", "Add noise", "Delete data"], correct: 1 },
      { question: "Decision tree advantage?", options: ["Always accurate", "Easy to understand", "Never overfits", "Requires no data"], correct: 1 },
      { question: "ASM stands for?", options: ["Automatic System Model", "Attribute Selection Measure", "Advanced Statistical Method", "Applied Science Mathematics"], correct: 1 },
      { question: "CART algorithm uses?", options: ["Information gain", "Gini index", "Random selection", "No criterion"], correct: 1 }
    ]
  },
  {
    id: 8,
    title: "K-Nearest Neighbors",
    pages: "213-228",
    summary: "This unit covers the K-Nearest Neighbors algorithm, a non-parametric, lazy learning method for classification and regression. It explains the algorithm workflow, distance metrics (Euclidean), choosing K value, and KNN advantages and disadvantages.",
    keyTakeaways: [
      "KNN classifies based on similarity to K nearest neighbors",
      "Uses distance metrics like Euclidean distance",
      "Non-parametric lazy learner stores all training data",
      "Simple to implement but computationally expensive for large datasets"
    ],
    quiz: [
      { question: "KNN is a type of?", options: ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "No learning"], correct: 0 },
      { question: "KNN is called lazy learner because?", options: ["It's slow", "Doesn't learn during training", "Always sleeps", "Uses no data"], correct: 1 },
      { question: "K in KNN represents?", options: ["Constant", "Number of neighbors", "Kernel", "Key"], correct: 1 },
      { question: "Common distance metric in KNN?", options: ["Manhattan", "Euclidean", "Cosine", "All of these"], correct: 3 },
      { question: "KNN classification decision based on?", options: ["Random", "Majority vote of K neighbors", "First neighbor", "Last neighbor"], correct: 1 },
      { question: "KNN can be used for?", options: ["Classification only", "Regression only", "Both classification and regression", "Neither"], correct: 2 },
      { question: "Choosing K value?", options: ["Always 1", "Should be odd for binary classification", "Always 100", "Doesn't matter"], correct: 1 },
      { question: "KNN disadvantage?", options: ["Too simple", "Computationally expensive for large datasets", "Too fast", "No accuracy"], correct: 1 },
      { question: "KNN stores?", options: ["Nothing", "All training data", "Only K points", "Only errors"], correct: 1 },
      { question: "Small K value leads to?", options: ["Underfitting", "Overfitting/noise sensitivity", "No effect", "Better accuracy"], correct: 1 }
    ]
  },
  {
    id: 9,
    title: "Boosting Algorithms",
    pages: "229-248",
    summary: "This unit explains boosting ensemble techniques that convert weak learners into strong learners. It covers AdaBoost, gradient boosting, the sequential training process, and applications. It also discusses boosting benefits, challenges, and real-world uses.",
    keyTakeaways: [
      "Boosting combines weak learners into strong classifier",
      "Sequential training focuses on misclassified examples",
      "AdaBoost adjusts weights, Gradient Boosting minimizes gradients",
      "Effective but can overfit and sensitive to outliers"
    ],
    quiz: [
      { question: "Boosting is a type of?", options: ["Single model", "Ensemble learning", "Clustering", "Dimension reduction"], correct: 1 },
      { question: "Boosting converts?", options: ["Strong to weak", "Weak learners to strong", "Data to features", "Features to data"], correct: 1 },
      { question: "Boosting trains models?", options: ["In parallel", "Sequentially", "Randomly", "Never"], correct: 1 },
      { question: "Boosting focuses on?", options: ["Easy examples", "Misclassified examples", "Random examples", "All equally"], correct: 1 },
      { question: "AdaBoost adjusts?", options: ["Learning rate", "Weights of misclassified samples", "Number of trees", "Data size"], correct: 1 },
      { question: "Gradient boosting minimizes?", options: ["Time", "Loss function gradients", "Memory", "Features"], correct: 1 },
      { question: "Boosting advantage?", options: ["Always fast", "Reduces bias and improves accuracy", "Needs no data", "Never overfits"], correct: 1 },
      { question: "Boosting challenge?", options: ["Too simple", "Sensitive to outliers", "Too fast", "No parameters"], correct: 1 },
      { question: "Boosting used in?", options: ["Only images", "Healthcare, IT, finance", "Only text", "Nothing"], correct: 1 },
      { question: "Weak learner is?", options: ["Complex model", "Simple model slightly better than random", "Perfect model", "No model"], correct: 1 }
    ]
  },
  {
    id: 10,
    title: "Random Forests",
    pages: "249-263",
    summary: "This unit covers Random Forest ensemble method that combines multiple decision trees. It explains bagging, feature randomness, voting mechanisms, and how random forests reduce overfitting while maintaining high accuracy across various applications.",
    keyTakeaways: [
      "Random forests combine multiple decision trees via bagging",
      "Each tree trained on random subset with feature randomness",
      "Predictions made by majority voting (classification) or averaging (regression)",
      "Reduces overfitting and handles high-dimensional data well"
    ],
    quiz: [
      { question: "Random forest is based on?", options: ["One tree", "Ensemble of decision trees", "Linear models", "Clustering"], correct: 1 },
      { question: "Random forest uses which technique?", options: ["Boosting", "Bagging (Bootstrap Aggregating)", "Pruning only", "No technique"], correct: 1 },
      { question: "Random forest prediction is by?", options: ["First tree", "Majority voting/averaging", "Last tree", "Random tree"], correct: 1 },
      { question: "Random forest reduces?", options: ["Accuracy", "Overfitting", "Speed", "Data"], correct: 1 },
      { question: "Each tree in RF trained on?", options: ["All data", "Random subset of data", "No data", "Same data"], correct: 1 },
      { question: "Feature randomness means?", options: ["All features used", "Random subset of features per split", "No features", "One feature only"], correct: 1 },
      { question: "RF advantage over single tree?", options: ["Faster", "More accurate and robust", "Simpler", "Less memory"], correct: 1 },
      { question: "RF application includes?", options: ["Banking", "Medicine", "Land use", "All of these"], correct: 3 },
      { question: "RF handles what well?", options: ["Only small data", "Large datasets with high dimensionality", "No data", "Only images"], correct: 1 },
      { question: "RF disadvantage?", options: ["Too simple", "Less interpretable than single tree", "Too fast", "No accuracy"], correct: 1 }
    ]
  },
  {
    id: 11,
    title: "Deep Neural Networks",
    pages: "264-283",
    summary: "This unit introduces deep learning and neural network architectures. It covers feedforward, recurrent, and convolutional neural networks, explaining layers, activation functions, backpropagation, and applications in computer vision, NLP, and more.",
    keyTakeaways: [
      "Deep learning uses neural networks with multiple hidden layers",
      "Different architectures (CNN, RNN, DNN) suit different tasks",
      "Backpropagation and gradient descent train the networks",
      "Applications include image recognition, speech, and autonomous systems"
    ],
    quiz: [
      { question: "Deep learning is subset of?", options: ["Statistics", "Machine Learning", "Mathematics", "Physics"], correct: 1 },
      { question: "Neural networks inspired by?", options: ["Computers", "Human brain neurons", "Mathematics", "Physics"], correct: 1 },
      { question: "Hidden layers are between?", options: ["Input and output", "Two inputs", "Two outputs", "Nowhere"], correct: 0 },
      { question: "CNN primarily used for?", options: ["Text", "Image classification", "Audio only", "Nothing"], correct: 1 },
      { question: "RNN is used for?", options: ["Static images", "Sequential data like time series", "Clustering", "Dimensionality"], correct: 1 },
      { question: "Feedforward network means?", options: ["Data flows backward", "Data flows forward without cycles", "Random flow", "No flow"], correct: 1 },
      { question: "Deep learning requires?", options: ["Small data", "Large amounts of data", "No data", "One sample"], correct: 1 },
      { question: "Backpropagation is used for?", options: ["Prediction", "Training by adjusting weights", "Testing", "Nothing"], correct: 1 },
      { question: "DNN application includes?", options: ["Self-driving cars", "Voice assistants", "Image captioning", "All of these"], correct: 3 },
      { question: "Deep learning disadvantage?", options: ["Too simple", "Computationally expensive", "Too fast", "Needs no data"], correct: 1 }
    ]
  },
  {
    id: 12,
    title: "Naive Bayes Classifier",
    pages: "284-295",
    summary: "This unit covers Naive Bayes probabilistic classifier based on Bayes' theorem. It explains the naive independence assumption, probability calculations, different Naive Bayes variants (Gaussian, Multinomial, Bernoulli), and applications in text classification.",
    keyTakeaways: [
      "Based on Bayes' theorem with independence assumption",
      "Fast and effective for text classification tasks",
      "Three main types: Gaussian, Multinomial, Bernoulli",
      "Works well despite 'naive' feature independence assumption"
    ],
    quiz: [
      { question: "Naive Bayes is based on?", options: ["Newton's law", "Bayes' theorem", "Einstein's theory", "No theorem"], correct: 1 },
      { question: "Why is it called 'Naive'?", options: ["It's simple", "Assumes feature independence", "It's old", "No reason"], correct: 1 },
      { question: "Naive Bayes is used for?", options: ["Regression only", "Classification", "Clustering only", "Optimization"], correct: 1 },
      { question: "P(A|B) represents?", options: ["Prior probability", "Posterior probability", "Likelihood", "Evidence"], correct: 1 },
      { question: "Gaussian NB assumes features follow?", options: ["Uniform distribution", "Normal distribution", "Binomial", "No distribution"], correct: 1 },
      { question: "Multinomial NB used for?", options: ["Continuous data", "Document classification", "Images only", "Audio only"], correct: 1 },
      { question: "Bernoulli NB uses?", options: ["Continuous values", "Boolean/binary variables", "Text only", "Images"], correct: 1 },
      { question: "NB is popular for?", options: ["Gaming", "Spam filtering", "Graphics", "Nothing"], correct: 1 },
      { question: "NB advantage?", options: ["Slow training", "Fast and simple", "Complex", "Needs lots of data"], correct: 1 },
      { question: "NB limitation?", options: ["Too accurate", "Assumes feature independence", "Too slow", "No limitations"], correct: 1 }
    ]
  },
  {
    id: 13,
    title: "Model Selection & Evaluation",
    pages: "309-316",
    summary: "This unit covers techniques for selecting the best model for a problem. It explains cross-validation, train-test split, bootstrap methods, and evaluation criteria like AIC. It discusses balancing model complexity, performance, and generalization.",
    keyTakeaways: [
      "Model selection chooses best model for the problem",
      "Cross-validation evaluates model on multiple data splits",
      "AIC provides probabilistic measure for model comparison",
      "Balance between model complexity and performance is crucial"
    ],
    quiz: [
      { question: "Model selection is?", options: ["Choosing features", "Choosing best model for problem", "Deleting data", "Adding noise"], correct: 1 },
      { question: "K-Fold cross-validation splits data into?", options: ["Two parts", "K equal folds", "Random parts", "No split"], correct: 1 },
      { question: "Train-test split samples?", options: ["With replacement", "Without replacement", "Randomly always", "Never"], correct: 1 },
      { question: "Bootstrap samples?", options: ["Without replacement", "With replacement", "Never", "Once"], correct: 1 },
      { question: "AIC stands for?", options: ["Automated Information Center", "Akaike Information Criterion", "Advanced Integration Code", "None"], correct: 1 },
      { question: "Lower AIC indicates?", options: ["Worse model", "Better model", "No difference", "Error"], correct: 1 },
      { question: "Cross-validation helps prevent?", options: ["Training", "Overfitting", "Testing", "Nothing"], correct: 1 },
      { question: "Leave-one-out CV leaves out?", options: ["Half data", "One sample per iteration", "All data", "No data"], correct: 1 },
      { question: "Model selection considers?", options: ["Only accuracy", "Performance, robustness, complexity", "Only speed", "Nothing"], correct: 1 },
      { question: "Validation set is used for?", options: ["Training", "Tuning hyperparameters", "Final testing", "Nothing"], correct: 1 }
    ]
  },
  {
    id: 14,
    title: "Clustering Algorithms",
    pages: "317-351",
    summary: "This unit covers unsupervised clustering methods including K-means, hierarchical, density-based (DBSCAN), and distribution-based clustering. It explains cluster formation, distance metrics, applications in customer segmentation, and image analysis.",
    keyTakeaways: [
      "Clustering groups similar data points without labels",
      "K-means partitions data into K clusters using centroids",
      "Different clustering types suit different data structures",
      "Applications include customer segmentation and anomaly detection"
    ],
    quiz: [
      { question: "Clustering is what type of learning?", options: ["Supervised", "Unsupervised", "Reinforcement", "Semi-supervised"], correct: 1 },
      { question: "K in K-means represents?", options: ["Kernel", "Number of clusters", "Constant", "Key"], correct: 1 },
      { question: "K-means uses?", options: ["Supervised labels", "Centroids", "Decision trees", "No algorithm"], correct: 1 },
      { question: "DBSCAN is?", options: ["Centroid-based", "Density-based", "Model-based", "Random"], correct: 1 },
      { question: "Hierarchical clustering builds?", options: ["One cluster", "Tree of clusters (dendrogram)", "No structure", "Random clusters"], correct: 1 },
      { question: "K-means minimizes?", options: ["Clusters", "Within-cluster variance", "Between-cluster variance", "Nothing"], correct: 1 },
      { question: "Clustering application?", options: ["Supervised learning", "Customer segmentation", "Regression", "Classification"], correct: 1 },
      { question: "Elbow method helps find?", options: ["Optimal K value", "Best algorithm", "Data quality", "Nothing"], correct: 0 },
      { question: "K-means assumes clusters are?", options: ["Any shape", "Spherical", "Linear", "Random"], correct: 1 },
      { question: "DBSCAN advantage?", options: ["Needs K value", "Finds arbitrary shaped clusters", "Very slow", "Needs labels"], correct: 1 }
    ]
  },
  {
    id: 15,
    title: "Dimensionality Reduction",
    pages: "330-339",
    summary: "This unit explains techniques to reduce feature dimensions while preserving information. It covers feature selection, feature extraction, PCA (Principal Component Analysis), and the benefits of reducing dimensionality for visualization and computational efficiency.",
    keyTakeaways: [
      "Reduces number of features while retaining information",
      "PCA finds principal components explaining most variance",
      "Helps with visualization, computation speed, and storage",
      "Can remove redundant/correlated features"
    ],
    quiz: [
      { question: "Dimensionality reduction reduces?", options: ["Data rows", "Number of features", "Accuracy", "Nothing"], correct: 1 },
      { question: "PCA finds?", options: ["Clusters", "Principal components", "Labels", "Errors"], correct: 1 },
      { question: "Dimensionality reduction helps with?", options: ["Adding features", "Visualization and computation", "Data collection", "Labeling"], correct: 1 },
      { question: "Feature selection is?", options: ["Creating new features", "Choosing subset of relevant features", "Deleting all data", "Random process"], correct: 1 },
      { question: "Feature extraction?", options: ["Deletes features", "Transforms features to new space", "Does nothing", "Adds noise"], correct: 1 },
      { question: "PCA benefit?", options: ["Increases dimensions", "Reduces dimensions while preserving variance", "Adds complexity", "No benefit"], correct: 1 },
      { question: "Curse of dimensionality refers to?", options: ["Good performance", "Problems with too many features", "Fast computation", "Nothing"], correct: 1 },
      { question: "Forward feature selection?", options: ["Removes features", "Progressively adds best features", "Adds all features", "Random"], correct: 1 },
      { question: "Dimensionality reduction disadvantage?", options: ["Too fast", "Possible information loss", "Too simple", "No disadvantage"], correct: 1 },
      { question: "PCA components are?", options: ["Random", "Orthogonal vectors", "Original features", "Labels"], correct: 1 }
    ]
  },
  {
    id: 16,
    title: "Expectation-Maximization Algorithm",
    pages: "352-362",
    summary: "This unit covers the EM algorithm for estimating parameters with missing or latent data. It explains the E-step (expectation) and M-step (maximization), convergence properties, and applications in clustering, mixture models, and handling incomplete data.",
    keyTakeaways: [
      "EM iteratively estimates parameters with missing data",
      "E-step computes expected values of latent variables",
      "M-step maximizes likelihood using E-step results",
      "Used in Gaussian Mixture Models and missing data problems"
    ],
    quiz: [
      { question: "EM stands for?", options: ["Error Management", "Expectation-Maximization", "Efficient Modeling", "Exact Method"], correct: 1 },
      { question: "EM is used when data is?", options: ["Complete", "Missing or has latent variables", "Perfect", "Labeled"], correct: 1 },
      { question: "E-step computes?", options: ["Final answer", "Expected values of latent variables", "Error only", "Nothing"], correct: 1 },
      { question: "M-step does what?", options: ["Minimizes parameters", "Maximizes likelihood", "Deletes data", "Adds noise"], correct: 1 },
      { question: "EM algorithm iterates until?", options: ["Never stops", "Convergence reached", "One iteration", "Error increases"], correct: 1 },
      { question: "EM is used in?", options: ["Supervised learning only", "Gaussian Mixture Models", "Linear regression", "Nothing"], correct: 1 },
      { question: "EM advantage?", options: ["Handles missing data", "Very fast always", "No computation", "Perfect accuracy"], correct: 0 },
      { question: "EM disadvantage?", options: ["Too simple", "Slow convergence", "Too fast", "No disadvantage"], correct: 1 },
      { question: "EM guarantees?", options: ["Global optimum", "Likelihood increases per iteration", "Random results", "Nothing"], correct: 1 },
      { question: "EM applications include?", options: ["Only clustering", "Image reconstruction, medical data", "Only text", "Nothing"], correct: 1 }
    ]
  },
  {
    id: 17,
    title: "Gaussian Mixture Models",
    pages: "363-369",
    summary: "This unit explains Gaussian Mixture Models (GMM) for density estimation and clustering. GMM assumes data comes from multiple Gaussian distributions. It covers model parameters, EM algorithm for training, and applications in clustering and anomaly detection.",
    keyTakeaways: [
      "GMM models data as mixture of Gaussian distributions",
      "Each component has mean, variance, and mixing weight",
      "Trained using EM algorithm",
      "Flexible for clustering and density estimation"
    ],
    quiz: [
      { question: "GMM assumes data from?", options: ["One distribution", "Mixture of Gaussian distributions", "Uniform distribution", "No distribution"], correct: 1 },
      { question: "Each Gaussian component has?", options: ["Only mean", "Mean, variance, and weight", "No parameters", "Only variance"], correct: 1 },
      { question: "GMM is trained using?", options: ["Gradient descent", "EM algorithm", "Random search", "No training"], correct: 1 },
      { question: "Mixing weight represents?", options: ["Error", "Probability of component", "Mean", "Variance"], correct: 1 },
      { question: "GMM can be used for?", options: ["Only classification", "Clustering and density estimation", "Only regression", "Nothing"], correct: 1 },
      { question: "GMM advantage over K-means?", options: ["Faster", "Models elliptical clusters with probabilities", "Simpler", "No advantage"], correct: 1 },
      { question: "In GMM, data points?", options: ["Belong to one cluster", "Have probability of belonging to each cluster", "Have no cluster", "Are deleted"], correct: 1 },
      { question: "GMM applications include?", options: ["Image segmentation", "Speech recognition", "Anomaly detection", "All of these"], correct: 3 },
      { question: "Number of Gaussian components?", options: ["Always 1", "Must be specified (like K)", "Infinite", "Zero"], correct: 1 },
      { question: "GMM disadvantage?", options: ["Too simple", "Sensitive to initialization", "Too fast", "No parameters"], correct: 1 }
    ]
  },
  {
    id: 18,
    title: "Mixture of Naive Bayes",
    pages: "370-376",
    summary: "This unit covers Mixture of Naive Bayes models that combine multiple Naive Bayes classifiers. It explains how MNB handles data from multiple subpopulations, latent cluster variables, and improves classification by modeling complex distributions.",
    keyTakeaways: [
      "Combines multiple Naive Bayes models for complex data",
      "Assumes data from multiple subpopulations/clusters",
      "Uses latent variables to indicate cluster membership",
      "More flexible than single Naive Bayes"
    ],
    quiz: [
      { question: "Mixture of Naive Bayes combines?", options: ["One model", "Multiple Naive Bayes classifiers", "Decision trees", "SVMs"], correct: 1 },
      { question: "MNB assumes data from?", options: ["One population", "Multiple subpopulations", "No population", "Random source"], correct: 1 },
      { question: "Latent variable in MNB indicates?", options: ["Error", "Cluster membership", "Accuracy", "Time"], correct: 1 },
      { question: "MNB is useful when?", options: ["Data is simple", "Single NB assumption too restrictive", "No data", "Always"], correct: 1 },
      { question: "Within each cluster, MNB assumes?", options: ["No independence", "Feature independence given class", "All dependent", "Random"], correct: 1 },
      { question: "MNB classification assigns to?", options: ["Random cluster", "Cluster with highest posterior probability", "First cluster", "No cluster"], correct: 1 },
      { question: "MNB advantage?", options: ["Simpler than NB", "Captures complex relationships", "Faster", "Needs no data"], correct: 1 },
      { question: "MNB used in?", options: ["Only images", "Text classification", "Only audio", "Nothing"], correct: 1 },
      { question: "MNB parameters include?", options: ["Only one set", "Multiple sets for each cluster", "No parameters", "Random values"], correct: 1 },
      { question: "Compared to single NB, MNB is?", options: ["Simpler", "More flexible and complex", "Same", "Worse"], correct: 1 }
    ]
  },
  {
    id: 19,
    title: "Hidden Markov Models",
    pages: "377-391",
    summary: "This unit introduces Hidden Markov Models for modeling sequential data with hidden states. It covers HMM components (states, observations, transitions, emissions), the Forward-Backward and Viterbi algorithms, and applications in speech recognition and bioinformatics.",
    keyTakeaways: [
      "HMMs model sequences with observable and hidden states",
      "Consists of transition and emission probabilities",
      "Viterbi finds most likely state sequence",
      "Applications in speech, NLP, and bioinformatics"
    ],
    quiz: [
      { question: "HMM models?", options: ["Static data", "Sequential data with hidden states", "Images only", "No data"], correct: 1 },
      { question: "In HMM, states are?", options: ["Always visible", "Hidden/unobservable", "Random", "Deleted"], correct: 1 },
      { question: "HMM has two types of probabilities?", options: ["Only transition", "Transition and emission", "Only emission", "No probabilities"], correct: 1 },
      { question: "Viterbi algorithm finds?", options: ["Random sequence", "Most likely hidden state sequence", "All sequences", "No sequence"], correct: 1 },
      { question: "Forward-Backward algorithm computes?", options: ["Nothing", "Probability of observation sequence", "Only states", "Random value"], correct: 1 },
      { question: "HMM used in?", options: ["Only images", "Speech recognition", "Static data", "Nothing"], correct: 1 },
      { question: "Transition probability is?", options: ["State to observation", "State to state", "Random", "Constant"], correct: 1 },
      { question: "Emission probability is?", options: ["State to state", "State to observation", "Random", "Zero"], correct: 1 },
      { question: "HMM assumes?", options: ["All states visible", "Markov property", "No dependencies", "Random process"], correct: 1 },
      { question: "HMM limitation?", options: ["Too simple", "Fixed state space assumption", "Too fast", "No limitation"], correct: 1 }
    ]
  },
  {
    id: 20,
    title: "Reinforcement Learning",
    pages: "392-411",
    summary: "This unit covers reinforcement learning where agents learn through interaction with environment. It explains key concepts (agent, environment, state, action, reward), policy, value functions, Q-learning, and applications in robotics, games, and autonomous systems.",
    keyTakeaways: [
      "Agent learns by interacting with environment through trial and error",
      "Receives rewards/penalties for actions taken",
      "Goal is to maximize cumulative long-term reward",
      "Applications in robotics, games, autonomous vehicles"
    ],
    quiz: [
      { question: "In RL, agent learns through?", options: ["Labeled data", "Trial and error interaction", "Memorization", "No learning"], correct: 1 },
      { question: "Agent receives what for actions?", options: ["Nothing", "Rewards or penalties", "Only data", "Labels"], correct: 1 },
      { question: "RL goal is to?", options: ["Minimize reward", "Maximize cumulative reward", "Random actions", "No goal"], correct: 1 },
      { question: "Policy in RL is?", options: ["Data", "Strategy for selecting actions", "Reward", "State"], correct: 1 },
      { question: "Value function estimates?", options: ["Current reward", "Expected long-term reward", "Action count", "Nothing"], correct: 1 },
      { question: "Q-value represents?", options: ["Quantum value", "Quality of action in state", "Quick value", "No meaning"], correct: 1 },
      { question: "RL environment is typically?", options: ["Deterministic", "Stochastic", "Static", "None"], correct: 1 },
      { question: "Model-based RL uses?", options: ["No model", "Virtual model of environment", "Only data", "Random model"], correct: 1 },
      { question: "RL applications include?", options: ["Self-driving cars", "Game playing", "Robot control", "All of these"], correct: 3 },
      { question: "Exploration vs exploitation in RL?", options: ["No tradeoff", "Balance trying new vs using known actions", "Always explore", "Always exploit"], correct: 1 }
    ]
  }
];