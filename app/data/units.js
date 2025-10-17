export const units = [
  {
    id: 1,
    title: "Introduction to Machine Learning",
    pages: "1–15",
    pdfFile: "lecture_notes.pdf",
    summary: `### Overview
Machine Learning (ML) is a **subfield of Artificial Intelligence (AI)** that enables computers to learn patterns from data and make predictions or decisions **without explicit programming**. The primary goal of ML is to allow systems to **improve performance over time** through experience.

### Key Concepts
- **Data:** The foundation of ML, existing as structured (tables), unstructured (text, images), or semi-structured formats.
- **Features:** Characteristics or measurable properties of the data that are fed into models.
- **Labels:** In supervised learning, these represent the known outputs or target variables.
- **Algorithms/Models:** Mathematical methods that learn patterns from data to perform tasks like classification or prediction.
- **Training:** The process where the model adjusts internal parameters to minimize prediction error.
- **Validation & Testing:** Ensures the trained model generalizes well to unseen data and avoids overfitting.
- **Metrics:** Quantitative measures of model performance, such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **RMSE**.
- **Hyperparameters:** External configuration settings (e.g., learning rate, number of layers) that guide model training.
- **Deployment:** Integration of the trained model into a real-world environment to make predictions.
- **Iterative Process:** Machine learning involves continuous refinement — retraining models as data and requirements evolve.

### Learning Paradigms
1. **Supervised Learning:** Uses labeled data (features + known outputs) for prediction.
   - Examples: Linear Regression, Decision Trees, Neural Networks.
2. **Unsupervised Learning:** Uses unlabeled data to discover hidden patterns or groupings.
   - Examples: Clustering, Dimensionality Reduction.
3. **Semi-Supervised Learning:** Mix of labeled and unlabeled data.
4. **Reinforcement Learning:** Learning by interaction with an environment to maximize rewards (used in robotics, gaming).

### Applications
Machine learning powers systems such as:
- **Speech & Image Recognition**
- **Recommendation Engines** (Netflix, Spotify, Amazon)
- **Healthcare Diagnostics**
- **Autonomous Vehicles**
- **Fraud Detection**

### Ethical & Practical Considerations
ML systems must consider **bias, fairness, transparency**, and **data quality** to ensure responsible use.

### Example Lifecycle
1. Collect and preprocess data.
2. Choose algorithm and split data (train/validate/test).
3. Train the model.
4. Evaluate using metrics.
5. Deploy and monitor performance.`,
    keyTakeaways: [
      "Machine Learning is a subset of AI focused on learning from data rather than explicit programming.",
      "Data quality and representation directly impact model performance.",
      "Supervised, unsupervised, and reinforcement learning form the main ML paradigms.",
      "Models require iterative tuning through training, validation, and testing.",
      "Overfitting occurs when models memorize training data instead of learning patterns.",
      "Hyperparameters control the learning process and must be tuned carefully.",
      "Deployment integrates models into production for real-world predictions.",
      "ML applications range from healthcare to finance, robotics, and entertainment.",
    ],
    quiz: [
      {
        question: "### What is the primary goal of Machine Learning?",
        options: [
          "To store large amounts of data",
          "To program computers explicitly",
          "To enable computers to learn and improve from data",
          "To automate hardware control",
        ],
        correct: 2,
      },
      {
        question:
          "### Which of the following describes *features* in a dataset?",
        options: [
          "The predicted outputs or targets",
          "The characteristics or measurable properties of data",
          "The algorithms used to process data",
          "The labels assigned during training",
        ],
        correct: 1,
      },
      {
        question: "### What type of data does **supervised learning** use?",
        options: [
          "Only unlabeled data",
          "Partially labeled data",
          "Labeled input-output pairs",
          "Data without features",
        ],
        correct: 2,
      },
      {
        question: "### Which algorithm belongs to unsupervised learning?",
        options: [
          "Decision Trees",
          "Linear Regression",
          "Clustering",
          "Logistic Regression",
        ],
        correct: 2,
      },
      {
        question: "### What is overfitting?",
        options: [
          "When a model performs well on new data",
          "When a model memorizes training data but fails on unseen data",
          "When a model uses too few features",
          "When the learning rate is too high",
        ],
        correct: 1,
      },
      {
        question:
          "### Which of these metrics is commonly used for classification tasks?",
        options: [
          "Root Mean Squared Error (RMSE)",
          "Accuracy and F1-score",
          "R-squared",
          "Mean Absolute Error",
        ],
        correct: 1,
      },
      {
        question: "### What is the purpose of validation data?",
        options: [
          "Used for training only",
          "Used to tune hyperparameters and check overfitting",
          "Used for deployment",
          "Used to collect raw data",
        ],
        correct: 1,
      },
      {
        question: "### In reinforcement learning, what drives learning?",
        options: [
          "Manual labeling",
          "Supervised datasets",
          "Reward signals from environment interactions",
          "Random weight updates",
        ],
        correct: 2,
      },
      {
        question: "### Which is a correct example of a **hyperparameter**?",
        options: [
          "Predicted label values",
          "Learning rate",
          "Training data sample",
          "Validation accuracy",
        ],
        correct: 1,
      },
      {
        question: "### Which step ensures that a model can handle unseen data?",
        options: ["Training", "Validation", "Testing", "Feature scaling"],
        correct: 2,
      },
    ],
  },
  {
    id: 2,
    title: "Perspectives and Issues in Machine Learning",
    pages: "16–30",
    pdfFile: "lecture_notes.pdf",
    summary: `### Overview
Machine Learning (ML) has transformed industries with automation, prediction, and data-driven insights. However, it introduces **critical challenges** — such as bias, fairness, interpretability, transparency, privacy, and ethical concerns — that require both technical and societal solutions.

### Perspectives
#### 1. Technological Advancements
ML drives innovation by automating repetitive tasks, improving prediction accuracy, and enabling intelligent systems.

#### 2. Data-Driven Insights
Organizations use ML to uncover hidden insights from massive datasets, supporting better decisions and personalized experiences.

#### 3. Personalization
From recommendations to precision medicine, ML provides **context-aware, user-specific experiences**.

#### 4. Automation & Efficiency
ML minimizes human effort in processes such as customer support, logistics, and manufacturing.

#### 5. Scientific Discovery
ML accelerates progress in genomics, physics, and material science by analyzing complex data relationships.

### Issues and Challenges
1. **Bias and Fairness** — Unbalanced or biased datasets can produce unfair models.
2. **Interpretability** — Deep models can become black boxes, hard to explain.
3. **Data Privacy** — Sensitive data requires secure handling.
4. **Data Quality** — Garbage in, garbage out — data drives outcomes.
5. **Overfitting** — Weak generalization to unseen data.
6. **Ethical Considerations** — AI decisions can impact human lives.
7. **Transparency** — Understanding how a model makes decisions is essential.
8. **Resource Intensity** — Deep learning consumes high energy and compute.
9. **Domain Knowledge** — Lacking context can cause mispredictions.
10. **Job Displacement** — Automation can replace human roles.
11. **Governance & Regulation** — Needed for fairness and accountability.

### Responsible Development
Ethical AI requires cooperation between **researchers, policymakers, and technologists**, ensuring fairness, transparency, and sustainability.`,
    keyTakeaways: [
      "Machine learning offers transformative benefits but introduces ethical and societal challenges.",
      "Bias in data leads to unfair predictions — fairness and equity must be prioritized.",
      "Interpretability and transparency are vital for accountability and trust.",
      "Data privacy must be safeguarded using secure and ethical practices.",
      "High-quality, diverse datasets are essential for robust and generalizable models.",
      "Overfitting reduces a model’s ability to perform on unseen data.",
      "ML can increase automation but may also contribute to job displacement.",
      "Energy consumption and resource use in deep models raise environmental concerns.",
      "Legal and regulatory frameworks are critical to guide responsible AI adoption.",
      "Collaboration between technologists, ethicists, and policymakers ensures responsible innovation.",
    ],
    quiz: [
      {
        question:
          "### What is one of the main benefits of machine learning from a technological perspective?",
        options: [
          "It reduces access to data.",
          "It automates tasks and improves decision-making.",
          "It replaces human intelligence completely.",
          "It limits scalability of computation.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which of the following best defines *bias* in machine learning?",
        options: [
          "A mathematical adjustment to increase accuracy.",
          "Systematic errors introduced by unrepresentative data or assumptions.",
          "A random variation in model outputs.",
          "The process of making models faster.",
        ],
        correct: 1,
      },
      {
        question: "### Why is interpretability important in ML models?",
        options: [
          "Because it increases computation time.",
          "Because it helps users understand and trust predictions.",
          "Because it simplifies the algorithm mathematically.",
          "Because it reduces dataset size.",
        ],
        correct: 1,
      },
      {
        question: "### What is **differential privacy** used for?",
        options: [
          "Encrypting model parameters.",
          "Protecting individual data while performing analysis.",
          "Improving accuracy through large datasets.",
          "Reducing model training time.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which issue arises when ML models consume excessive computational resources?",
        options: [
          "Model simplicity",
          "Data leakage",
          "Environmental and energy concerns",
          "Reduced interpretability",
        ],
        correct: 2,
      },
    ],
  },
  {
    id: 3,
    title: "Concept Learning",
    pages: "31–42",
    pdfFile: "lecture_notes.pdf",
    summary: `### Overview
**Concept learning** is a foundational process in both **machine learning** and **cognitive science**.  
It refers to a model’s ability to **acquire, understand, and generalize concepts** or categories from examples and experiences.  
The primary goal is to enable a system to **classify new, unseen instances** into appropriate categories based on learned patterns.

### Process of Concept Learning
1. **Data Collection** — Gathering labeled examples or instances representing various categories.
2. **Feature Extraction** — Identifying the most relevant characteristics (features) that distinguish categories.
3. **Training Phase** — Using labeled examples to learn patterns or rules that define each concept.
4. **Generalization** — Applying learned rules to classify new, unseen examples correctly.
5. **Testing & Evaluation** — Measuring model accuracy using unseen data (test sets) and metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
6. **Concept Evolution** — Updating learned concepts when new data introduces variations or exceptions.

### Types of Concept Learning
#### 1. Inductive Learning
- Involves inferring **general rules from specific examples**.
- Example: Observing that several birds can fly → generalizing “birds can fly.”
- Used in most machine learning models.

#### 2. Deductive Learning
- Derives **specific examples from general rules**.
- Example: Knowing “all mammals are warm-blooded” → deducing “a whale is warm-blooded.”

#### 3. Abductive Learning
- Forming **hypotheses** that best explain observed data.
- Often used in diagnostic systems (e.g., medical diagnosis).

#### 4. Instance-Based Learning
- Instead of learning abstract rules, this approach stores **specific examples** in memory.
- When a new instance appears, it is compared against stored examples to make a decision.
- Examples: *k-Nearest Neighbors (kNN)*, *Case-Based Reasoning.*

### Applications
Concept learning supports key ML tasks:
- **Image recognition** (e.g., classifying animals or objects)
- **Natural language processing** (e.g., word categorization)
- **Medical diagnostics** (classifying symptoms into diseases)
- **Recommendation systems** (grouping user preferences)
- **Fraud detection and anomaly detection**

### Importance
Concept learning is what allows ML systems to **mimic human-like categorization** — identifying, organizing, and generalizing knowledge from experiences.  
It bridges **data-driven learning** and **symbolic reasoning**, improving adaptability and contextual understanding.

---`,
    keyTakeaways: [
      "Concept learning enables systems to form generalizations from examples.",
      "It involves key stages: data collection, feature extraction, training, generalization, and evaluation.",
      "Inductive learning generalizes from examples; deductive learning applies known rules.",
      "Abductive reasoning helps generate hypotheses explaining observed phenomena.",
      "Instance-based learning classifies by comparing new examples with stored instances.",
      "Concept learning underlies major ML applications like classification, NLP, and image recognition.",
      "Concept evolution ensures that models remain adaptable to new patterns.",
      "Evaluation metrics such as accuracy, precision, and recall measure concept learning effectiveness.",
    ],
    quiz: [
      {
        question: "### What is the main objective of concept learning?",
        options: [
          "To memorize all training examples exactly.",
          "To generalize concepts from examples for classifying new instances.",
          "To optimize neural network architectures.",
          "To cluster data without supervision.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which step involves identifying relevant attributes that distinguish categories?",
        options: [
          "Data Collection",
          "Feature Extraction",
          "Generalization",
          "Testing and Evaluation",
        ],
        correct: 1,
      },
      {
        question:
          "### What kind of learning infers general rules from specific examples?",
        options: [
          "Deductive Learning",
          "Inductive Learning",
          "Abductive Learning",
          "Instance-Based Learning",
        ],
        correct: 1,
      },
      {
        question: "### In **deductive learning**, knowledge flows from:",
        options: [
          "Specific to general",
          "General to specific",
          "Unsupervised to supervised",
          "Data to metadata",
        ],
        correct: 1,
      },
      {
        question:
          "### Which learning type is most useful in diagnostic systems?",
        options: [
          "Inductive Learning",
          "Abductive Learning",
          "Instance-Based Learning",
          "Deductive Learning",
        ],
        correct: 1,
      },
      {
        question:
          "### Which algorithm is an example of instance-based learning?",
        options: [
          "Decision Trees",
          "k-Nearest Neighbors (kNN)",
          "Neural Networks",
          "Naive Bayes",
        ],
        correct: 1,
      },
      {
        question: "### What is **concept evolution**?",
        options: [
          "The process of retraining a model on the same data repeatedly.",
          "The adaptation of a concept over time as new data introduces variations.",
          "The deletion of outdated data points.",
          "The conversion of data into numerical features.",
        ],
        correct: 1,
      },
      {
        question:
          "### What does the generalization step in concept learning involve?",
        options: [
          "Evaluating model performance on test data.",
          "Using learned concepts to correctly classify unseen data.",
          "Extracting important features.",
          "Removing irrelevant data.",
        ],
        correct: 1,
      },
      {
        question: "### Why is evaluation important in concept learning?",
        options: [
          "It identifies new patterns for labeling data.",
          "It determines how well the learned concept applies to new examples.",
          "It increases dataset size.",
          "It automatically extracts features.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which metric combination is commonly used for evaluating classification performance?",
        options: [
          "Accuracy, Precision, Recall, and F1-score",
          "Variance, Bias, and Covariance",
          "ROC, IOU, and Perplexity",
          "Entropy, Gini, and RMSE",
        ],
        correct: 0,
      },
      {
        question:
          "### What is a key difference between inductive and deductive learning?",
        options: [
          "Inductive goes from data to rules; deductive applies rules to data.",
          "Deductive uses examples; inductive uses symbolic reasoning.",
          "Inductive is deterministic; deductive is probabilistic.",
          "There is no difference.",
        ],
        correct: 0,
      },
      {
        question:
          "### In what way does instance-based learning differ from rule-based learning?",
        options: [
          "It discards all training data after training.",
          "It stores and uses specific instances instead of abstract rules.",
          "It requires labeled data.",
          "It cannot generalize to new examples.",
        ],
        correct: 1,
      },
      {
        question:
          "### What kind of reasoning is abductive learning most similar to?",
        options: [
          "Forming hypotheses to explain evidence.",
          "Deriving conclusions from axioms.",
          "Memorizing data patterns.",
          "Eliminating irrelevant variables.",
        ],
        correct: 0,
      },
    ],
  },
  {
    id: 4,
    title: "Related Areas of Machine Learning",
    pages: "43–59",
    pdfFile: "lecture_notes.pdf",
    summary: `### Overview
Machine Learning (ML) is an interdisciplinary field that overlaps with many related areas of computing and data science.  
These interconnected domains enhance ML’s capabilities, applications, and theoretical foundations — from artificial intelligence to data analysis, optimization, and beyond.

---

### 1. Artificial Intelligence (AI)
- **AI** is the broader field that encompasses ML.
- Focuses on building intelligent systems capable of reasoning, perception, planning, and learning.
- ML is the data-driven component of AI that allows systems to learn from experience rather than rules.

---

### 2. Deep Learning (DL)
- A **subset of ML** using *multi-layered neural networks* (deep architectures) to model complex relationships.
- Excels in recognizing patterns from large datasets (e.g., images, text, audio).
- **Applications:** image recognition, NLP, reinforcement learning, and game AI.
- Frameworks: TensorFlow, PyTorch, Keras.

---

### 3. Neural Networks (NN)
- Inspired by the **structure of the human brain**.
- Composed of layers of interconnected “neurons.”
- Used for **pattern recognition**, function approximation, and learning complex mappings between inputs and outputs.
- Basis of modern deep learning.

---

### 4. Data Science
- The broader discipline involving **data collection, cleaning, exploration, visualization**, and **statistical analysis**.
- ML algorithms are essential tools in data science for prediction and decision-making.
- Combines programming, statistics, and domain expertise.

---

### 5. Natural Language Processing (NLP)
- Focuses on enabling computers to **understand, interpret, and generate human language**.
- Core tasks:
  - Sentiment analysis
  - Text summarization
  - Machine translation
  - Named entity recognition
  - Chatbots and conversational AI
- NLP combines **linguistics, ML, and deep learning**.

---

### 6. Computer Vision (CV)
- Enables machines to **interpret and analyze visual data** (images and videos).
- Applications:
  - Object and facial recognition
  - Image segmentation
  - Gesture recognition
  - Autonomous navigation
- Uses **convolutional neural networks (CNNs)** for spatial data understanding.

---

### 7. Reinforcement Learning (RL)
- Learning via **interaction with an environment**.
- Agents learn to take actions that **maximize long-term rewards**.
- Widely used in robotics, gaming (e.g., AlphaGo), and adaptive control systems.

---

### 8. Unsupervised and Semi-Supervised Learning
- **Unsupervised Learning:** Learns from unlabeled data to find structure (e.g., clustering, dimensionality reduction).
- **Semi-Supervised Learning:** Combines small amounts of labeled data with large amounts of unlabeled data.

---

### 9. Transfer Learning
- Involves **reusing knowledge** learned from one task to improve performance on another.
- Example: Using a pretrained CNN on ImageNet and fine-tuning it for medical image classification.
- Saves time and computation, especially when labeled data is scarce.

---

### 10. Explainable AI (XAI)
- Aims to make ML systems **transparent, interpretable, and explainable**.
- Helps users understand how predictions are made, improving accountability and trust.
- Crucial for regulated fields like finance, law, and healthcare.

---

### 11. Ethics in AI
- Addresses **bias, fairness, accountability, and transparency**.
- Promotes responsible AI development that prioritizes **human values and societal impact**.
- Involves data governance, privacy protection, and inclusivity.

---

### 12. Bayesian Learning
- Uses **probabilistic reasoning** for prediction and inference.
- Incorporates prior knowledge and updates beliefs with new evidence.
- Ideal for uncertain and dynamic environments.

---

### 13. Causal Inference
- Focuses on identifying **cause-and-effect relationships** from data.
- Moves beyond correlation to enable reliable, interpretable decision-making.
- Important in medicine, economics, and policy analysis.

---

### 14. Optimization
- Optimization techniques help ML models **minimize loss functions** and tune parameters effectively.
- Methods include:
  - Gradient Descent
  - Stochastic Gradient Descent
  - Genetic Algorithms
  - Convex optimization
- Optimization is fundamental to all model training.

---

### 15. Time Series Analysis
- Concerned with data indexed over time (e.g., stock prices, weather data, IoT readings).
- ML models detect patterns, trends, and seasonal variations.
- Applications: forecasting, anomaly detection, and signal processing.

---

### 16. Quantum Machine Learning (QML)
- A cutting-edge area combining **quantum computing and ML**.
- Explores how quantum mechanics can enhance learning efficiency.
- Promising for solving **high-dimensional, computationally intensive problems**.

---

### Summary
Together, these related areas extend ML’s reach into various domains — enabling smarter, faster, and more responsible AI systems across science, business, and everyday life.`,
    keyTakeaways: [
      "Machine learning overlaps with several fields that enhance its capabilities and applications.",
      "Artificial Intelligence (AI) is the parent discipline of ML.",
      "Deep learning uses multi-layered neural networks for complex pattern recognition.",
      "NLP enables computers to understand and generate human language.",
      "Computer Vision helps machines interpret visual data using convolutional neural networks.",
      "Reinforcement learning trains agents through interaction and feedback.",
      "Transfer learning accelerates development by reusing pre-trained models.",
      "Explainable AI promotes transparency, fairness, and accountability in model behavior.",
      "Bayesian learning integrates probabilistic reasoning into model inference.",
      "Optimization techniques form the mathematical foundation of all ML training.",
    ],
    quiz: [
      {
        question:
          "### What is the relationship between AI and Machine Learning?",
        options: [
          "ML is a subset of AI.",
          "AI is a subset of ML.",
          "They are unrelated fields.",
          "ML replaces AI completely.",
        ],
        correct: 0,
      },
      {
        question: "### What distinguishes deep learning from traditional ML?",
        options: [
          "It relies on rule-based logic.",
          "It uses multi-layer neural networks to learn complex patterns.",
          "It does not require any data.",
          "It only handles numerical data.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which field focuses on enabling machines to process and generate human language?",
        options: [
          "Computer Vision",
          "Optimization",
          "Natural Language Processing (NLP)",
          "Bayesian Inference",
        ],
        correct: 2,
      },
      {
        question:
          "### Which network type is the foundation of modern deep learning?",
        options: [
          "Decision Trees",
          "Neural Networks",
          "Markov Models",
          "Genetic Algorithms",
        ],
        correct: 1,
      },
      {
        question: "### What is the main function of **Computer Vision**?",
        options: [
          "Analyzing numerical time-series data",
          "Recognizing and interpreting images and videos",
          "Understanding natural language",
          "Predicting user preferences",
        ],
        correct: 1,
      },
      {
        question: "### What does **Reinforcement Learning** emphasize?",
        options: [
          "Learning by imitation",
          "Learning by reward and punishment through environment interaction",
          "Learning from labeled data only",
          "Learning purely from text data",
        ],
        correct: 1,
      },
      {
        question: "### What is the key advantage of **Transfer Learning**?",
        options: [
          "It trains models from scratch for every task.",
          "It allows leveraging previously trained models to save resources.",
          "It removes the need for labeled data.",
          "It simplifies neural architectures completely.",
        ],
        correct: 1,
      },
      {
        question: "### What does Explainable AI (XAI) aim to achieve?",
        options: [
          "To make AI predictions faster only.",
          "To make models transparent and interpretable to humans.",
          "To eliminate all bias automatically.",
          "To replace training data with rules.",
        ],
        correct: 1,
      },
      {
        question:
          "### Which area focuses on uncovering cause-and-effect relationships?",
        options: [
          "Optimization",
          "Causal Inference",
          "Semi-Supervised Learning",
          "Transfer Learning",
        ],
        correct: 1,
      },
      {
        question: "### What is **Bayesian Learning** primarily based on?",
        options: [
          "Statistical inference and probabilistic reasoning",
          "Deterministic rule-based systems",
          "Manual labeling",
          "Quantum computation",
        ],
        correct: 0,
      },
      {
        question: "### What is the purpose of optimization in ML?",
        options: [
          "To visualize data",
          "To minimize or maximize a model’s objective function",
          "To store datasets efficiently",
          "To encode language representations",
        ],
        correct: 1,
      },
      {
        question:
          "### Which field combines quantum mechanics and machine learning?",
        options: [
          "Quantum Machine Learning",
          "Bayesian Networks",
          "Explainable AI",
          "Cognitive Computing",
        ],
        correct: 0,
      },
      {
        question:
          "### Which field forms the foundation for predictive analytics?",
        options: [
          "Data Science",
          "Computer Vision",
          "Deep Learning",
          "Quantum Computing",
        ],
        correct: 0,
      },
    ],
  },
  {
  id: 5,
  title: "Applications of Machine Learning",
  pages: "60–77",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
Machine Learning (ML) has become a transformative force across nearly every industry, automating decision-making, improving predictions, and enabling new forms of intelligence.  
Its versatility stems from its ability to **learn from data**, identify hidden patterns, and **adapt to changing environments**.

### 1. Healthcare
- **Diagnostics:** ML models detect diseases from X-rays, MRIs, or blood tests (e.g., cancer, pneumonia detection).
- **Drug Discovery:** Predict molecular behavior and simulate chemical reactions.
- **Personalized Medicine:** Predict optimal treatments for individual patients using genetic and lifestyle data.
- **Epidemiology:** Predict disease spread and optimize response planning.

**Key Techniques:** Deep Learning (CNNs, RNNs), Regression, and Ensemble Learning.

---

### 2. Finance and Banking
- **Fraud Detection:** ML detects suspicious transaction patterns and anomalies.
- **Credit Scoring:** Predict creditworthiness using multiple behavioral and financial indicators.
- **Algorithmic Trading:** Predict stock price movements using real-time market data.
- **Risk Management:** Simulate potential losses using probabilistic and predictive models.

**Common Models:** Decision Trees, SVMs, and Neural Networks.

---

### 3. Retail and E-commerce
- **Recommendation Systems:** Suggest products based on user history and preferences (e.g., Amazon, Netflix).
- **Customer Segmentation:** Cluster customers into distinct groups using unsupervised learning.
- **Demand Forecasting:** Predict future sales trends based on seasonality, promotions, and user behavior.
- **Churn Prediction:** Identify customers likely to leave and take retention measures.

**Key Techniques:** Collaborative Filtering, Clustering, Regression.

---

### 4. Manufacturing and Industry
- **Predictive Maintenance:** Forecast machine failures before they occur, reducing downtime.
- **Quality Control:** Detect product defects using image recognition.
- **Supply Chain Optimization:** Improve production efficiency and logistics.
- **Robotics:** ML enables adaptive robotic control and process automation.

**Core Methods:** Time Series Forecasting, Reinforcement Learning, Computer Vision.

---

### 5. Agriculture
- **Crop Yield Prediction:** Use satellite data and weather patterns to forecast yields.
- **Soil Monitoring:** Classify soil quality and nutrient composition.
- **Pest Detection:** Identify and classify pest infestations using image analysis.
- **Smart Irrigation:** Optimize water use with sensor-based predictive control.

**Techniques Used:** Decision Trees, CNNs, and IoT-integrated ML.

---

### 6. Transportation
- **Autonomous Vehicles:** Use deep reinforcement learning for path planning, control, and perception.
- **Traffic Prediction:** Forecast congestion and optimize routes using live data.
- **Fleet Management:** Optimize logistics and delivery schedules.

**Approaches:** Reinforcement Learning, Neural Networks, and Regression Analysis.

---

### 7. Energy and Environment
- **Power Load Forecasting:** Predict demand patterns for grid management.
- **Renewable Energy Optimization:** Optimize solar and wind systems through ML-driven control.
- **Climate Modeling:** Detect long-term environmental trends and anomalies.
- **Pollution Monitoring:** Predict and visualize air quality metrics.

**Methods:** Time-Series Models, Neural Networks, Gradient Boosting.

---

### 8. Education
- **Intelligent Tutoring Systems:** Adapt learning materials based on student progress.
- **Performance Prediction:** Identify struggling students early.
- **Automated Grading:** Grade assignments using NLP and text classification.
- **Curriculum Optimization:** Analyze learning outcomes to design better teaching strategies.

**Techniques:** NLP, Clustering, and Supervised Classification.

---

### 9. Entertainment and Media
- **Content Personalization:** Platforms like Spotify, YouTube, and Netflix use ML to tailor content.
- **Game AI:** Adaptive agents that learn player strategies.
- **Generative Models:** Create art, music, or synthetic media using GANs and Transformers.

**Techniques:** Reinforcement Learning, Deep Generative Models, NLP.

---

### 10. Security and Defense
- **Anomaly Detection:** Identify cyber threats or intrusions.
- **Facial Recognition:** Authenticate individuals in secure systems.
- **Behavioral Analytics:** Identify insider threats or suspicious actions.

**Methods:** CNNs, Autoencoders, and Ensemble Learning.

---

### 11. Government and Public Sector
- **Smart City Planning:** Analyze traffic, energy, and waste management systems.
- **Policy Modeling:** Predict outcomes of social or economic policies.
- **Fraud Detection:** Identify irregularities in taxation or benefits systems.

---

### 12. Emerging Areas
- **Legal Tech:** Predict case outcomes and assist with document discovery.
- **Healthcare Genomics:** Personalized treatment based on genetic sequencing.
- **Climate AI:** Forecast natural disasters and optimize resource management.
- **Ethical AI:** Monitor bias, fairness, and inclusivity in algorithmic systems.

---

### Summary
Machine learning has applications across every domain — enabling automation, improving decisions, and unlocking new possibilities in science, business, and daily life.`,
  keyTakeaways: [
    "Machine learning applications span healthcare, finance, agriculture, manufacturing, and more.",
    "Healthcare uses ML for diagnostics, drug discovery, and personalized medicine.",
    "Finance applies ML for fraud detection, credit scoring, and algorithmic trading.",
    "Retail leverages ML for recommendations, demand forecasting, and customer segmentation.",
    "Manufacturing uses predictive maintenance and quality control powered by ML.",
    "Agriculture benefits from yield prediction, pest detection, and irrigation optimization.",
    "Transportation integrates ML for autonomous vehicles and route optimization.",
    "Energy management relies on ML for demand forecasting and renewable optimization.",
    "Education and entertainment use ML for personalization and intelligent tutoring.",
    "Emerging domains like legal tech and ethical AI showcase ML’s evolving impact."
  ],
  quiz: [
    {
      question: "### What is a key use of ML in healthcare?",
      options: [
        "Predicting stock prices",
        "Diagnosing diseases from images and patient data",
        "Designing mobile apps",
        "Rendering 3D graphics"
      ],
      correct: 1
    },
    {
      question: "### Which ML application is common in finance?",
      options: [
        "Predicting molecular reactions",
        "Fraud detection and risk scoring",
        "Autonomous vehicle control",
        "Personalized tutoring systems"
      ],
      correct: 1
    },
    {
      question: "### Which ML technique powers product recommendation systems?",
      options: [
        "Clustering",
        "Regression Trees",
        "Collaborative Filtering",
        "Generative Adversarial Networks"
      ],
      correct: 2
    },
    {
      question: "### What is predictive maintenance?",
      options: [
        "Forecasting system failures before they happen",
        "Building new machines automatically",
        "Cleaning manufacturing data",
        "Increasing employee productivity"
      ],
      correct: 0
    },
    {
      question: "### How is ML used in agriculture?",
      options: [
        "Crop yield forecasting and pest detection",
        "Credit scoring",
        "Facial recognition",
        "Text summarization"
      ],
      correct: 0
    },
    {
      question: "### Which ML area powers autonomous vehicles?",
      options: [
        "Reinforcement Learning",
        "Clustering",
        "Regression Analysis",
        "Semi-Supervised Learning"
      ],
      correct: 0
    },
    {
      question: "### What is a typical ML use case in the energy sector?",
      options: [
        "Predicting power demand and optimizing grid control",
        "Image classification of solar panels",
        "Chatbot development",
        "Legal document summarization"
      ],
      correct: 0
    },
    {
      question: "### How does ML enhance education?",
      options: [
        "Through intelligent tutoring and personalized learning systems",
        "By generating random quizzes",
        "By detecting malware in school networks",
        "By creating new languages"
      ],
      correct: 0
    },
    {
      question: "### What ML methods are common in entertainment?",
      options: [
        "Reinforcement learning and deep generative models",
        "Clustering and K-means only",
        "Simple linear regression only",
        "Data encryption techniques"
      ],
      correct: 0
    },
    {
      question: "### What is a common ML technique for fraud detection?",
      options: [
        "Decision Trees and Neural Networks",
        "Convolutional Layers",
        "Text Embeddings",
        "K-means Clustering only"
      ],
      correct: 0
    },
    {
      question: "### Which ML approach is used for climate prediction and disaster forecasting?",
      options: [
        "Time-Series Forecasting and Deep Learning",
        "Clustering and PCA",
        "Binary Classification only",
        "Monte Carlo Simulation only"
      ],
      correct: 0
    },
    {
      question: "### In retail, how is ML used to prevent customer churn?",
      options: [
        "Predicting which customers are likely to leave",
        "Detecting financial fraud",
        "Reinforcing warehouse robots",
        "Encrypting user data"
      ],
      correct: 0
    },
    {
      question: "### Which ML concept supports robotics and adaptive control?",
      options: [
        "Reinforcement Learning",
        "Natural Language Processing",
        "Transfer Learning",
        "Causal Inference"
      ],
      correct: 0
    }
  ]
},
{
  id: 6,
  title: "Software Tools for Machine Learning",
  pages: "78–89",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
Machine Learning (ML) development depends heavily on powerful **software tools, libraries, and environments** that simplify data processing, model training, and deployment.  
These tools range from general-purpose programming libraries to specialized ML frameworks that provide **ready-to-use algorithms, visualization utilities, and model management features**.

---

### 1. Python Ecosystem
Python is the **most popular programming language** for machine learning because of its simplicity, large community, and extensive libraries.

#### Key Libraries:
- **NumPy:** Provides efficient numerical computations and array operations.
- **Pandas:** For data manipulation and preprocessing using DataFrames.
- **Matplotlib & Seaborn:** Visualization tools for data insights and model performance plots.
- **SciPy:** Supports scientific and statistical computing.

---

### 2. Scikit-learn
- One of the most widely used ML libraries for **classical algorithms**.
- Features include:
  - Regression, classification, clustering, dimensionality reduction
  - Model evaluation and cross-validation tools
  - Pipelines for preprocessing and training
- Ideal for **beginners** and **prototyping** standard ML workflows.

---

### 3. TensorFlow
- Developed by **Google**, TensorFlow is a powerful framework for **deep learning** and **neural networks**.
- Features:
  - GPU/TPU acceleration
  - Computational graph-based execution
  - Large-scale distributed training
  - TensorBoard for visualization
- Ideal for **research and production deployment**.

---

### 4. Keras
- A **high-level neural network API** built on top of TensorFlow (and previously Theano).
- Provides simple abstractions for:
  - Defining deep models with minimal code
  - Layer-based model design
  - Model saving and loading
- Popular for its **ease of use** and **rapid prototyping** capabilities.

---

### 5. PyTorch
- Developed by **Facebook’s AI Research (FAIR)**.
- A deep learning framework with **dynamic computation graphs**, making it flexible for experimentation.
- Features:
  - Easy debugging
  - Autograd for automatic differentiation
  - Strong community and integration with Hugging Face
- Used extensively in **research**, **NLP**, and **computer vision**.

---

### 6. Weka
- A **Java-based ML platform** with a **graphical user interface (GUI)**.
- Supports:
  - Data preprocessing
  - Classification, clustering, and association rule mining
  - Visual analysis
- Suitable for users with limited programming background.

---

### 7. RapidMiner
- A **drag-and-drop data science platform** for predictive analytics.
- Features:
  - Data preparation and model deployment tools
  - Visual workflow design
  - Integration with Python and R
- Ideal for **enterprise-level ML applications**.

---

### 8. Orange
- **Open-source visual programming** tool for machine learning and data mining.
- Modules for:
  - Classification
  - Regression
  - Clustering
  - Visualization
- Perfect for **educational use** and **quick experimentation**.

---

### 9. Jupyter Notebook
- An interactive development environment (IDE) for **data exploration, code, and documentation**.
- Supports live code execution and rich media outputs.
- Widely used in research, tutorials, and ML model prototyping.

---

### 10. Google Colab
- Cloud-based version of Jupyter that provides **free GPU/TPU access**.
- Useful for training deep learning models without local setup.
- Integrated with Google Drive for data management.

---

### 11. MLflow
- An open-source platform for **managing the ML lifecycle**.
- Supports:
  - Experiment tracking
  - Model versioning
  - Deployment pipelines
- Works with TensorFlow, PyTorch, and Scikit-learn.

---

### 12. Hugging Face
- Provides pre-trained models and APIs for **NLP and transformer-based architectures**.
- Integrates with PyTorch and TensorFlow.
- Supports text generation, summarization, translation, and more.

---

### Summary
These tools streamline the entire ML pipeline — from data preprocessing to model deployment.  
Choosing the right tool depends on **task complexity**, **team expertise**, and **deployment environment**.`,
  keyTakeaways: [
    "Python dominates ML development due to its simplicity and robust library ecosystem.",
    "Scikit-learn is ideal for traditional ML algorithms and rapid prototyping.",
    "TensorFlow and PyTorch are the two most popular deep learning frameworks.",
    "Keras simplifies model creation with user-friendly abstractions.",
    "Weka, RapidMiner, and Orange offer GUI-based workflows for non-programmers.",
    "Jupyter and Google Colab provide interactive, notebook-style development environments.",
    "MLflow enables experiment tracking and model management for reproducibility.",
    "Hugging Face simplifies NLP development with pre-trained transformer models.",
    "Selecting the right tool depends on scalability, performance, and usability needs."
  ],
  quiz: [
    {
      question: "### Why is Python preferred for machine learning?",
      options: [
        "It is faster than C++",
        "It has a rich ecosystem of ML libraries and simplicity of syntax",
        "It requires less memory",
        "It doesn’t need dependencies"
      ],
      correct: 1
    },
    {
      question: "### Which library is most commonly used for numerical computation in ML?",
      options: [
        "NumPy",
        "TensorFlow",
        "Pandas",
        "Weka"
      ],
      correct: 0
    },
    {
      question: "### Scikit-learn is best suited for which type of tasks?",
      options: [
        "Deep neural networks",
        "Traditional ML algorithms like regression and classification",
        "Speech recognition",
        "Autonomous robotics"
      ],
      correct: 1
    },
    {
      question: "### What is TensorFlow primarily used for?",
      options: [
        "Statistical visualization",
        "Deep learning and distributed neural network training",
        "Database management",
        "Web development"
      ],
      correct: 1
    },
    {
      question: "### What makes PyTorch popular among researchers?",
      options: [
        "It is GUI-based",
        "It provides dynamic computation graphs and easy debugging",
        "It is built for small datasets only",
        "It has no GPU support"
      ],
      correct: 1
    },
    {
      question: "### Which framework provides a drag-and-drop interface for ML workflows?",
      options: [
        "Scikit-learn",
        "RapidMiner",
        "TensorFlow",
        "Hugging Face"
      ],
      correct: 1
    },
    {
      question: "### Which tool is ideal for non-programmers and educational use?",
      options: [
        "Orange",
        "PyTorch",
        "NumPy",
        "Keras"
      ],
      correct: 0
    },
    {
      question: "### What is the primary benefit of Jupyter Notebook?",
      options: [
        "It allows code execution, visualization, and documentation in one place",
        "It compiles code faster than Python",
        "It’s only for deep learning",
        "It stores data in the cloud"
      ],
      correct: 0
    },
    {
      question: "### What does MLflow specialize in?",
      options: [
        "Hyperparameter tuning",
        "Experiment tracking and model lifecycle management",
        "Visualization of datasets",
        "Cloud hosting"
      ],
      correct: 1
    },
    {
      question: "### Which tool provides free GPU access for model training?",
      options: [
        "Weka",
        "Google Colab",
        "Scikit-learn",
        "Orange"
      ],
      correct: 1
    },
    {
      question: "### What is Hugging Face mainly known for?",
      options: [
        "Building decision trees",
        "Providing NLP transformer models and APIs",
        "Optimizing GPU usage",
        "Database connectivity"
      ],
      correct: 1
    },
    {
      question: "### Which tool visualizes TensorFlow training performance?",
      options: [
        "TensorBoard",
        "Seaborn",
        "RapidMiner",
        "Keras Dashboard"
      ],
      correct: 0
    },
    {
      question: "### Which ML library is built in Java and offers a GUI for beginners?",
      options: [
        "Weka",
        "Scikit-learn",
        "PyTorch",
        "Keras"
      ],
      correct: 0
    }
  ]
},
{
  id: 6,
  title: "Software Tools for Machine Learning",
  pages: "78–89",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
Machine Learning (ML) development depends heavily on powerful **software tools, libraries, and environments** that simplify data processing, model training, and deployment.  
These tools range from general-purpose programming libraries to specialized ML frameworks that provide **ready-to-use algorithms, visualization utilities, and model management features**.

---

### 1. Python Ecosystem
Python is the **most popular programming language** for machine learning because of its simplicity, large community, and extensive libraries.

#### Key Libraries:
- **NumPy:** Provides efficient numerical computations and array operations.
- **Pandas:** For data manipulation and preprocessing using DataFrames.
- **Matplotlib & Seaborn:** Visualization tools for data insights and model performance plots.
- **SciPy:** Supports scientific and statistical computing.

---

### 2. Scikit-learn
- One of the most widely used ML libraries for **classical algorithms**.
- Features include:
  - Regression, classification, clustering, dimensionality reduction
  - Model evaluation and cross-validation tools
  - Pipelines for preprocessing and training
- Ideal for **beginners** and **prototyping** standard ML workflows.

---

### 3. TensorFlow
- Developed by **Google**, TensorFlow is a powerful framework for **deep learning** and **neural networks**.
- Features:
  - GPU/TPU acceleration
  - Computational graph-based execution
  - Large-scale distributed training
  - TensorBoard for visualization
- Ideal for **research and production deployment**.

---

### 4. Keras
- A **high-level neural network API** built on top of TensorFlow (and previously Theano).
- Provides simple abstractions for:
  - Defining deep models with minimal code
  - Layer-based model design
  - Model saving and loading
- Popular for its **ease of use** and **rapid prototyping** capabilities.

---

### 5. PyTorch
- Developed by **Facebook’s AI Research (FAIR)**.
- A deep learning framework with **dynamic computation graphs**, making it flexible for experimentation.
- Features:
  - Easy debugging
  - Autograd for automatic differentiation
  - Strong community and integration with Hugging Face
- Used extensively in **research**, **NLP**, and **computer vision**.

---

### 6. Weka
- A **Java-based ML platform** with a **graphical user interface (GUI)**.
- Supports:
  - Data preprocessing
  - Classification, clustering, and association rule mining
  - Visual analysis
- Suitable for users with limited programming background.

---

### 7. RapidMiner
- A **drag-and-drop data science platform** for predictive analytics.
- Features:
  - Data preparation and model deployment tools
  - Visual workflow design
  - Integration with Python and R
- Ideal for **enterprise-level ML applications**.

---

### 8. Orange
- **Open-source visual programming** tool for machine learning and data mining.
- Modules for:
  - Classification
  - Regression
  - Clustering
  - Visualization
- Perfect for **educational use** and **quick experimentation**.

---

### 9. Jupyter Notebook
- An interactive development environment (IDE) for **data exploration, code, and documentation**.
- Supports live code execution and rich media outputs.
- Widely used in research, tutorials, and ML model prototyping.

---

### 10. Google Colab
- Cloud-based version of Jupyter that provides **free GPU/TPU access**.
- Useful for training deep learning models without local setup.
- Integrated with Google Drive for data management.

---

### 11. MLflow
- An open-source platform for **managing the ML lifecycle**.
- Supports:
  - Experiment tracking
  - Model versioning
  - Deployment pipelines
- Works with TensorFlow, PyTorch, and Scikit-learn.

---

### 12. Hugging Face
- Provides pre-trained models and APIs for **NLP and transformer-based architectures**.
- Integrates with PyTorch and TensorFlow.
- Supports text generation, summarization, translation, and more.

---

### Summary
These tools streamline the entire ML pipeline — from data preprocessing to model deployment.  
Choosing the right tool depends on **task complexity**, **team expertise**, and **deployment environment**.`,
  keyTakeaways: [
    "Python dominates ML development due to its simplicity and robust library ecosystem.",
    "Scikit-learn is ideal for traditional ML algorithms and rapid prototyping.",
    "TensorFlow and PyTorch are the two most popular deep learning frameworks.",
    "Keras simplifies model creation with user-friendly abstractions.",
    "Weka, RapidMiner, and Orange offer GUI-based workflows for non-programmers.",
    "Jupyter and Google Colab provide interactive, notebook-style development environments.",
    "MLflow enables experiment tracking and model management for reproducibility.",
    "Hugging Face simplifies NLP development with pre-trained transformer models.",
    "Selecting the right tool depends on scalability, performance, and usability needs."
  ],
  quiz: [
    {
      question: "### Why is Python preferred for machine learning?",
      options: [
        "It is faster than C++",
        "It has a rich ecosystem of ML libraries and simplicity of syntax",
        "It requires less memory",
        "It doesn’t need dependencies"
      ],
      correct: 1
    },
    {
      question: "### Which library is most commonly used for numerical computation in ML?",
      options: [
        "NumPy",
        "TensorFlow",
        "Pandas",
        "Weka"
      ],
      correct: 0
    },
    {
      question: "### Scikit-learn is best suited for which type of tasks?",
      options: [
        "Deep neural networks",
        "Traditional ML algorithms like regression and classification",
        "Speech recognition",
        "Autonomous robotics"
      ],
      correct: 1
    },
    {
      question: "### What is TensorFlow primarily used for?",
      options: [
        "Statistical visualization",
        "Deep learning and distributed neural network training",
        "Database management",
        "Web development"
      ],
      correct: 1
    },
    {
      question: "### What makes PyTorch popular among researchers?",
      options: [
        "It is GUI-based",
        "It provides dynamic computation graphs and easy debugging",
        "It is built for small datasets only",
        "It has no GPU support"
      ],
      correct: 1
    },
    {
      question: "### Which framework provides a drag-and-drop interface for ML workflows?",
      options: [
        "Scikit-learn",
        "RapidMiner",
        "TensorFlow",
        "Hugging Face"
      ],
      correct: 1
    },
    {
      question: "### Which tool is ideal for non-programmers and educational use?",
      options: [
        "Orange",
        "PyTorch",
        "NumPy",
        "Keras"
      ],
      correct: 0
    },
    {
      question: "### What is the primary benefit of Jupyter Notebook?",
      options: [
        "It allows code execution, visualization, and documentation in one place",
        "It compiles code faster than Python",
        "It’s only for deep learning",
        "It stores data in the cloud"
      ],
      correct: 0
    },
    {
      question: "### What does MLflow specialize in?",
      options: [
        "Hyperparameter tuning",
        "Experiment tracking and model lifecycle management",
        "Visualization of datasets",
        "Cloud hosting"
      ],
      correct: 1
    },
    {
      question: "### Which tool provides free GPU access for model training?",
      options: [
        "Weka",
        "Google Colab",
        "Scikit-learn",
        "Orange"
      ],
      correct: 1
    },
    {
      question: "### What is Hugging Face mainly known for?",
      options: [
        "Building decision trees",
        "Providing NLP transformer models and APIs",
        "Optimizing GPU usage",
        "Database connectivity"
      ],
      correct: 1
    },
    {
      question: "### Which tool visualizes TensorFlow training performance?",
      options: [
        "TensorBoard",
        "Seaborn",
        "RapidMiner",
        "Keras Dashboard"
      ],
      correct: 0
    },
    {
      question: "### Which ML library is built in Java and offers a GUI for beginners?",
      options: [
        "Weka",
        "Scikit-learn",
        "PyTorch",
        "Keras"
      ],
      correct: 0
    }
  ]
},
{
  id: 7,
  title: "Supervised Learning and Regression",
  pages: "90–151",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**Supervised Learning** is one of the most fundamental paradigms in machine learning.  
It uses **labeled data** — examples where both inputs (features) and desired outputs (labels) are known — to learn a mapping function from inputs to outputs.  
The trained model can then **predict outputs for new, unseen data**.

Regression is a major branch of supervised learning focused on **predicting continuous numerical values**.

---

### 1. Principles of Supervised Learning
- The model is trained using pairs of input data \\( (x_i, y_i) \\).
- The objective is to minimize the difference between **predicted output** and **actual output**.
- Supervised learning tasks are divided into:
  - **Regression:** Predicting continuous values.
  - **Classification:** Predicting discrete categories.

#### General Supervised Learning Workflow:
1. Collect and label dataset.
2. Split data into **training**, **validation**, and **testing** sets.
3. Choose model/algorithm.
4. Train the model using training data.
5. Evaluate using performance metrics.
6. Deploy and monitor.

---

### 2. Regression Analysis
Regression estimates the **relationship between dependent (target) and independent (predictor) variables**.

#### Types of Regression:
1. **Linear Regression**
   - Models a straight-line relationship:
     \\[
     y = \\beta_0 + \\beta_1x + \\epsilon
     \\]
   - Used for trend estimation and forecasting.
   - Assumes linearity and independence.

2. **Multiple Linear Regression**
   - Uses multiple predictors:
     \\[
     y = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n + \\epsilon
     \\]

3. **Polynomial Regression**
   - Adds polynomial terms to capture curvature:
     \\[
     y = \\beta_0 + \\beta_1x + \\beta_2x^2 + ... + \\beta_nx^n + \\epsilon
     \\]
   - Useful for non-linear patterns.

4. **Ridge Regression**
   - Adds **L2 regularization** to reduce overfitting by penalizing large coefficients.

5. **LASSO Regression**
   - Uses **L1 regularization**, which can drive some coefficients to zero — performing **feature selection**.

6. **Elastic Net**
   - A hybrid of L1 and L2 regularization.

---

### 3. Assumptions of Linear Regression
- Linearity between variables  
- Homoscedasticity (constant variance of errors)  
- Independence of errors  
- Normal distribution of residuals  
- No multicollinearity among predictors  

Violations of these assumptions can lead to inaccurate or biased models.

---

### 4. Evaluation Metrics for Regression
- **Mean Squared Error (MSE):**
  \\[
  MSE = \\frac{1}{n} \\sum (y_i - \\hat{y_i})^2
  \\]
- **Root Mean Squared Error (RMSE):** \\( \\sqrt{MSE} \\)
- **Mean Absolute Error (MAE):**
  \\[
  MAE = \\frac{1}{n} \\sum |y_i - \\hat{y_i}|
  \\]
- **R-squared (Coefficient of Determination):**
  \\[
  R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}
  \\]
  Measures the proportion of variance explained by the model.

---

### 5. Regularization
Regularization techniques (Ridge, LASSO, Elastic Net) help prevent **overfitting** by constraining coefficient magnitude.  
They improve **model generalization** to unseen data.

---

### 6. Gradient Descent in Regression
Regression models often minimize a **loss function (MSE)** using **gradient descent**:
1. Compute gradient of loss with respect to parameters.
2. Update parameters:
   \\[
   \\theta := \\theta - \\alpha \\cdot \\nabla J(\\theta)
   \\]
   where \\( \\alpha \\) is the **learning rate**.

---

### 7. Applications of Regression
- **Predicting house prices**
- **Forecasting stock values**
- **Modeling population growth**
- **Estimating demand or sales**
- **Predicting temperature or pollution**

---

### 8. Limitations
- Sensitive to outliers.
- Assumes linear relationships.
- May not capture complex, non-linear interactions unless extended (e.g., polynomial regression).
- Requires good data preprocessing and feature scaling.

---

### Summary
Regression analysis provides interpretable and efficient models for prediction, trend estimation, and forecasting.  
Regularization and careful validation are crucial for achieving **robust and generalizable** results.`,
  keyTakeaways: [
    "Supervised learning uses labeled data to map inputs to outputs.",
    "Regression predicts continuous numerical values; classification predicts categories.",
    "Linear regression assumes linear relationships between variables.",
    "Ridge and LASSO regression help prevent overfitting via regularization.",
    "Polynomial regression models non-linear patterns by adding power terms.",
    "MSE, RMSE, MAE, and R² are core regression evaluation metrics.",
    "Gradient descent is commonly used to minimize regression loss functions.",
    "Regression is widely used in forecasting and quantitative analysis."
  ],
  quiz: [
    {
      question: "### What distinguishes supervised learning from unsupervised learning?",
      options: [
        "Supervised learning uses labeled data, while unsupervised uses unlabeled data.",
        "Supervised learning uses clustering, while unsupervised uses regression.",
        "Supervised learning requires reinforcement signals.",
        "Supervised learning cannot generalize to new data."
      ],
      correct: 0
    },
    {
      question: "### What does regression aim to predict?",
      options: [
        "Discrete categories",
        "Continuous numerical values",
        "Cluster labels",
        "Reinforcement rewards"
      ],
      correct: 1
    },
    {
      question: "### In linear regression, what does the slope (β₁) represent?",
      options: [
        "The intercept",
        "The rate of change in y for a unit change in x",
        "The residual error",
        "The variance of the model"
      ],
      correct: 1
    },
    {
      question: "### Which regression technique adds an L2 penalty to reduce overfitting?",
      options: [
        "LASSO Regression",
        "Ridge Regression",
        "Polynomial Regression",
        "Elastic Net"
      ],
      correct: 1
    },
    {
      question: "### What is the main advantage of LASSO regression?",
      options: [
        "It removes all bias automatically",
        "It performs feature selection by shrinking some coefficients to zero",
        "It works only for categorical outputs",
        "It has no regularization penalty"
      ],
      correct: 1
    },
    {
      question: "### Which metric measures the average magnitude of errors regardless of direction?",
      options: [
        "R²",
        "MAE (Mean Absolute Error)",
        "RMSE",
        "Variance"
      ],
      correct: 1
    },
    {
      question: "### What does R-squared indicate?",
      options: [
        "Model accuracy for classification tasks",
        "Proportion of variance in the dependent variable explained by the model",
        "The sum of squared residuals",
        "Learning rate effectiveness"
      ],
      correct: 1
    },
    {
      question: "### What is the purpose of regularization?",
      options: [
        "To increase training accuracy regardless of overfitting",
        "To prevent overfitting by penalizing large coefficients",
        "To reduce dataset size",
        "To automatically tune learning rates"
      ],
      correct: 1
    },
    {
      question: "### What does gradient descent minimize in regression models?",
      options: [
        "Entropy",
        "Loss function (e.g., MSE)",
        "Correlation coefficient",
        "Regularization constant"
      ],
      correct: 1
    },
    {
      question: "### Which regression type can model curved relationships?",
      options: [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "LASSO Regression"
      ],
      correct: 1
    },
    {
      question: "### What is a limitation of linear regression?",
      options: [
        "It captures non-linear interactions perfectly.",
        "It assumes linearity and can be sensitive to outliers.",
        "It doesn’t require any labeled data.",
        "It automatically removes multicollinearity."
      ],
      correct: 1
    },
    {
      question: "### Which learning algorithm is NOT a regression type?",
      options: [
        "Linear Regression",
        "Logistic Regression",
        "Polynomial Regression",
        "Random Forest Classification"
      ],
      correct: 3
    },
    {
      question: "### Which of these represents the correct formula for Mean Squared Error (MSE)?",
      options: [
        "MSE = Σ(y_i - ŷ_i) / n",
        "MSE = Σ(y_i - ŷ_i)² / n",
        "MSE = √Σ(y_i - ŷ_i)",
        "MSE = log(y_i - ŷ_i)"
      ],
      correct: 1
    }
  ]
},
{
  id: 8,
  title: "Optimization and Gradient Descent",
  pages: "166–187",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**Optimization** is the mathematical foundation of model training in Machine Learning (ML).  
It focuses on finding the **best set of parameters (weights)** that minimize the model’s **loss function** — a measure of prediction error.  
The most widely used optimization method in ML is **Gradient Descent**, which iteratively adjusts model parameters to reduce loss.

---

### 1. Objective and Loss Functions
- The **objective function** defines what the algorithm aims to minimize or maximize.
- In supervised learning, it’s typically a **loss function** measuring prediction error.

#### Common Loss Functions:
- **Mean Squared Error (MSE)** for regression
- **Cross-Entropy Loss** for classification
- **Hinge Loss** for SVMs
- **Log Loss** for logistic regression

The optimization process aims to find parameter values that **minimize these loss functions**.

---

### 2. Gradient Descent — The Core Idea
Gradient Descent finds the minimum of a function by **moving in the direction of the negative gradient**.

#### Update Rule:
\\[
\\theta := \\theta - \\alpha \\cdot \\nabla J(\\theta)
\\]
Where:
- \\( \\theta \\): model parameters  
- \\( \\alpha \\): learning rate  
- \\( \\nabla J(\\theta) \\): gradient (partial derivatives of the loss function)

Each step reduces the error slightly until convergence (minimum point).

---

### 3. Learning Rate (α)
- Controls how large the parameter updates are.
- **Too small:** Slow convergence.  
- **Too large:** Divergence (overshooting the minimum).  
- Proper tuning is essential for stable learning.

---

### 4. Types of Gradient Descent
#### 1. Batch Gradient Descent
- Uses the **entire dataset** to compute the gradient.
- Stable but computationally expensive for large datasets.

#### 2. Stochastic Gradient Descent (SGD)
- Updates weights for **each training example** individually.
- Faster and more scalable but introduces noise (oscillations).

#### 3. Mini-Batch Gradient Descent
- Compromise between batch and stochastic.
- Uses **small random subsets (batches)** of data for each update.
- Most commonly used in practice.

---

### 5. Gradient Descent Variants
#### 1. Momentum
- Adds a velocity term that **smooths updates** and avoids oscillation.
- Speeds up convergence in valleys of the cost surface.

#### 2. Nesterov Accelerated Gradient (NAG)
- Improves momentum by making a **“look-ahead” correction** before applying updates.

#### 3. Adagrad
- Adapts learning rate individually for each parameter.
- Works well for sparse data, but learning rate may decay too quickly.

#### 4. RMSProp
- Fixes Adagrad’s decaying issue by using **exponential moving averages** of squared gradients.

#### 5. Adam (Adaptive Moment Estimation)
- Combines Momentum + RMSProp.
- Maintains running averages of gradients and squared gradients.
- Default optimizer for most deep learning tasks.

---

### 6. Cost Surface and Convergence
- The **cost surface** represents the relationship between parameters and loss.
- Gradient Descent moves “downhill” along this surface.
- Challenges:
  - **Local minima:** May trap models in suboptimal points.
  - **Saddle points:** Gradients close to zero but not true minima.
  - **Plateaus:** Slow learning regions.

---

### 7. Learning Rate Scheduling
Learning rates can be **dynamically adjusted** during training:
- Step decay  
- Exponential decay  
- Cyclical learning rates  
- Adaptive learning rate schedulers (Adam, RMSProp handle this internally)

---

### 8. Regularization in Optimization
Regularization (L1, L2) introduces penalty terms in the loss function:
\\[
J'(\\theta) = J(\\theta) + \\lambda \\sum ||\\theta||^p
\\]
This discourages overfitting and improves generalization.

---

### 9. Practical Considerations
- Normalize and scale features for stable convergence.
- Use random weight initialization to break symmetry.
- Monitor **training and validation loss** to detect overfitting.
- Combine optimizers with regularization and dropout in deep learning.

---

### Summary
Gradient Descent and its variants power nearly all ML optimization processes.  
By iteratively minimizing loss, they enable models to learn efficiently and generalize better.`,
  keyTakeaways: [
    "Optimization minimizes loss to improve model accuracy.",
    "Gradient Descent updates parameters in the direction of negative gradients.",
    "Learning rate determines the step size during parameter updates.",
    "Mini-Batch Gradient Descent balances efficiency and stability.",
    "Momentum and NAG help accelerate convergence and reduce oscillations.",
    "Adam optimizer combines momentum and RMSProp for adaptive learning.",
    "Feature scaling and normalization improve optimization stability.",
    "Loss functions define what the model learns to minimize (e.g., MSE, Cross-Entropy).",
    "Learning rate scheduling dynamically adjusts training pace for better results.",
    "Regularization improves model generalization during optimization."
  ],
  quiz: [
    {
      question: "### What is the main goal of optimization in ML?",
      options: [
        "To minimize the model’s loss function",
        "To increase dataset size",
        "To visualize training results",
        "To reduce computation speed"
      ],
      correct: 0
    },
    {
      question: "### In Gradient Descent, what does the learning rate (α) control?",
      options: [
        "The number of neurons in a model",
        "The step size of parameter updates",
        "The dataset shuffle frequency",
        "The model’s regularization strength"
      ],
      correct: 1
    },
    {
      question: "### What happens if the learning rate is too large?",
      options: [
        "The model converges faster without error",
        "The updates may overshoot the minimum and diverge",
        "Training becomes more stable",
        "Gradients vanish completely"
      ],
      correct: 1
    },
    {
      question: "### Which type of Gradient Descent uses all data for each update?",
      options: [
        "Stochastic Gradient Descent",
        "Batch Gradient Descent",
        "Mini-Batch Gradient Descent",
        "Momentum Gradient Descent"
      ],
      correct: 1
    },
    {
      question: "### What is the key advantage of Mini-Batch Gradient Descent?",
      options: [
        "It avoids the need for backpropagation",
        "It balances computational efficiency and convergence stability",
        "It doesn’t require feature scaling",
        "It always finds the global minimum"
      ],
      correct: 1
    },
    {
      question: "### What does the Adam optimizer combine?",
      options: [
        "Ridge and LASSO Regression",
        "Momentum and RMSProp",
        "SGD and Batch Normalization",
        "Dropout and Weight Decay"
      ],
      correct: 1
    },
    {
      question: "### What problem do Momentum and Nesterov Accelerated Gradient solve?",
      options: [
        "Overfitting",
        "Oscillations and slow convergence",
        "Data imbalance",
        "Feature redundancy"
      ],
      correct: 1
    },
    {
      question: "### Which optimizer adapts learning rates for each parameter?",
      options: [
        "Adagrad",
        "Ridge",
        "Gradient Clipping",
        "Polynomial Regression"
      ],
      correct: 0
    },
    {
      question: "### What are saddle points in optimization?",
      options: [
        "Flat regions where gradients are nearly zero but not minima",
        "The optimal points with minimum loss",
        "Regions of rapid convergence",
        "Overfitted model regions"
      ],
      correct: 0
    },
    {
      question: "### What is the function of regularization in optimization?",
      options: [
        "To reduce model overfitting by penalizing large weights",
        "To increase learning rate automatically",
        "To visualize cost surfaces",
        "To randomize gradient direction"
      ],
      correct: 0
    },
    {
      question: "### Why is feature scaling important before optimization?",
      options: [
        "To prevent variable dominance and improve convergence speed",
        "To reduce dataset size",
        "To simplify the model architecture",
        "To ensure integer-based weight updates"
      ],
      correct: 0
    },
    {
      question: "### Which loss function is commonly used for regression?",
      options: [
        "Cross-Entropy Loss",
        "Mean Squared Error (MSE)",
        "Hinge Loss",
        "Log Loss"
      ],
      correct: 1
    },
    {
      question: "### What is the primary issue with a very small learning rate?",
      options: [
        "It may skip minima",
        "It leads to very slow convergence",
        "It increases variance",
        "It prevents weight updates"
      ],
      correct: 1
    }
  ]
},

{
  id: 9,
  title: "Decision Trees",
  pages: "196–212",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**Decision Trees** are supervised learning models that represent decisions and their possible outcomes as a **tree-like structure**.  
They are used for both **classification** and **regression** tasks and are valued for their **interpretability and simplicity**.

A Decision Tree partitions the dataset into smaller subsets based on feature conditions, forming a hierarchical structure of decisions leading to predictions.

---

### 1. Structure of a Decision Tree
- **Root Node:** Represents the entire dataset (no split yet).
- **Internal Nodes:** Represent decisions based on features.
- **Branches:** Represent outcomes of tests (conditions).
- **Leaf Nodes (Terminal Nodes):** Represent final predictions or outputs.

Each split aims to create subsets that are as **pure as possible** (homogeneous in terms of class labels).

---

### 2. Decision Tree Algorithms
Popular algorithms for building trees include:
- **ID3 (Iterative Dichotomiser 3):** Uses *information gain* based on **entropy**.
- **C4.5:** Extension of ID3; handles continuous attributes and pruning.
- **CART (Classification and Regression Trees):** Uses *Gini index* for classification and *MSE* for regression.

---

### 3. Key Concepts

#### a. Entropy
Measures impurity or uncertainty in the dataset:
\\[
Entropy(S) = - \\sum p_i \\log_2(p_i)
\\]
Where \\( p_i \\) is the probability of class \\( i \\).

- **Low entropy:** More homogeneous (pure) dataset.
- **High entropy:** More heterogeneous (impure).

#### b. Information Gain
The reduction in entropy achieved by a split:
\\[
Gain(S, A) = Entropy(S) - \\sum \\frac{|S_v|}{|S|} Entropy(S_v)
\\]
Used to select the **best attribute** for splitting.

#### c. Gini Index
Used by CART to measure impurity:
\\[
Gini(S) = 1 - \\sum p_i^2
\\]

#### d. Gain Ratio
Used by C4.5 to correct Information Gain’s bias toward features with many values.

---

### 4. Tree Construction Process
1. Select the attribute with the highest Information Gain or lowest Gini index.  
2. Split the dataset based on that attribute.  
3. Repeat recursively on each subset until:
   - All samples in a node belong to one class.
   - Maximum depth is reached.
   - No improvement in purity occurs.

---

### 5. Pruning
- **Pre-Pruning (Early Stopping):** Stop tree growth early using criteria like max depth or min samples per node.
- **Post-Pruning:** Grow the full tree and then prune branches that don’t improve accuracy (reduces overfitting).

---

### 6. Handling Continuous and Categorical Features
- Continuous features are split by thresholding (e.g., “Age < 30”).
- Categorical features split by distinct values.

---

### 7. Advantages
- Easy to interpret and visualize.
- Handles both numerical and categorical data.
- Requires little data preprocessing.
- Can model non-linear relationships.

---

### 8. Disadvantages
- Prone to **overfitting**, especially with deep trees.
- Small changes in data can lead to different tree structures (high variance).
- Biased toward attributes with many distinct values (handled by gain ratio).

---

### 9. Decision Trees in Regression
- Regression trees use MSE as the impurity measure.
- Predictions are the **average of target values** in each leaf node.

---

### 10. Applications
- **Credit scoring**
- **Medical diagnosis**
- **Customer segmentation**
- **Stock market prediction**
- **Fraud detection**

---

### Summary
Decision Trees offer a powerful, interpretable framework for supervised learning.  
However, without pruning or ensemble techniques, they risk overfitting — leading to unstable performance on unseen data.`,
  keyTakeaways: [
    "Decision Trees split data recursively based on feature values to predict outcomes.",
    "Entropy and Information Gain measure data purity and guide splitting.",
    "CART uses the Gini index for impurity measurement.",
    "Pruning helps prevent overfitting by simplifying the model.",
    "Decision Trees handle both categorical and numerical data.",
    "They are interpretable but sensitive to data changes (high variance).",
    "Regression trees predict continuous values using MSE as a criterion.",
    "Decision Trees form the foundation of ensemble methods like Random Forests and Gradient Boosting."
  ],
  quiz: [
    {
      question: "### What is a Decision Tree primarily used for?",
      options: [
        "Unsupervised clustering",
        "Supervised classification and regression",
        "Feature scaling",
        "Dimensionality reduction"
      ],
      correct: 1
    },
    {
      question: "### What does the root node of a Decision Tree represent?",
      options: [
        "The final prediction result",
        "The entire dataset before any splits",
        "A single feature’s subset",
        "A leaf with maximum purity"
      ],
      correct: 1
    },
    {
      question: "### Which algorithm uses Information Gain based on Entropy?",
      options: [
        "CART",
        "ID3",
        "KNN",
        "Random Forest"
      ],
      correct: 1
    },
    {
      question: "### What is the formula for Entropy?",
      options: [
        "Entropy = Σ(p_i²)",
        "Entropy = -Σ(p_i log₂ p_i)",
        "Entropy = 1 - Σ(p_i²)",
        "Entropy = p_i - q_i"
      ],
      correct: 1
    },
    {
      question: "### What does Information Gain measure?",
      options: [
        "The increase in impurity after a split",
        "The reduction in entropy after splitting on an attribute",
        "The correlation between features",
        "The model’s overall accuracy"
      ],
      correct: 1
    },
    {
      question: "### What impurity measure does the CART algorithm use?",
      options: [
        "Entropy",
        "Gini Index",
        "Gain Ratio",
        "Variance Reduction"
      ],
      correct: 1
    },
    {
      question: "### What is the purpose of pruning in Decision Trees?",
      options: [
        "To improve model interpretability by removing redundant branches",
        "To increase training accuracy",
        "To add more layers to the tree",
        "To reduce data size"
      ],
      correct: 0
    },
    {
      question: "### Which type of pruning grows the full tree first?",
      options: [
        "Pre-Pruning",
        "Post-Pruning",
        "Gain Pruning",
        "Depth Pruning"
      ],
      correct: 1
    },
    {
      question: "### Why are Decision Trees prone to overfitting?",
      options: [
        "They use regularization by default",
        "They memorize the training data without pruning",
        "They ignore feature relationships",
        "They require normalization"
      ],
      correct: 1
    },
    {
      question: "### How do Decision Trees handle continuous features?",
      options: [
        "By splitting at threshold values (e.g., X < 30)",
        "By removing continuous data",
        "By rounding them to integers",
        "By converting them to text"
      ],
      correct: 0
    },
    {
      question: "### What is the prediction method for regression trees?",
      options: [
        "The average target value of the samples in a leaf node",
        "The majority class of the samples in a leaf node",
        "The highest information gain",
        "The Gini index of the branch"
      ],
      correct: 0
    },
    {
      question: "### Which metric helps choose the best attribute for a split?",
      options: [
        "Mean Absolute Error",
        "Information Gain",
        "Variance Score",
        "R-squared"
      ],
      correct: 1
    },
    {
      question: "### What is one major limitation of Decision Trees?",
      options: [
        "They cannot handle numeric data",
        "They are highly sensitive to small data changes",
        "They cannot perform regression",
        "They require unsupervised data"
      ],
      correct: 1
    }
  ]
},
{
  id: 10,
  title: "K-Nearest Neighbors (KNN) and Kernel Methods",
  pages: "212–226",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**K-Nearest Neighbors (KNN)** and **Kernel Methods** represent **non-parametric learning approaches** in machine learning.  
They do not assume an explicit functional form of the data but instead rely on **distance, similarity, or kernel functions** to make predictions.  
Both techniques are widely used for classification, regression, and pattern recognition.

---

### 1. K-Nearest Neighbors (KNN)
KNN is one of the simplest **instance-based** or **lazy learning** algorithms.  
It makes predictions based on the **majority label (classification)** or **average value (regression)** of the K nearest samples in the training data.

#### Algorithm Steps:
1. Choose the number of neighbors \\( K \\).
2. Compute the distance between the query point and all training examples.
3. Select the **K nearest neighbors** based on a distance metric.
4. Predict the label (classification) or mean value (regression) among those neighbors.

---

### 2. Distance Metrics
Common distance functions used in KNN:
- **Euclidean Distance:**  
  \\[
  d(x, y) = \\sqrt{\\sum_i (x_i - y_i)^2}
  \\]
- **Manhattan Distance:**  
  \\[
  d(x, y) = \\sum_i |x_i - y_i|
  \\]
- **Minkowski Distance:** Generalized form of Euclidean/Manhattan.
- **Cosine Similarity:** Measures angle-based similarity for high-dimensional data (e.g., text).

Choosing the right distance metric significantly impacts KNN’s accuracy.

---

### 3. Choosing the Value of K
- **Small K:** Model becomes sensitive to noise (overfitting).  
- **Large K:** Model becomes too smooth (underfitting).  
- **Common practice:** Choose K via **cross-validation**.

Odd K values are often preferred for binary classification to avoid ties.

---

### 4. Feature Scaling
Because KNN relies on distances, **features must be normalized or standardized** to prevent variables with large scales from dominating the distance computation.

---

### 5. Advantages of KNN
- Simple and intuitive.
- No training phase — predictions happen at query time.
- Works for both classification and regression.
- Naturally handles multi-class problems.

---

### 6. Disadvantages of KNN
- **Computationally expensive** for large datasets.
- Sensitive to irrelevant or highly correlated features.
- Requires feature scaling.
- Performance declines in high-dimensional spaces (curse of dimensionality).

---

### 7. Kernel Methods
Kernel methods enable **nonlinear learning** by implicitly mapping data into a higher-dimensional feature space.  
Instead of transforming data explicitly, a **kernel function** computes the similarity between two samples in that space.

#### Kernel Trick
The **kernel trick** allows inner product computation in a high-dimensional space **without explicitly transforming the data**:
\\[
K(x, x') = \\langle \\phi(x), \\phi(x') \\rangle
\\]
Where \\( \\phi \\) is a feature mapping function.

---

### 8. Common Kernel Functions
1. **Linear Kernel:**  
   \\( K(x, x') = x^T x' \\)
2. **Polynomial Kernel:**  
   \\( K(x, x') = (x^T x' + c)^d \\)
3. **Radial Basis Function (RBF) / Gaussian Kernel:**  
   \\( K(x, x') = e^{-\\frac{||x - x'||^2}{2\\sigma^2}} \\)
4. **Sigmoid Kernel:**  
   \\( K(x, x') = \\tanh(\\alpha x^T x' + c) \\)

---

### 9. Support Vector Machines (SVMs)
- SVMs are one of the most popular algorithms using kernel methods.  
- They find the **optimal separating hyperplane** that maximizes the margin between classes.  
- Kernels allow SVMs to handle **nonlinear decision boundaries**.

---

### 10. Advantages of Kernel Methods
- Handle complex, nonlinear relationships.
- Allow flexible decision boundaries.
- Powerful in high-dimensional feature spaces.
- Core of many ML algorithms beyond SVM (e.g., PCA, clustering).

---

### 11. Limitations
- Choosing the right kernel and hyperparameters (like \\( \\sigma \\), \\( C \\)) is challenging.
- Kernel methods scale poorly with large datasets (computationally heavy).
- Sensitive to noise and irrelevant features.

---

### 12. Applications
- **Image classification**
- **Text categorization**
- **Bioinformatics**
- **Face recognition**
- **Recommender systems**

---

### Summary
KNN and Kernel Methods provide powerful, flexible techniques for both linear and nonlinear learning problems.  
KNN is simple but computationally heavy, while kernel-based algorithms like SVM offer strong generalization capabilities for complex data.`,
  keyTakeaways: [
    "KNN is an instance-based learning algorithm relying on similarity or distance metrics.",
    "The choice of K greatly affects model performance — small K overfits, large K underfits.",
    "Feature scaling is essential in KNN since it’s distance-sensitive.",
    "Common distance metrics include Euclidean, Manhattan, and Cosine similarity.",
    "Kernel methods enable nonlinear learning by mapping data into high-dimensional feature spaces.",
    "The kernel trick allows inner product computation without explicit feature transformation.",
    "SVMs use kernels to find optimal hyperplanes for classification.",
    "RBF, Polynomial, and Linear kernels are commonly used.",
    "Both KNN and kernel methods can handle complex decision boundaries but are computationally demanding.",
    "Model performance depends heavily on hyperparameter tuning and scaling."
  ],
  quiz: [
    {
      question: "### What type of algorithm is KNN?",
      options: [
        "Parametric and model-based",
        "Non-parametric and instance-based",
        "Rule-based and linear",
        "Probabilistic and ensemble-based"
      ],
      correct: 1
    },
    {
      question: "### What does K represent in KNN?",
      options: [
        "Number of layers in the model",
        "Number of nearest neighbors used for prediction",
        "Number of clusters in the dataset",
        "Number of iterations for training"
      ],
      correct: 1
    },
    {
      question: "### What is the most common distance metric used in KNN?",
      options: [
        "Cosine Distance",
        "Euclidean Distance",
        "Jaccard Distance",
        "Hamming Distance"
      ],
      correct: 1
    },
    {
      question: "### What happens when K is too small?",
      options: [
        "The model becomes overly smooth (underfits)",
        "The model becomes too sensitive to noise (overfits)",
        "The model stops learning",
        "The dataset becomes linearly separable"
      ],
      correct: 1
    },
    {
      question: "### Why is feature scaling necessary in KNN?",
      options: [
        "To make sure all features contribute equally to distance calculations",
        "To speed up neural network convergence",
        "To reduce memory requirements",
        "To improve visualization clarity"
      ],
      correct: 0
    },
    {
      question: "### What is the key idea of the kernel trick?",
      options: [
        "Explicitly transforming data to higher dimensions",
        "Computing inner products in high-dimensional space without explicit mapping",
        "Reducing the dimensionality of data",
        "Eliminating nonlinearity from the model"
      ],
      correct: 1
    },
    {
      question: "### Which kernel is most widely used in nonlinear SVMs?",
      options: [
        "Linear Kernel",
        "RBF (Gaussian) Kernel",
        "Sigmoid Kernel",
        "Polynomial Kernel"
      ],
      correct: 1
    },
    {
      question: "### What does SVM aim to maximize?",
      options: [
        "Model accuracy",
        "Distance (margin) between separating hyperplanes",
        "Training speed",
        "Feature correlation"
      ],
      correct: 1
    },
    {
      question: "### Which of the following is NOT a valid kernel function?",
      options: [
        "RBF Kernel",
        "Polynomial Kernel",
        "Logarithmic Kernel",
        "Linear Kernel"
      ],
      correct: 2
    },
    {
      question: "### What is one disadvantage of KNN?",
      options: [
        "It cannot handle multiple classes",
        "It requires storing the entire dataset and is computationally heavy",
        "It cannot use distance metrics",
        "It only works with categorical data"
      ],
      correct: 1
    },
    {
      question: "### In high-dimensional spaces, KNN performance degrades due to?",
      options: [
        "Vanishing gradients",
        "Curse of dimensionality",
        "Regularization bias",
        "Data augmentation"
      ],
      correct: 1
    },
    {
      question: "### What is the function of the kernel in SVM?",
      options: [
        "To linearize data before training",
        "To measure similarity in transformed feature space",
        "To remove noise from the dataset",
        "To normalize features"
      ],
      correct: 1
    },
    {
      question: "### Which applications commonly use KNN or kernel methods?",
      options: [
        "Image recognition and text classification",
        "Database indexing",
        "Neural architecture search",
        "Genetic algorithms"
      ],
      correct: 0
    }
  ]
},
{
  id: 11,
  title: "Ensemble Learning (Bagging, Boosting, and Random Forests)",
  pages: "228–262",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**Ensemble Learning** combines multiple individual models (often called *weak learners*) to create a single **strong learner** that improves predictive performance and robustness.  
The key principle: **“Many weak models can outperform one strong model.”**

Ensemble methods reduce variance, bias, and overfitting by aggregating predictions from multiple models.

---

### 1. Types of Ensemble Methods
1. **Bagging (Bootstrap Aggregating)** — reduces variance.
2. **Boosting** — reduces bias.
3. **Stacking** — combines different models through a meta-learner.

---

### 2. Bagging (Bootstrap Aggregating)
#### Concept:
- Multiple models (e.g., Decision Trees) are trained on **random subsets of data sampled with replacement**.
- Predictions are **averaged (regression)** or **voted (classification)**.

#### Steps:
1. Generate multiple bootstrap samples from the dataset.  
2. Train a separate model on each sample.  
3. Combine predictions through majority voting or averaging.

#### Key Benefits:
- Reduces variance and overfitting.
- Improves model stability.
- Works best with high-variance, low-bias models (like Decision Trees).

---

### 3. Random Forests
**Random Forests** are an extension of bagging that introduces **feature randomness** in addition to data sampling.

#### Key Properties:
- Each tree uses a random subset of features for splitting.
- Aggregates predictions from many decorrelated Decision Trees.
- Works well for classification, regression, and feature importance ranking.

#### Advantages:
- Highly accurate and robust.
- Handles missing values and noisy data.
- Provides internal feature importance metrics.

#### Limitations:
- Less interpretable than single trees.
- Slower for real-time predictions due to multiple trees.

---

### 4. Boosting
Boosting combines multiple **weak learners sequentially**, where each new model focuses on **correcting errors** made by previous ones.

#### Core Idea:
Each model is trained on a weighted dataset emphasizing the misclassified samples from the previous round.

#### General Process:
1. Initialize model weights equally.  
2. Train base learner and calculate errors.  
3. Increase weights for misclassified samples.  
4. Train the next learner to focus on those difficult cases.  
5. Combine all learners using weighted voting or averaging.

---

### 5. AdaBoost (Adaptive Boosting)
- Each learner contributes to the final prediction with a **weighted influence** based on its accuracy.  
- Focuses iteratively on samples that previous models misclassified.  
- Works well with shallow Decision Trees (*stumps*).

#### AdaBoost Algorithm Highlights:
\\[
F(x) = \\sum_{m=1}^{M} \\alpha_m h_m(x)
\\]
Where:
- \\( h_m(x) \\): weak learner  
- \\( \\alpha_m \\): weight proportional to accuracy  

---

### 6. Gradient Boosting
Uses **gradient descent optimization** to minimize the loss function by adding new trees that correct residual errors.

#### Process:
1. Start with an initial prediction (often the mean of targets).  
2. Compute residuals (errors).  
3. Fit a new weak learner to these residuals.  
4. Update predictions iteratively.

#### Advantages:
- High accuracy for structured data.
- Can handle various loss functions.
- Forms the basis for advanced algorithms like XGBoost, LightGBM, and CatBoost.

---

### 7. Stacking (Stacked Generalization)
- Combines multiple base models using a **meta-learner** that learns how to best combine their predictions.
- Example: Use Logistic Regression as a meta-learner over Random Forest, SVM, and Neural Network predictions.

---

### 8. Bias–Variance Tradeoff in Ensembles
- **Bagging:** Reduces variance (averaging effect).
- **Boosting:** Reduces bias (sequential improvement).
- **Random Forests:** Balance both variance and interpretability.

---

### 9. Applications
- Credit scoring and fraud detection  
- Medical diagnostics  
- Text classification  
- Stock price prediction  
- Image recognition  
- Recommendation systems

---

### Summary
Ensemble methods improve robustness and accuracy by leveraging the collective wisdom of multiple models.  
They form the backbone of **state-of-the-art ML systems**, particularly for structured/tabular data.`,
  keyTakeaways: [
    "Ensemble Learning combines multiple models to enhance performance and reduce overfitting.",
    "Bagging builds multiple models on bootstrapped datasets and aggregates their predictions.",
    "Random Forests add feature randomness to bagging for better generalization.",
    "Boosting trains models sequentially, focusing on correcting previous errors.",
    "AdaBoost assigns weights to weak learners based on their accuracy.",
    "Gradient Boosting uses residual errors and gradient optimization.",
    "Stacking blends multiple models using a meta-learner.",
    "Bagging reduces variance, while boosting reduces bias.",
    "Ensemble methods power many modern ML algorithms like XGBoost and CatBoost.",
    "Although powerful, ensembles can be computationally intensive and less interpretable."
  ],
  quiz: [
    {
      question: "### What is the main goal of Ensemble Learning?",
      options: [
        "To train a single deep model",
        "To combine multiple models for better accuracy and stability",
        "To remove feature correlations",
        "To reduce the dataset size"
      ],
      correct: 1
    },
    {
      question: "### What does Bagging stand for?",
      options: [
        "Batch Aggregation",
        "Bootstrap Aggregating",
        "Binary Aggregation",
        "Boosted Aggregation"
      ],
      correct: 1
    },
    {
      question: "### What problem does Bagging primarily solve?",
      options: [
        "High bias",
        "High variance",
        "Low dimensionality",
        "Feature correlation"
      ],
      correct: 1
    },
    {
      question: "### How does Random Forest improve upon standard Bagging?",
      options: [
        "By using deeper trees",
        "By introducing random feature selection at splits",
        "By removing bootstrapping",
        "By increasing bias"
      ],
      correct: 1
    },
    {
      question: "### What does Boosting focus on reducing?",
      options: [
        "Bias",
        "Variance",
        "Overfitting",
        "Model depth"
      ],
      correct: 0
    },
    {
      question: "### Which ensemble method trains models sequentially?",
      options: [
        "Bagging",
        "Boosting",
        "Random Forest",
        "Stacking"
      ],
      correct: 1
    },
    {
      question: "### In AdaBoost, which samples get more weight?",
      options: [
        "Easily classified samples",
        "Misclassified samples from previous rounds",
        "Random samples",
        "Samples with missing values"
      ],
      correct: 1
    },
    {
      question: "### What is the purpose of Gradient Boosting?",
      options: [
        "To fit residual errors from previous models using gradient descent",
        "To train independent models in parallel",
        "To reduce dataset size",
        "To select best features only"
      ],
      correct: 0
    },
    {
      question: "### What is a meta-learner in Stacking?",
      options: [
        "A model that predicts hyperparameters",
        "A model that combines predictions from base learners",
        "A neural network used for deep stacking",
        "A feature engineering tool"
      ],
      correct: 1
    },
    {
      question: "### Which of the following reduces model variance?",
      options: [
        "Boosting",
        "Bagging",
        "Regularization",
        "Gradient Clipping"
      ],
      correct: 1
    },
    {
      question: "### Which algorithm forms the basis of Random Forest?",
      options: [
        "Decision Trees",
        "KNN",
        "SVM",
        "Naive Bayes"
      ],
      correct: 0
    },
    {
      question: "### Which ensemble technique focuses on model diversity and weighted voting?",
      options: [
        "Boosting",
        "Bagging",
        "Stacking",
        "Dropout"
      ],
      correct: 2
    },
    {
      question: "### What is one limitation of ensemble models?",
      options: [
        "They cannot be used for classification tasks",
        "They tend to be computationally intensive and less interpretable",
        "They reduce both bias and variance simultaneously always",
        "They cannot handle continuous data"
      ],
      correct: 1
    }
  ]
},
{
  id: 12,
  title: "Deep Learning",
  pages: "263–end",
  pdfFile: "lecture_notes.pdf",
  summary: `### Overview
**Deep Learning (DL)** is a subfield of Machine Learning that uses **multi-layered artificial neural networks (ANNs)** to learn complex hierarchical representations from data.  
It has powered breakthroughs in **computer vision, natural language processing, speech recognition, and generative modeling.**

Unlike traditional ML algorithms, which rely heavily on manual feature engineering, deep learning models **automatically learn features** directly from raw data.

---

### 1. Artificial Neural Networks (ANNs)
#### Structure:
An ANN consists of:
- **Input Layer:** Receives data.
- **Hidden Layers:** Perform transformations and feature extraction.
- **Output Layer:** Produces predictions (classification or regression).

Each connection between neurons has a **weight (w)** that determines the influence of one neuron on another.

#### Neuron Operation:
\\[
z = w^T x + b
\\]
\\[
a = f(z)
\\]
Where:
- \\( f \\) is an activation function  
- \\( a \\) is the neuron output  

---

### 2. Activation Functions
Activation functions introduce **nonlinearity**, allowing networks to model complex relationships.

Common examples:
- **Sigmoid:** \\( f(x) = \\frac{1}{1 + e^{-x}} \\)
- **Tanh:** \\( f(x) = \\tanh(x) \\)
- **ReLU (Rectified Linear Unit):** \\( f(x) = \\max(0, x) \\)
- **Leaky ReLU:** Variant of ReLU preventing dead neurons.
- **Softmax:** Converts output scores into probabilities (used in classification).

---

### 3. Training Neural Networks
#### Forward Propagation:
- Inputs flow through the network to generate predictions.

#### Loss Function:
- Measures how far predictions are from true labels.
- Common examples:
  - **Cross-Entropy Loss** (classification)
  - **Mean Squared Error (MSE)** (regression)

#### Backpropagation:
- Computes gradients of loss with respect to weights using the **chain rule**.
- Weights are updated using an optimizer (e.g., **SGD**, **Adam**).

#### Weight Update Rule:
\\[
w := w - \\alpha \\frac{\\partial J}{\\partial w}
\\]
Where \\( \\alpha \\) is the learning rate.

---

### 4. Deep Neural Networks (DNNs)
- Networks with **multiple hidden layers**.
- Capable of learning hierarchical representations — from low-level features (edges, shapes) to high-level concepts (faces, objects, meanings).
- Require large datasets and high computational power.

---

### 5. Convolutional Neural Networks (CNNs)
Used primarily for **image and spatial data**.

#### Key Components:
- **Convolution Layers:** Apply filters to detect local patterns (edges, textures).
- **Pooling Layers:** Reduce dimensionality and computation.
- **Fully Connected Layers:** Combine features for final classification.

#### Applications:
- Image recognition
- Object detection
- Medical imaging
- Autonomous driving

---

### 6. Recurrent Neural Networks (RNNs)
Used for **sequential data** (time series, text, audio).

#### Characteristics:
- Maintain **internal memory (state)** to process variable-length sequences.
- Suffer from **vanishing/exploding gradients**, solved by:
  - **LSTM (Long Short-Term Memory)**
  - **GRU (Gated Recurrent Unit)**

#### Applications:
- Speech recognition
- Text generation
- Machine translation

---

### 7. Autoencoders
- Neural networks designed to **compress and reconstruct** input data.
- Used for **dimensionality reduction**, **denoising**, and **anomaly detection**.
- Structure: Encoder → Bottleneck → Decoder.

---

### 8. Deep Belief Networks (DBNs)
- Composed of multiple **Restricted Boltzmann Machines (RBMs)** stacked together.
- Learn probabilistic representations of input data.
- Predecessor to modern deep neural networks.

---

### 9. Generative Models
Deep learning can also generate new data samples resembling training data.

#### Types:
- **Variational Autoencoders (VAEs):** Learn latent variable distributions.
- **Generative Adversarial Networks (GANs):** Two networks (generator + discriminator) compete to generate realistic data.
- **Diffusion Models:** Generate images via noise removal (used in tools like DALL·E, Midjourney).

---

### 10. Transfer Learning
- Reusing knowledge learned from one task (pretrained model) for another related task.
- Example: Fine-tuning ImageNet-trained CNNs for medical images.
- Saves computation and improves accuracy on limited data.

---

### 11. Regularization in Deep Learning
To prevent overfitting:
- **Dropout:** Randomly disables neurons during training.
- **Batch Normalization:** Normalizes layer inputs for stable learning.
- **Weight Decay:** Adds L2 regularization to loss function.
- **Early Stopping:** Stops training when validation performance plateaus.

---

### 12. Frameworks and Libraries
Common frameworks for building deep learning models:
- **TensorFlow / Keras** — High-level, scalable, and production-ready.
- **PyTorch** — Dynamic computation graphs and flexibility for research.
- **JAX** — Optimized for high-performance numerical computing.

---

### 13. Applications of Deep Learning
- Computer vision (image classification, object detection)
- Natural language processing (chatbots, translation)
- Speech recognition
- Healthcare (disease detection)
- Autonomous vehicles
- Generative art and AI creativity

---

### 14. Challenges in Deep Learning
- Requires large labeled datasets and powerful hardware.
- Difficult to interpret (black-box models).
- Prone to overfitting on small data.
- High energy consumption during training.

---

### Summary
Deep Learning models mimic how the brain processes information through layers of abstraction.  
They power most of today’s AI systems — from recommendation engines to generative AI — and continue to evolve through more efficient architectures and training techniques.`,
  keyTakeaways: [
    "Deep Learning uses multi-layer neural networks to learn hierarchical data representations.",
    "Neurons compute weighted sums and apply nonlinear activation functions.",
    "Backpropagation and gradient descent drive weight updates during training.",
    "CNNs specialize in spatial data like images using filters and pooling layers.",
    "RNNs handle sequential data through feedback connections (LSTM, GRU).",
    "Autoencoders compress and reconstruct data for unsupervised learning.",
    "GANs and VAEs are powerful generative models.",
    "Transfer learning reuses pretrained networks for new tasks.",
    "Dropout, batch normalization, and weight decay prevent overfitting.",
    "Deep learning powers modern AI applications across industries."
  ],
  quiz: [
    {
      question: "### What defines Deep Learning compared to traditional ML?",
      options: [
        "It uses linear regression only",
        "It employs multi-layered neural networks that learn features automatically",
        "It eliminates the need for data",
        "It replaces supervised learning entirely"
      ],
      correct: 1
    },
    {
      question: "### What is the purpose of an activation function?",
      options: [
        "To add nonlinearity and allow modeling of complex relationships",
        "To compute gradients faster",
        "To normalize feature scales",
        "To store neuron weights"
      ],
      correct: 0
    },
    {
      question: "### Which activation function is most commonly used in deep networks?",
      options: [
        "Sigmoid",
        "ReLU",
        "Tanh",
        "Softmax"
      ],
      correct: 1
    },
    {
      question: "### What algorithm adjusts neural network weights during training?",
      options: [
        "Forward Propagation",
        "Backpropagation",
        "Pooling",
        "Convolution"
      ],
      correct: 1
    },
    {
      question: "### What problem does dropout address?",
      options: [
        "Vanishing gradients",
        "Overfitting",
        "Data imbalance",
        "Feature scaling"
      ],
      correct: 1
    },
    {
      question: "### Which network type is used for image recognition?",
      options: [
        "RNN",
        "CNN",
        "Autoencoder",
        "GAN"
      ],
      correct: 1
    },
    {
      question: "### What is the main challenge with RNNs that LSTM solves?",
      options: [
        "Overfitting",
        "Vanishing and exploding gradients",
        "Low computational efficiency",
        "Data normalization"
      ],
      correct: 1
    },
    {
      question: "### What is the key difference between CNNs and RNNs?",
      options: [
        "CNNs process spatial data; RNNs process sequential data",
        "RNNs require no training",
        "CNNs are used for audio data only",
        "They are identical architectures"
      ],
      correct: 0
    },
    {
      question: "### What does a generator do in a GAN?",
      options: [
        "Classifies input data",
        "Creates synthetic data resembling the training set",
        "Measures gradient updates",
        "Encodes text features"
      ],
      correct: 1
    },
    {
      question: "### What is the function of batch normalization?",
      options: [
        "To normalize layer inputs and stabilize training",
        "To increase network depth",
        "To reduce dataset size",
        "To handle missing data"
      ],
      correct: 0
    },
    {
      question: "### Which optimizer is most commonly used for training deep networks?",
      options: [
        "Adam",
        "RMSProp",
        "SGD",
        "Adagrad"
      ],
      correct: 0
    },
    {
      question: "### What is transfer learning used for?",
      options: [
        "Reusing pretrained models for related tasks",
        "Training models without any data",
        "Replacing gradient descent",
        "Combining decision trees"
      ],
      correct: 0
    },
    {
      question: "### Which of the following is a generative deep learning model?",
      options: [
        "SVM",
        "GAN",
        "Random Forest",
        "Naive Bayes"
      ],
      correct: 1
    },
    {
      question: "### What is one major drawback of deep learning?",
      options: [
        "It cannot handle images or text",
        "It requires large datasets and is computationally expensive",
        "It cannot model nonlinearity",
        "It eliminates the need for optimization"
      ],
      correct: 1
    }
  ]
},
  {
    id: 21,
    title: "Python Programming Fundamentals",
    pages: "All",
    pdfFile: "beginners_python_cheat_sheet_pcc_all.pdf",
    summary:
      "This unit covers essential Python programming concepts including variables, data types, lists, dictionaries, control flow, functions, classes, file handling, and exception handling. It provides a comprehensive foundation for Python programming with practical examples and best practices.",
    keyTakeaways: [
      "Python provides versatile data structures: lists, dictionaries, tuples for different use cases",
      "Python uses dynamic typing with variables storing values of any type",
      "Object-oriented programming with classes enables code reusability and inheritance",

      "Lists, dictionaries, and tuples are fundamental data structures",
      "Multiple libraries extend Python's capabilities: Pygame for games, matplotlib/Pygal for visualization, Django for web development",

      "Functions and classes enable code reusability and organization",
      "Exception handling ensures robust error management in programs",
      "Testing with unittest and proper exception handling ensure robust, maintainable code",
    ],
    quiz: [
      {
        question: "How do you create a string in Python?",
        options: [
          "Using square brackets []",
          "Using curly braces {}",
          "Using single or double quotes",
          "Using parentheses ()",
        ],
        correct: 2,
      },
      {
        question: "What does the append() method do to a list?",
        options: [
          "Removes last item",
          "Adds item to the end",
          "Sorts the list",
          "Clears the list",
        ],
        correct: 1,
      },
      {
        question: "How do you access the last item in a list?",
        options: ["list[0]", "list[-1]", "list[end]", "list.last()"],
        correct: 1,
      },
      {
        question: "What is the correct syntax for a dictionary?",
        options: [
          "['key': 'value']",
          "('key': 'value')",
          "{'key': 'value'}",
          "<'key': 'value'>",
        ],
        correct: 2,
      },
      {
        question: "What keyword is used to define a function?",
        options: ["function", "def", "func", "define"],
        correct: 1,
      },
      {
        question: "How do you check if a value is in a list?",
        options: [
          "value inside list",
          "value in list",
          "list.contains(value)",
          "list.has(value)",
        ],
        correct: 1,
      },
      {
        question: "What does the range() function return by default?",
        options: ["A list", "A tuple", "An iterable sequence", "A dictionary"],
        correct: 2,
      },
      {
        question: "Which block handles errors in Python?",
        options: ["catch", "except", "error", "handle"],
        correct: 1,
      },
      {
        question: "What does '__init__' do in a class?",
        options: [
          "Deletes the object",
          "Initializes object attributes",
          "Ends the program",
          "Creates a copy",
        ],
        correct: 1,
      },
      {
        question: "How do you open a file for reading in Python?",
        options: [
          "open(file, 'r')",
          "read(file)",
          "file.open('r')",
          "open('r', file)",
        ],
        correct: 0,
      },
      // Variables and Strings
      {
        question: "What is string concatenation in Python?",
        options: [
          "Splitting strings",
          "Combining strings using +",
          "Converting strings to integers",
          "Removing spaces",
        ],
        correct: 1,
      },
      {
        question: "How do you create a multi-line string?",
        options: [
          "Using \\n",
          "Using triple quotes '''",
          "Using + operator",
          "Using semicolons",
        ],
        correct: 1,
      },

      // Lists
      {
        question: "What does list slicing [1:4] return?",
        options: [
          "Items at index 1, 2, 3",
          "Items at index 1, 2, 3, 4",
          "Items at index 0, 1, 2, 3",
          "Only item at index 1",
        ],
        correct: 0,
      },
      {
        question: "How do you create an empty list?",
        options: [
          "list = empty",
          "list = []",
          "list = new List()",
          "list = nil",
        ],
        correct: 1,
      },
      {
        question: "What does the insert() method do?",
        options: [
          "Adds item at specific position",
          "Removes an item",
          "Sorts the list",
          "Reverses the list",
        ],
        correct: 0,
      },
      {
        question: "What is a list comprehension?",
        options: [
          "A way to document lists",
          "Concise way to create lists",
          "Method to print lists",
          "Way to delete lists",
        ],
        correct: 1,
      },
      {
        question: "How do you copy a list properly?",
        options: [
          "new_list = old_list",
          "new_list = old_list[:]",
          "new_list = old_list.copy",
          "new_list = copy(old_list)",
        ],
        correct: 1,
      },

      // Tuples
      {
        question: "What is the main difference between lists and tuples?",
        options: [
          "Tuples are faster",
          "Tuples are immutable",
          "Lists are immutable",
          "No difference",
        ],
        correct: 1,
      },
      {
        question: "How do you define a tuple?",
        options: [
          "Using square brackets",
          "Using curly braces",
          "Using parentheses",
          "Using angle brackets",
        ],
        correct: 2,
      },

      // Dictionaries
      {
        question: "How do you access a dictionary value?",
        options: ["dict.value", "dict['key']", "dict->key", "dict(key)"],
        correct: 1,
      },
      {
        question: "What method returns all keys in a dictionary?",
        options: [
          "dict.keys()",
          "dict.getKeys()",
          "dict.allKeys()",
          "dict.keyList()",
        ],
        correct: 0,
      },
      {
        question: "How do you add a new key-value pair to a dictionary?",
        options: [
          "dict.add(key, value)",
          "dict[key] = value",
          "dict.insert(key, value)",
          "dict.push(key, value)",
        ],
        correct: 1,
      },
      {
        question: "What does dict.items() return?",
        options: [
          "Only keys",
          "Only values",
          "Key-value pairs",
          "Dictionary size",
        ],
        correct: 2,
      },

      // If Statements
      {
        question: "What is the correct syntax for an if statement?",
        options: ["if x == 5:", "if (x == 5)", "if x = 5:", "if x equals 5:"],
        correct: 0,
      },
      {
        question: "What operator checks if two values are NOT equal?",
        options: ["<>", "=/=", "!=", "NOT"],
        correct: 2,
      },
      {
        question: "What does the 'and' operator do?",
        options: [
          "Adds numbers",
          "Returns True if all conditions are True",
          "Combines strings",
          "Creates lists",
        ],
        correct: 1,
      },

      // While Loops
      {
        question: "What does a while loop do?",
        options: [
          "Runs once",
          "Repeats while condition is True",
          "Never runs",
          "Runs exactly 10 times",
        ],
        correct: 1,
      },
      {
        question: "What keyword exits a loop immediately?",
        options: ["exit", "stop", "break", "end"],
        correct: 2,
      },
      {
        question: "What does 'continue' do in a loop?",
        options: [
          "Exits the loop",
          "Skips to next iteration",
          "Pauses the loop",
          "Restarts the loop",
        ],
        correct: 1,
      },

      // User Input
      {
        question: "How do you get user input in Python?",
        options: ["get()", "input()", "read()", "scan()"],
        correct: 1,
      },
      {
        question: "What type does input() return?",
        options: ["Integer", "String", "Float", "Boolean"],
        correct: 1,
      },
      {
        question: "How do you convert input to an integer?",
        options: [
          "integer(input())",
          "int(input())",
          "toInt(input())",
          "input.int()",
        ],
        correct: 1,
      },

      // Functions
      {
        question: "What keyword defines a function?",
        options: ["function", "def", "func", "define"],
        correct: 1,
      },
      {
        question: "What does 'return' do in a function?",
        options: [
          "Exits program",
          "Sends back a value",
          "Prints output",
          "Deletes function",
        ],
        correct: 1,
      },
      {
        question: "What are default parameter values?",
        options: [
          "Required values",
          "Values used if no argument provided",
          "Maximum values",
          "Error values",
        ],
        correct: 1,
      },
      {
        question: "What is *args used for?",
        options: [
          "Fixed arguments",
          "Arbitrary number of arguments",
          "Keyword arguments",
          "No arguments",
        ],
        correct: 1,
      },
      {
        question: "What is **kwargs used for?",
        options: [
          "Fixed keywords",
          "Arbitrary keyword arguments",
          "No keywords",
          "Single keyword",
        ],
        correct: 1,
      },

      // Classes
      {
        question: "What keyword defines a class?",
        options: ["class", "object", "define", "struct"],
        correct: 0,
      },
      {
        question: "What is 'self' in a class method?",
        options: [
          "Class name",
          "Instance reference",
          "Parent class",
          "Module name",
        ],
        correct: 1,
      },
      {
        question: "What is inheritance?",
        options: [
          "Deleting classes",
          "Child class inherits from parent",
          "Creating objects",
          "Calling functions",
        ],
        correct: 1,
      },
      {
        question: "What does super() do?",
        options: [
          "Creates superuser",
          "Accesses parent class",
          "Makes class powerful",
          "Deletes class",
        ],
        correct: 1,
      },
      {
        question: "What is an instance?",
        options: [
          "Class definition",
          "Object created from class",
          "Method name",
          "Variable type",
        ],
        correct: 1,
      },

      // Files and Exceptions
      {
        question: "What mode opens a file for writing?",
        options: ["'r'", "'w'", "'x'", "'p'"],
        correct: 1,
      },
      {
        question: "What is the 'with' statement used for?",
        options: [
          "Conditionals",
          "Loops",
          "Safe file handling",
          "Function definition",
        ],
        correct: 2,
      },
      {
        question: "What does 'a' mode do?",
        options: [
          "Reads file",
          "Appends to file",
          "Deletes file",
          "Creates file",
        ],
        correct: 1,
      },
      {
        question: "What block catches exceptions?",
        options: ["catch", "except", "error", "handle"],
        correct: 1,
      },
      {
        question: "Where does code that might cause error go?",
        options: ["except block", "try block", "finally block", "else block"],
        correct: 1,
      },
      {
        question: "What is ZeroDivisionError?",
        options: [
          "File error",
          "Division by zero error",
          "Syntax error",
          "Memory error",
        ],
        correct: 1,
      },

      // Testing
      {
        question: "What module provides testing tools?",
        options: ["test", "unittest", "pytest", "testlib"],
        correct: 1,
      },
      {
        question: "What is a test case?",
        options: [
          "Single test",
          "Collection of unit tests",
          "Error message",
          "Function name",
        ],
        correct: 1,
      },
      {
        question: "What does assertEqual() do?",
        options: [
          "Assigns values",
          "Verifies two values are equal",
          "Adds numbers",
          "Creates variables",
        ],
        correct: 1,
      },
      {
        question: "What is setUp() method used for?",
        options: [
          "Ending tests",
          "Running before each test",
          "Deleting tests",
          "Counting tests",
        ],
        correct: 1,
      },

      // Pygame
      {
        question: "What does pygame.init() do?",
        options: [
          "Ends game",
          "Initializes Pygame modules",
          "Creates window",
          "Loads images",
        ],
        correct: 1,
      },
      {
        question: "What is a rect object?",
        options: [
          "Image file",
          "Rectangular area for positioning",
          "Sound effect",
          "Color value",
        ],
        correct: 1,
      },
      {
        question: "What does pg.display.flip() do?",
        options: [
          "Rotates screen",
          "Updates the display",
          "Closes window",
          "Saves game",
        ],
        correct: 1,
      },
      {
        question: "What event type is a key press?",
        options: ["KEYDOWN", "KEYPRESS", "KEY", "PRESS"],
        correct: 0,
      },
      {
        question: "How do you detect collisions in Pygame?",
        options: ["collision()", "spritecollide()", "hit()", "overlap()"],
        correct: 1,
      },

      // Matplotlib
      {
        question: "What function creates a scatter plot?",
        options: ["plot()", "scatter()", "graph()", "point()"],
        correct: 1,
      },
      {
        question: "What does plt.show() do?",
        options: [
          "Hides plot",
          "Displays the plot",
          "Saves plot",
          "Deletes plot",
        ],
        correct: 1,
      },
      {
        question: "How do you add a title to a plot?",
        options: ["plt.name()", "plt.title()", "plt.header()", "plt.label()"],
        correct: 1,
      },
      {
        question: "What is a colormap?",
        options: [
          "Image filter",
          "Color variation scheme",
          "Plot type",
          "File format",
        ],
        correct: 1,
      },

      // Pygal
      {
        question: "What does chart.render_to_file() do?",
        options: [
          "Deletes chart",
          "Saves chart to file",
          "Prints chart",
          "Edits chart",
        ],
        correct: 1,
      },
      {
        question: "What is Pygal primarily used for?",
        options: [
          "Gaming",
          "Data visualization",
          "Web scraping",
          "File handling",
        ],
        correct: 1,
      },
      {
        question: "What does chart.add() do?",
        options: [
          "Adds data series",
          "Deletes data",
          "Creates chart",
          "Closes chart",
        ],
        correct: 0,
      },

      // Django
      {
        question: "What command creates a new Django project?",
        options: [
          "create project",
          "django-admin.py startproject",
          "new project",
          "make project",
        ],
        correct: 1,
      },
      {
        question: "What does python manage.py migrate do?",
        options: [
          "Deletes database",
          "Updates database structure",
          "Backs up data",
          "Creates project",
        ],
        correct: 1,
      },
      {
        question: "What is a Django model?",
        options: [
          "Template file",
          "Database structure definition",
          "URL pattern",
          "View function",
        ],
        correct: 1,
      },
      {
        question: "What does a view function do?",
        options: [
          "Displays database",
          "Processes requests and returns responses",
          "Styles pages",
          "Creates URLs",
        ],
        correct: 1,
      },
      {
        question: "What is a template in Django?",
        options: [
          "Python file",
          "HTML structure for pages",
          "Database table",
          "CSS file",
        ],
        correct: 1,
      },
      {
        question: "What does {% extends %} do?",
        options: [
          "Adds data",
          "Inherits from parent template",
          "Creates loop",
          "Defines variable",
        ],
        correct: 1,
      },
      {
        question: "What is the purpose of urls.py?",
        options: [
          "Stores data",
          "Maps URLs to views",
          "Defines models",
          "Handles errors",
        ],
        correct: 1,
      },
      {
        question: "What decorator restricts access to logged-in users?",
        options: ["@login", "@login_required", "@authenticated", "@secure"],
        correct: 1,
      },
    ],
  },
];
