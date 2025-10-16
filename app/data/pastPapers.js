export const pastPapers = [
  {
    id: 1,
    year: 2023,
    date: "Tuesday 28 November, 2023",
    sections: [
      {
        id: "section-a",
        name: "Section A",
        mandatory: true,
        introText: "NOTE: Your answers should NOT be brief. Give detailed essay style reasoned answers with structured paragraphs and headings.",
        questions: [
          {
            id: "q1",
            questionNumber: "1",
            question: `Product recommendations have become an integral part of modern consumer experiences. In an era where options abound, navigating through the multitude of products available can be overwhelming. Hence, the significance of tailored suggestions cannot be overstated. From personalized algorithms on e-commerce platforms to word-of-mouth recommendations, consumers rely on these pointers to make informed choices.

These recommendations are often driven by a blend of data analytics, user preferences, and behavioral patterns. Algorithms crunch vast amounts of data, examining purchase history, browsing habits, and demographic information to generate suggestions. The aim is to anticipate and fulfill consumer needs, presenting them with options that align with their tastes and requirements.

Moreover, the influence of peer recommendations remains potent. Word-of-mouth, whether through social media, reviews, or direct interactions, holds sway over consumer decisions. The human touch in these recommendations adds a layer of trust and relatability, often guiding individuals towards products they might not have considered otherwise.

Ultimately, product recommendations serve not only as a convenience but as a means to streamline choices in an increasingly saturated market. By offering tailored suggestions, they facilitate decision-making, saving time and effort while enhancing the likelihood of a satisfying purchase experience. In this digital age, where information overflow can be daunting, these recommendations act as guiding beacons, aiding consumers in navigating the seas of available products.`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q1a",
                questionNumber: "a",
                question:
                  "Identify the type of machine learning algorithms you would use for task of Product recommendations. Explain your reasoning at length.",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1b",
                questionNumber: "b",
                question:
                  "Identify, explain and Justify the features you would extract from the Product recommendations to train the machine learning model.",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1c",
                questionNumber: "c",
                question:
                  "Discuss how you would evaluate the performance of the machine learning model during the training phase.",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1d",
                questionNumber: "d",
                question:
                  "Identify and evaluate ethical considerations you would need to address when using a machine learning model to recommend products.",
                sampleAnswer: ``,
                marks: 10,
              },
            ],
          },
        ],
      },
      {
        id: "section-b-2023",
        name: "Section B",
        mandatory: false,
        introText: "Attempt any THREE questions in this section",
        questions: [
          {
            id: "q2-2023",
            questionNumber: "2",
            question: `Machine learning's efficacy intertwines with robust algorithms, diverse data, and intricate architectures. Algorithms span supervised and unsupervised learning, harnessing techniques from regression to reinforcement learning. Quality data, pivotal for model efficacy, demands cleanliness and relevance, amplified by quantity yet not solely reliant on volume. Architectures, notably neural networks, boast specialized structures like CNNs and RNNs tailored for specific data types. Performance hinges on metrics like accuracy, precision, and recall, while continuous enhancements via regularization, hyperparameter tuning, and ethical considerations propel the field. Deployment efficiency and scalability culminate the process, ensuring models endure real-world demands while upholding ethical standards.`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q2a-2023",
                questionNumber: "a",
                question:
                  "Explain the concepts of overfitting and underfitting in machine learning. Provide examples of scenarios where each might occur.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q2b-2023",
                questionNumber: "b",
                question:
                  "Compare and contrast feedforward neural networks with recurrent neural networks, highlighting their respective architectures and typical applications.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q2c-2023",
                questionNumber: "c",
                question: `Write short notes with examples on the following aspects of machine learning:
i) Hyper-parameter Tuning
ii) Feature Engineering
iii) Accuracy
iv) Precision`,
                sampleAnswer: ``,
                marks: 8,
              },
            ],
          },
          {
            id: "q3-2023",
            questionNumber: "3",
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q3a-2023",
                questionNumber: "a",
                question:
                  "The K-Nearest Neighbors (KNN) algorithm is a simple, instance-based learning method used for both classification and regression tasks in machine learning. Explain how the KNN algorithm makes predictions for classification and regression tasks.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q3b-2023",
                questionNumber: "b",
                question:
                  "The Expectation-Maximization (EM) algorithm is an iterative method used in unsupervised machine learning to estimate parameters in probabilistic models with latent or unobserved variables. Describe how the EM algorithm iteratively maximizes the likelihood function in the context of unsupervised learning and latent variable models.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q3c-2023",
                questionNumber: "c",
                question:
                  "Hidden Markov Models (HMMs) are statistical models used to model sequential data, particularly when dealing with temporal or sequential patterns. Describe four areas where HMMs can be used.",
                sampleAnswer: ``,
                marks: 8,
              },
            ],
          },
          {
            id: "q4-2023",
            questionNumber: "4",
            question: `Machine learning algorithms epitomize versatility and precision, molding the landscape of AI applications. Supervised algorithms, from linear regression to neural networks, masterfully predict outcomes and classify data. Unsupervised counterparts, like clustering and dimensionality reduction, uncover hidden patterns within vast datasets. Reinforcement learning algorithms navigate environments, learning through trial and error to optimize decision-making. Each algorithm thrives on distinct data nuances, demanding clean, diverse datasets for optimal performance. The evolution of these algorithms is relentless, perpetually refining through ensemble methods, hyperparameter tuning, and ethical considerations to yield more accurate, fair, and impactful outcomes across industries and domains.`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q4a-2023",
                questionNumber: "a",
                question:
                  "Describe the primary objective of Gradient Descent and how it operates to minimize the cost or loss function in model training.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q4b-2023",
                questionNumber: "b",
                question:
                  "Decision trees are predictive models used in machine learning for both classification and regression tasks. Explain how decision trees partition the feature space and make predictions or classifications.",
                sampleAnswer: ``,
                marks: 8,
              },
              {
                id: "q4c-2023",
                questionNumber: "c",
                question:
                  "Clustering or cluster analysis is a machine learning technique, which groups the unlabelled dataset. Describe how clustering is used to partition images into distinct regions or objects based on pixel attributes.",
                sampleAnswer: ``,
                marks: 6,
              },
            ],
          },
          {
            id: "q5-2023",
            questionNumber: "5",
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q5a-2023",
                questionNumber: "a",
                question:
                  "Explain the role of Matplotlib python library in visualizing data for machine learning tasks. Provide examples of how this library can be used to create various types of plots for data exploration and model evaluation.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q5b-2023",
                questionNumber: "b",
                question:
                  "Describe Reinforcement Learning (RL) and outline its key components, including agents, environments, actions, rewards, and the exploration-exploitation tradeoff.",
                sampleAnswer: ``,
                marks: 8,
              },
              {
                id: "q5c-2023",
                questionNumber: "c",
                question:
                  "Consider a retail company aiming to forecast sales based on historical data. Describe how linear regression can be applied to predict future sales trends, including the key features (variables) that might influence the sales prediction.",
                sampleAnswer: ``,
                marks: 6,
              },
            ],
          },
        ],
      },
    ],
  },
  {
    id: 2,
    year: 2024,
    date: "Tuesday, 26 November 2024",
    sections: [
      {
        id: "section-a-2024",
        name: "Section A",
        mandatory: true,
        introText: "NOTE: Your answers should NOT be brief. Give detailed essay style reasoned answers with structured paragraphs and headings.",
        questions: [
          {
            id: "q1-2024",
            questionNumber: "1",
            question: `Using the model code and its corresponding output given below, critically study both and answer the following questions:

**Python Model Code:**

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Simulating data for retail customer behavior
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(18, 65, n_samples),
    'annual_income': np.round(np.random.uniform(20000, 120000, n_samples), 2),
    'purchase_frequency': np.random.randint(1, 12, n_samples),
    'membership_type': np.random.choice(['Basic', 'Premium', 'VIP'], n_samples),
    'online_shopping': np.random.choice(['Yes', 'No'], n_samples),
    'loyalty_program': np.random.choice(['Yes', 'No'], n_samples),
    'large_purchase': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Encoding categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Splitting data into features (X) and target (y)
X = df_encoded.drop('large_purchase', axis=1)
y = df_encoded['large_purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Making predictions
y_pred = svm_model.predict(X_test)

# Evaluating the model
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='large_purchase')
plt.title('Large Purchase Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='large_purchase', y='annual_income')
plt.title('Annual Income vs Large Purchase')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='large_purchase', y='purchase_frequency')
plt.title('Purchase Frequency vs Large Purchase')
plt.show()
\`\`\`

**Model Output:**

Confusion Matrix:
[[149   0]
 [ 51   0]]

Classification Report:
              precision    recall  f1-score   support
           0       0.74      1.00      0.85       149
           1       0.00      0.00      0.00        51
    accuracy                           0.74       200
   macro avg       0.37      0.50      0.43       200
weighted avg       0.55      0.74      0.64       200

Accuracy Score: 0.745`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q1a-2024",
                questionNumber: "a",
                question:
                  "Compare the performance metrics (confusion matrix, accuracy score, and classification report) generated by the SVM model. Which metric is the most informative for assessing the model's effectiveness, and why?",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1b-2024",
                questionNumber: "b",
                question:
                  "Critically study the results produced by the box plot graphs and interpret the results in the context of the given model.",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1c-2024",
                questionNumber: "c",
                question:
                  "Explain how the encoding of categorical variables impact the SVM model's ability to classify large_purchase accurately and identify any potential limitations or biases introduced during this step.",
                sampleAnswer: ``,
                marks: 10,
              },
              {
                id: "q1d-2024",
                questionNumber: "d",
                question:
                  "Based on the analysis of the prediction results, propose a strategy to improve the model's performance and suggest additional features or preprocessing steps that could enhance its predictive accuracy.",
                sampleAnswer: ``,
                marks: 10,
              },
            ],
          },
        ],
      },
      {
        id: "section-b-2024",
        name: "Section B",
        mandatory: false,
        introText: "Attempt any THREE questions in this section",
        questions: [
          {
            id: "q2-2024",
            questionNumber: "2",
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q2a-2024",
                questionNumber: "a",
                question:
                  "Convolutional Neural Networks (CNNs) are a type of artificial neural network specifically designed for processing and analyzing grid-like data. Describe any three applications of CNNs.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q2b-2024",
                questionNumber: "b",
                question:
                  "Evaluation metrics in machine learning are measures used to assess the performance of models. Compare and contrast accuracy and precision metrics.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q2c-2024",
                questionNumber: "c",
                question:
                  "With the aid of an example, evaluate how increasing the size of the training dataset impacts a machine learning model that is overfitting.",
                sampleAnswer: ``,
                marks: 8,
              },
            ],
          },
          {
            id: "q3-2024",
            questionNumber: "3",
            question: `Machine learning models are algorithms or mathematical frameworks that learn patterns and relationships from data to make predictions, classifications or decisions without explicit programming. These models process input data, adapt through training, and generalize to unseen scenarios. They are the core tools in machine learning systems for solving various tasks.`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q3a-2024",
                questionNumber: "a",
                question:
                  "Support Vector Machines (SVMs) are supervised machine learning algorithms used for classification, regression, and outlier detection. They are particularly effective for high-dimensional datasets and when the relationship between features and labels is non-linear. With the a aid of a diagram, explain how SVMs work.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q3b-2024",
                questionNumber: "b",
                question:
                  "Model selection refers to the process of choosing the most appropriate machine learning model for a given problem. Justify why model selection is important.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q3c-2024",
                questionNumber: "c",
                question: `A decision tree is a supervised machine learning algorithm used for classification and regression tasks. Write short notes on the following aspect of decision trees:
- Information Gain
- Gini Index
- Pruning
- Entropy`,
                sampleAnswer: ``,
                marks: 8,
              },
            ],
          },
          {
            id: "q4-2024",
            questionNumber: "4",
            question: `Machine learning can be categorized into three main types. Supervised learning involves training models with labeled data to predict outcomes. Unsupervised learning analyzes unlabeled data to uncover hidden patterns, such as clusters or associations. Reinforcement learning focuses on optimizing actions by learning from feedback through rewards or penalties.`,
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q4a-2024",
                questionNumber: "a",
                question:
                  "Machine learning (ML) can be broadly categorized into different types based on the learning approach and the nature of the task. Identify and describe the three main types of machine learning types.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q4b-2024",
                questionNumber: "b",
                question:
                  "K-Means is an unsupervised machine learning algorithm used for clustering, where data points are grouped into K clusters based on their similarity. List the seven steps of the K-means algorithms.",
                sampleAnswer: ``,
                marks: 8,
              },
              {
                id: "q4c-2024",
                questionNumber: "c",
                question:
                  "Naive Bayes is a supervised machine learning algorithm based on Bayes' Theorem, widely used for classification tasks. Identify and just three areas where you would use this algorithm.",
                sampleAnswer: ``,
                marks: 6,
              },
            ],
          },
          {
            id: "q5-2024",
            questionNumber: "5",
            isParentQuestion: true,
            subQuestions: [
              {
                id: "q5a-2024",
                questionNumber: "a",
                question:
                  "Exploratory Data Analysis (EDA) is a critical step in the data analysis and machine learning process. Justify why it is important to conduct EDA before Machine Learning models are trained and deployed.",
                sampleAnswer: ``,
                marks: 6,
              },
              {
                id: "q5b-2024",
                questionNumber: "b",
                question:
                  "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. Describe any four key features of RL.",
                sampleAnswer: ``,
                marks: 8,
              },
              {
                id: "q5c-2024",
                questionNumber: "c",
                question:
                  "Notebooks are interactive computational environments that combine code, text visualizations and data exploration in a single interface. Explain why it is important to use notebooks in Machine Learning Projects.",
                sampleAnswer: ``,
                marks: 6,
              },
            ],
          },
        ],
      },
    ],
  },
];