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
        introText: `Product recommendations have become an integral part of modern consumer experiences. In an era where options abound, navigating through the multitude of products available can be overwhelming. Hence, the significance of tailored suggestions cannot be overstated. From personalized algorithms on e-commerce platforms to word-of-mouth recommendations, consumers rely on these pointers to make informed choices.

These recommendations are often driven by a blend of data analytics, user preferences, and behavioral patterns. Algorithms crunch vast amounts of data, examining purchase history, browsing habits, and demographic information to generate suggestions. The aim is to anticipate and fulfill consumer needs, presenting them with options that align with their tastes and requirements.

Moreover, the influence of peer recommendations remains potent. Word-of-mouth, whether through social media, reviews, or direct interactions, holds sway over consumer decisions. The human touch in these recommendations adds a layer of trust and relatability, often guiding individuals towards products they might not have considered otherwise.

Ultimately, product recommendations serve not only as a convenience but as a means to streamline choices in an increasingly saturated market. By offering tailored suggestions, they facilitate decision-making, saving time and effort while enhancing the likelihood of a satisfying purchase experience. In this digital age, where information overflow can be daunting, these recommendations act as guiding beacons, aiding consumers in navigating the seas of available products.`,
        questions: [
          {
            id: "q1a",
            questionNumber: "1(a)",
            question:
              "For a product recommendation system, identify the type of machine learning algorithms you would use. Explain your reasoning at length.",
            sampleAnswer: `For product recommendations, I would use **Collaborative Filtering** and **Content-Based Filtering** algorithms, which fall under supervised and unsupervised learning approaches.

**Collaborative Filtering (Supervised/Unsupervised)**
This approach analyzes patterns from user behavior and preferences. User-based collaborative filtering finds similar users and recommends products they liked, while item-based filtering recommends products similar to those a user has interacted with. This works well when you have substantial user interaction data.

**Content-Based Filtering (Supervised Learning)**
This method recommends products based on product features and user preferences. If a user frequently purchases sports equipment, the system recommends similar sports-related products. This approach uses classification algorithms to match user profiles with product attributes.

**Hybrid Approach**
Modern recommendation systems combine both approaches to overcome individual limitations. Netflix and Amazon use hybrid systems that leverage both user behavior patterns and content characteristics.

**Why These Algorithms?**
- They handle the implicit feedback inherent in product recommendations
- They scale well with large product catalogs
- They can capture both user preferences and product similarities
- They improve over time as more data is collected

**Additional Techniques:**
- Matrix Factorization (SVD) for dimensionality reduction
- Deep Learning approaches (Neural Collaborative Filtering)
- Association Rule Learning (Market Basket Analysis) for "frequently bought together"`,
            marks: 10,
          },
          {
            id: "q1b",
            questionNumber: "1(b)",
            question:
              "Identify, explain and justify the features you would extract from product recommendations to train the machine learning model.",
            sampleAnswer: `**User-Based Features:**

1. **User Demographics**
   - Age, gender, location, income level
   - Justification: Different demographics show distinct purchasing patterns

2. **User Behavior Features**
   - Purchase history (products, categories, frequency)
   - Browsing patterns (time spent, pages viewed)
   - Search queries and keywords
   - Justification: Past behavior is the strongest predictor of future preferences

3. **User Engagement Metrics**
   - Click-through rates
   - Cart abandonment patterns
   - Product review ratings given
   - Justification: Indicates level of interest and satisfaction

**Product-Based Features:**

4. **Product Attributes**
   - Category, subcategory, brand
   - Price, price range
   - Product description (text features using NLP)
   - Justification: Similar products appeal to similar users

5. **Product Performance Metrics**
   - Sales volume
   - Average rating
   - Number of reviews
   - Return rate
   - Justification: Popular and well-reviewed products are safer recommendations

**Interaction Features:**

6. **User-Product Interactions**
   - Purchase indicator (binary: bought/not bought)
   - Rating given (if available)
   - Time since last interaction
   - Frequency of viewing
   - Justification: Direct feedback on user preferences

7. **Temporal Features**
   - Time of purchase
   - Seasonality (holidays, events)
   - Day of week, time of day
   - Justification: Purchase patterns vary with time

**Contextual Features:**

8. **Session Context**
   - Device type (mobile, desktop)
   - Session duration
   - Pages viewed in session
   - Justification: Context influences purchase decisions

**Feature Engineering Techniques:**
- One-hot encoding for categorical variables
- TF-IDF for product descriptions
- Embedding layers for user and item IDs
- Normalization for numerical features`,
            marks: 10,
          },
          {
            id: "q1c",
            questionNumber: "1(c)",
            question:
              "Discuss how you would evaluate the performance of the machine learning model during the training phase.",
            sampleAnswer: `**Evaluation Strategy for Recommendation Systems:**

**1. Offline Evaluation Metrics**

**Accuracy Metrics:**
- **Precision@K**: What proportion of top-K recommendations are relevant?
  - Formula: (Relevant items in top K) / K
  - Justification: Measures recommendation relevance

- **Recall@K**: What proportion of relevant items appear in top-K?
  - Formula: (Relevant items in top K) / (Total relevant items)
  - Justification: Measures coverage of user interests

- **F1-Score@K**: Harmonic mean of precision and recall
  - Justification: Balances precision and recall

**Ranking Metrics:**
- **Mean Average Precision (MAP)**: Average precision across all users
  - Justification: Considers order of recommendations

- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality
  - Justification: Higher-ranked relevant items contribute more to the score

**2. Cross-Validation Approach**

**Temporal Split:**
- Training: Historical data (e.g., 6 months)
- Validation: Next month
- Test: Following month
- Justification: Respects temporal nature of user behavior

**User-Based K-Fold:**
- Split users into K folds
- Train on K-1 folds, validate on 1 fold
- Justification: Ensures model generalizes to new users

**3. Training Monitoring**

**Loss Functions:**
- Binary Cross-Entropy for implicit feedback
- Mean Squared Error for explicit ratings
- Monitor training and validation loss curves
- Justification: Detect overfitting early

**Early Stopping:**
- Stop training when validation performance plateaus
- Justification: Prevents overfitting while maintaining performance

**4. Business Metrics**

Even during training, consider:
- **Coverage**: Percentage of products recommended
- **Diversity**: Variety in recommendations
- **Novelty**: Recommending new items vs. popular items
- Justification: These affect real-world system quality

**5. Evaluation Protocol**

**Steps:**
1. Split data (80% train, 10% validation, 10% test)
2. Train model on training set
3. Tune hyperparameters using validation set
4. Evaluate final model on test set
5. Track metrics across epochs

**Model Comparison:**
- Baseline model (e.g., popularity-based)
- Compare candidate models
- Statistical significance testing
- Justification: Ensure improvements are real, not random

The validation phase is critical to ensure the model will perform well on unseen data and provide valuable recommendations in production.`,
            marks: 10,
          },
          {
            id: "q1d",
            questionNumber: "1(d)",
            question:
              "Identify and evaluate ethical considerations when using a machine learning model to recommend products.",
            sampleAnswer: `**Major Ethical Considerations:**

**1. Bias and Fairness**

**Issue:**
Models may exhibit bias based on historical data, leading to discriminatory recommendations based on gender, age, race, or socioeconomic status.

**Example:**
- Recommending high-priced products only to certain demographics
- Gender-stereotyped product suggestions
- Geographic discrimination in product availability

**Mitigation:**
- Regular bias audits of recommendations
- Diverse training data collection
- Fairness constraints in the model
- Testing across demographic groups

**2. Privacy Concerns**

**Issue:**
Recommendation systems collect and analyze personal data, raising privacy concerns.

**Considerations:**
- What data is collected? (purchases, browsing, location)
- How is it stored and secured?
- Who has access to user data?
- Is data shared with third parties?

**Mitigation:**
- Data minimization (collect only necessary data)
- Strong encryption and security measures
- Clear privacy policies
- Comply with GDPR, CCPA regulations
- Offer opt-out options
- Anonymization and differential privacy techniques

**3. Manipulation and Exploitation**

**Issue:**
Recommendations can manipulate user behavior for profit rather than user benefit.

**Examples:**
- Recommending higher-margin products
- Creating artificial needs
- Exploiting vulnerable populations (children, gambling addiction)
- Dark patterns in recommendation displays

**Mitigation:**
- Align business goals with user interests
- Ethical guidelines for recommendation priorities
- Transparent about promotional content
- Age-appropriate recommendations
- Avoid exploiting addiction patterns

**4. Transparency and Explainability**

**Issue:**
Users often don't understand why they receive certain recommendations.

**Considerations:**
- Why was this product recommended?
- What data influenced the recommendation?
- Can users control their recommendations?

**Mitigation:**
- Provide explanations ("Based on your purchase of...")
- Allow users to provide feedback
- Give users control over recommendation factors
- Clear documentation of how the system works

**5. Filter Bubbles and Echo Chambers**

**Issue:**
Recommendations may limit user exposure to diverse products, creating narrow worldviews.

**Example:**
- Only recommending products similar to past purchases
- Reinforcing existing preferences
- Limiting discovery of new categories

**Mitigation:**
- Balance relevance with diversity
- Periodic introduction of diverse recommendations
- Serendipity features
- Allow easy exploration outside recommendations

**6. Accountability and Responsibility**

**Issue:**
Who is responsible when recommendations cause harm?

**Considerations:**
- Harmful products recommended
- Financial harm from overconsumption
- Misleading recommendations
- System errors

**Mitigation:**
- Clear accountability structures
- Regular audits and reviews
- Complaint mechanisms
- Ability to contest recommendations
- Insurance and liability consideration

**7. Informed Consent**

**Issue:**
Users may not fully understand what they're consenting to.

**Mitigation:**
- Clear, understandable consent forms
- Granular privacy controls
- Regular reminders about data usage
- Easy withdrawal of consent

**8. Environmental and Social Impact**

**Issue:**
Recommendations promoting overconsumption have environmental consequences.

**Considerations:**
- Promoting sustainable products
- Not encouraging wasteful consumption
- Social responsibility in recommendations

**Implementation Framework:**
1. Establish ethical guidelines
2. Regular ethical audits
3. Diverse ethics review board
4. Transparency reports
5. User feedback mechanisms
6. Continuous monitoring and improvement

Ethical considerations should be integrated from design through deployment and ongoing operation of recommendation systems.`,
            marks: 10,
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
        introText: `Using the model code and its corresponding output given below, critically study both and answer the following questions:

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

The code produces the following results:

**Confusion Matrix:**
\`\`\`
[[149   0]
 [ 51   0]]
\`\`\`

**Classification Report:**
\`\`\`
              precision    recall  f1-score   support

           0       0.74      1.00      0.85       149
           1       0.00      0.00      0.00        51

    accuracy                           0.74       200
   macro avg       0.37      0.50      0.43       200
weighted avg       0.55      0.74      0.64       200
\`\`\`

**Accuracy Score:** 0.745

**Box Plot Visualizations:**
- Large Purchase Distribution: Shows approximately 700 instances of class 0 (no large purchase) and 200 instances of class 1 (large purchase)
- Annual Income vs Large Purchase: Box plots show income distribution for both classes with medians around 60,000 for class 0 and 95,000 for class 1
- Purchase Frequency vs Large Purchase: Box plots show purchase frequency distribution with medians around 6 for class 0 and 7-8 for class 1

**Context:**
The model implements a Support Vector Machine (SVM) for binary classification to predict customer large purchase behavior. The dataset includes features such as age, annual income, purchase frequency, membership type (Basic, Premium, VIP), online shopping preference, and loyalty program participation.

The preprocessing includes:
- One-hot encoding for categorical variables using pd.get_dummies() with drop_first=True
- Feature scaling using StandardScaler
- 80/20 train-test split with random_state=42

The model uses a linear kernel SVM with C=1.0 parameter. The performance metrics reveal significant insights about the model's ability to classify customers who make large purchases versus those who don't.`,

        questions: [
          {
            id: "1(A)",
            question:
              "Compare the performance metrics (confusion matrix, accuracy score, and classification report) generated by the SVM model. Which metric is the most informative for assessing the model's effectiveness, and why?",
            sampleAnswer: `**Analysis of SVM Model Performance Metrics:**

**1. Confusion Matrix Analysis**

The confusion matrix shows:
- Class 0: Precision=0.74, Recall=1.00, F1=0.85
- Class 1: Precision=0.00, Recall=0.00, F1=0.00

**Interpretation:**
The model predicts ALL instances as Class 0 (no large purchases), achieving perfect recall for Class 0 but completely failing to identify Class 1 (large purchases). This indicates severe class imbalance or model bias.

**2. Accuracy Score: 0.745 (74.5%)**

While 74.5% accuracy seems reasonable, it's misleading because:
- The model achieves this by predicting everything as Class 0
- With 149 Class 0 instances and 51 Class 1 instances, predicting all Class 0 gives 149/200 = 74.5% accuracy
- **Accuracy paradox**: High accuracy despite poor predictive power

**Most Informative Metric: F1-Score (especially for Class 1)**

**Justification:**

1. **Balances Precision and Recall**
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean prevents high scores when either metric is zero
   - Class 1 F1 = 0.00 immediately reveals the problem

2. **Handles Class Imbalance**
   - Unlike accuracy, F1-score doesn't benefit from predicting majority class
   - Shows real predictive capability for each class

3. **Business Context Relevance**
   - For product recommendations, failing to identify potential large purchasers (Class 1) is critical
   - Missing high-value customers has direct business impact`,
            marks: 10,
          },
          ,
          {
            id: "1(B)",
            question:
              "Critically study the results produced by the box plot graphs and interpret the results in the context of the given model.",
            sampleAnswer: `**Box Plot Analysis for SVM Model:**

**1. Large Purchase Distribution (Left Plot)**

**Observations:**
- Two distinct groups: Class 0 (no large purchase) and Class 1 (large purchase)
- Class 0 (~700 instances) far outnumbers Class 1 (~200 instances)
- Clear class imbalance: approximately 3.5:1 ratio

**Implications:**
- **Class Imbalance Problem**: The model is biased toward predicting Class 0
- Explains why the model achieves 74.5% accuracy by predicting everything as Class 0
- Minority class (Class 1) lacks sufficient examples for the model to learn patterns
- **Action Needed**: Apply class balancing techniques (SMOTE, class weights)

**2. Annual Income vs Large Purchase (Middle Plot)**

**Key Findings:**

**Class 0 (No Large Purchase):**
- Median income: ~60,000
- Interquartile range (IQR): 50,000 to 95,000
- Some outliers above 100,000
- Wide income distribution

**Class 1 (Large Purchase):**
- Median income: ~95,000
- Interquartile range: Similar spread to Class 0
- Significant overlap with Class 0 distribution

**Critical Insights:**

**Overlap Analysis:**
The substantial overlap between income distributions suggests:
- **Income alone is insufficient** for classification
- Large purchasers exist across income levels
- The model struggles because this feature isn't discriminative enough
- Need additional features for better separation

**Why Model Performs Poorly:**
- When distributions overlap significantly, linear SVM kernel struggles
- The decision boundary cannot cleanly separate classes based on income
- This explains the model's tendency to predict all instances as one class

**3. Purchase Frequency vs Large Purchase (Right Plot)**

**Observations:**

**Class 0:**
- Median frequency: ~6 purchases
- Range: approximately 3-9 purchases
- Relatively tight distribution

**Class 1:**
- Median frequency: ~7-8 purchases
- Range: approximately 4-10 purchases
- Slight shift higher than Class 0

**Critical Analysis:**

**Modest Discrimination:**
- Purchase frequency shows better separation than income
- Class 1 generally has higher purchase frequency
- But still considerable overlap exists

**Feature Interaction:**
- Frequency alone isn't decisive
- Combination with income and other features might work better
- Suggests need for feature engineering (e.g., income × frequency)

**Overall Model Context Interpretation:**

**Why the Model Fails:**

1. **Feature Insufficiency**
   - The two primary features show too much overlap
   - Linear decision boundary cannot separate classes effectively
   - Need for:
     * More discriminative features
     * Non-linear kernel (RBF, polynomial)
     * Feature engineering

2. **Class Imbalance Impact**
   - 3.5:1 ratio heavily biases the model
   - Model learns to minimize error by predicting majority class
   - Validation: Class 1 precision, recall, and F1 all = 0.00

3. **Distribution Characteristics**
   - No clear separation in either feature dimension
   - Suggests customers' large purchase behavior depends on factors not captured
   - Potential missing features:
     * Membership type (Basic/Premium/VIP)
     * Product categories purchased
     * Time since customer acquisition
     * Online shopping preference
     * Loyalty program participation

**Model Improvement Recommendations:**

**Based on Box Plot Analysis:**

1. **Address Class Imbalance:**
   python
   # Use class_weight parameter
   svm_model = SVC(kernel='linear', class_weight='balanced')

   # Or SMOTE
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


2. **Try Non-Linear Kernel:**
   python
   # RBF kernel for non-linear boundaries
   svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')


3. **Feature Engineering:**
   - Create interaction terms: income × frequency
   - Add polynomial features
   - Normalize/standardize features (already done in the code)

4. **Additional Features Needed:**
   - Categorical features properly encoded
   - Membership type
   - Online shopping behavior
   - Temporal features

5. **Alternative Approaches:**
   - Try ensemble methods (Random Forest, Gradient Boosting)
   - Neural networks might capture complex interactions
   - Use different threshold for classification

**Conclusion:**

The box plots reveal fundamental issues:
- **Class imbalance** causes model to ignore minority class
- **Feature overlap** prevents effective separation
- **Limited features** fail to capture large purchase behavior complexity

The models poor performance (F1=0.00 for Class 1) is directly explained by these distributional characteristics. Without addressing class imbalance and adding more discriminative features, the model will continue to fail at identifying large purchasers.`,
            marks: 10,
          },
          {
            id: "1(C)",
            question:
              "Explain how the encoding of categorical variables impact the SVM model's ability to classify large_purchase accurately and identify any potential limitations or biases introduced during this step.",
            sampleAnswer: `**Impact of Categorical Variable Encoding on SVM Model:**

**Encoding Method Used:**
The code uses **pd.get_dummies()** with **drop_first=True**, which creates one-hot encoding for categorical variables while dropping the first category to avoid multicollinearity.

**1. How Encoding Impacts SVM Classification:**

**Positive Impacts:**

**A. Numerical Representation**
- SVMs require numerical input
- get_dummies() converts categories (Basic, Premium, VIP) to binary features
- Enables mathematical distance calculations in feature space
- Example transformation:

  membership_type: Premium → membership_type_Premium: 1, membership_type_VIP: 0
  membership_type: VIP → membership_type_Premium: 0, membership_type_VIP: 1
  membership_type: Basic → membership_type_Premium: 0, membership_type_VIP: 0


**B. Linear Separability**
- One-hot encoding creates axis-aligned features
- Each category becomes a dimension
- Can help SVM find hyperplane if categories are predictive
- Linear kernel can separate classes along these dimensions

**C. Feature Space Expansion**
- Increases dimensionality from 1 categorical variable to k-1 binary features
- More dimensions can allow better separation
- But can also lead to curse of dimensionality

**2. Potential Limitations:**

**A. Curse of Dimensionality**

**Issue:**
- High cardinality categoricals create many dimensions
- Example: if membership_type had 50 levels → 49 new features
- SVM performance degrades in very high dimensions

**Impact:**
- Sparse data in high-dimensional space
- Overfitting risk increases
- Computational complexity grows
- Distance metrics become less meaningful

**Mitigation:**
- Feature selection
- Dimensionality reduction (PCA)
- Target encoding for high-cardinality features

**B. Arbitrary Baseline Selection**

**Issue with drop_first=True:**
- First category (alphabetically: Basic) becomes baseline
- Premium and VIP encoded as separate features
- Basic represented implicitly (both Premium and VIP = 0)

**Potential Bias:**
python
Basic:   [0, 0]  ← implicit
Premium: [1, 0]  ← explicit
VIP:     [0, 1]  ← explicit


**Impact:**
- Asymmetric representation
- Model may treat Basic differently from Premium/VIP
- Distance calculations affected:
  * Distance(Basic, Premium) ≠ Distance(Premium, VIP)
  * All equal in original space, but not after encoding

**Better Alternative:**
python
# Use full one-hot encoding with regularization
df_encoded = pd.get_dummies(df, drop_first=False)

# Or use target encoding
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df_encoded = encoder.fit_transform(df['membership_type'], df['large_purchase'])


**C. Loss of Ordinal Relationships**

**Issue:**
If membership types have natural ordering:
- Basic < Premium < VIP (increasing value)

One-hot encoding treats them as **nominal** (unordered):
- Loses ordinal information
- Model can't learn that VIP > Premium > Basic

**Example Impact:**
python
# What we might want (ordinal):
Basic: 0, Premium: 1, VIP: 2

# What we get (nominal):
Basic: [0,0], Premium: [1,0], VIP: [0,1]


**Better for Ordinal Data:**
python
# Label encoding preserves order
membership_map = {'Basic': 0, 'Premium': 1, 'VIP': 2}
df['membership_encoded'] = df['membership_type'].map(membership_map)


**D. Imbalanced Category Representation**

**Potential Issue:**
If categorical distribution is imbalanced:

Basic: 5% of customers
Premium: 70% of customers
VIP: 25% of customers


**Impact:**
- Model may learn to ignore Basic (rare category)
- Binary features for Basic mostly 0
- Contributes to overall class imbalance problem
- Combined with target class imbalance, compounds the issue

**3. Biases Introduced:**

**A. Encoding-Induced Bias**

**Distance Metric Bias:**
In original space:
- All membership types equally different from each other

After one-hot encoding:
- Euclidean distance changes:
  * d(Basic, Premium) = √1 = 1
  * d(Basic, VIP) = √1 = 1
  * d(Premium, VIP) = √2 ≈ 1.41

**Result:** Premium and VIP treated as "more different" than either is from Basic

**B. Baseline Category Bias**

**Statistical Bias:**
- Coefficients learned for Premium and VIP are relative to Basic
- Interpretation becomes "effect of being Premium vs. Basic"
- Model implicitly uses Basic as reference point
- May not align with business logic

**C. Feature Scaling Interaction**

**Issue:**
After StandardScaler:
python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


**Impact:**
- Binary encoded features (0/1) scaled differently than continuous features
- After scaling, encoded features may have different variance
- SVM's decision boundary influenced by this scaling interaction
- Linear kernel particularly sensitive

**Example:**

Before scaling:
- income: [20000, 120000], range = 100,000
- membership_Premium: [0, 1], range = 1

After scaling:
- Both centered and scaled
- But original scale differences affect learned weights


**4. Specific Impact on Model Performance:**

**Current Model Issues Explained:**

**A. Why Class 1 (Large Purchase) Has 0.00 Precision/Recall:**

1. Categorical encoding adds dimensions but doesn't add discriminative power
2. If membership type isn't strongly predictive, encoding creates noise
3. Linear SVM struggles to find separating hyperplane
4. Model defaults to predicting majority class

**B. Feature Importance Masked:**

One-hot encoding makes it harder to assess which categories matter:
python
# Can't easily tell if membership_type matters overall
# Need to check both membership_Premium and membership_VIP coefficients


**5. Recommendations for Better Encoding:**

**A. For Current Model:**

python
# Try without dropping first
df_encoded = pd.get_dummies(df, drop_first=False)

# Then use regularization to handle multicollinearity
svm_model = SVC(kernel='linear', C=0.1)  # Lower C = more regularization


**B. Alternative Encoding Strategies:**

**1. Target Encoding:**
python
from category_encoders import TargetEncoder

# Encode based on target relationship
encoder = TargetEncoder(cols=['membership_type'])
X_encoded = encoder.fit_transform(X, y)
# Basic: 0.15 (15% large purchase rate)
# Premium: 0.45 (45% large purchase rate)
# VIP: 0.78 (78% large purchase rate)


**Benefits:**
- Captures relationship with target
- Single numerical feature
- Preserves ordinal nature if it exists
- Reduces dimensionality

**2. Frequency Encoding:**
python
freq_map = df['membership_type'].value_counts(normalize=True)
df['membership_freq'] = df['membership_type'].map(freq_map)


**3. For Ordinal Data:**
python
from sklearn.preprocessing import OrdinalEncoder

# If true ordering exists
encoder = OrdinalEncoder(categories=[['Basic', 'Premium', 'VIP']])
df['membership_encoded'] = encoder.fit_transform(df[['membership_type']])


**6. Interaction with Other Issues:**

**Compounding Effects:**


Class Imbalance (3.5:1)
       +
Categorical Encoding Bias
       +
Overlapping Feature Distributions
       =
Complete Class 1 Failure (F1=0.00)


**Conclusion:**

The encoding method significantly impacts the SVM model:

**Limitations:**
- Drop_first=True creates asymmetric representation
- One-hot encoding ignores ordinal relationships
- Increases dimensionality without necessarily adding predictive power
- Interacts poorly with class imbalance

**Biases:**
- Baseline category treated differently
- Distance metrics affected
- Feature scaling interaction creates implicit weighting

**Recommendations:**
1. Use target encoding for better relationship capture
2. Consider ordinal encoding if logical order exists
3. Don't drop first category unless multicollinearity is proven issue
4. Test multiple encoding strategies
5. Combine with class balancing techniques (SMOTE, class weights)

The current encoding contributes to the model's poor performance by failing to capture the true relationship between membership type and purchase behavior.`,
            marks: 10,
          },
          {
            id: "1(D)",
            question:
              "Based on the analysis of the prediction results, propose a strategy to improve the model's performance and suggest additional features or preprocessing steps that could enhance its predictive accuracy.",
            sampleAnswer: `**Comprehensive Strategy to Improve SVM Model Performance:**

## **Phase 1: Address Class Imbalance (Critical Priority)**

**Current Problem:**
- Class 0: 149 instances (74.5%)
- Class 1: 51 instances (25.5%)
- Model predicts everything as Class 0 to minimize error

**Solution Strategies:**

### **A. Resampling Techniques**

**1. SMOTE (Synthetic Minority Over-sampling Technique):**
python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Combine over and under sampling
over = SMOTE(sampling_strategy=0.7, random_state=42)
under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('over', over),
    ('under', under)
])

X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)

# Train SVM on balanced data
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)


**Why This Works:**
- Creates synthetic examples for minority class
- Balances dataset to 50:50 or desired ratio
- Model learns patterns from both classes equally

**2. Class Weights:**
python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train with class weights
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    class_weight=class_weight_dict,
    random_state=42
)
svm_model.fit(X_train, y_train)


**Why This Works:**
- Penalizes misclassification of minority class more
- No synthetic data creation
- Simpler than SMOTE

## **Phase 2: Improve Feature Engineering**

**Current Features:** age, annual_income, purchase_frequency, membership_type (encoded)

### **A. Additional Features to Add**

**1. Behavioral Features:**
python
# Average transaction value
df['avg_transaction_value'] = df['total_spent'] / df['purchase_frequency']

# Spending consistency
df['spending_std'] = df.groupby('customer_id')['transaction_amount'].transform('std')

# Days since last purchase
df['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days

# Customer lifetime (tenure)
df['customer_tenure_days'] = (pd.Timestamp.now() - df['registration_date']).dt.days

# Purchase velocity (purchases per month)
df['purchase_velocity'] = df['purchase_frequency'] / (df['customer_tenure_days'] / 30)

# Income to spending ratio
df['income_spending_ratio'] = df['annual_income'] / (df['total_spent'] + 1)


**2. Interaction Features:**
python
# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X[['annual_income', 'purchase_frequency']])

# Specific interactions
df['income_x_frequency'] = df['annual_income'] * df['purchase_frequency']
df['income_x_age'] = df['annual_income'] * df['age']
df['age_x_frequency'] = df['age'] * df['purchase_frequency']


**3. Temporal Features:**
python
# Extract from purchase dates if available
df['purchase_month'] = df['last_purchase_date'].dt.month
df['purchase_day_of_week'] = df['last_purchase_date'].dt.dayofweek
df['is_holiday_season'] = df['purchase_month'].isin([11, 12]).astype(int)
df['is_weekend'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)


**4. RFM Features (Recency, Frequency, Monetary):**
python
# Recency score (how recently they purchased)
df['recency_score'] = pd.qcut(df['days_since_last_purchase'],
                               q=4, labels=[4,3,2,1])

# Frequency score (normalized)
df['frequency_score'] = pd.qcut(df['purchase_frequency'],
                                 q=4, labels=[1,2,3,4], duplicates='drop')

# Monetary score (based on total spent)
df['monetary_score'] = pd.qcut(df['total_spent'],
                                q=4, labels=[1,2,3,4], duplicates='drop')

# Combined RFM score
df['rfm_score'] = (df['recency_score'].astype(int) +
                   df['frequency_score'].astype(int) +
                   df['monetary_score'].astype(int))


**5. Category Aggregation Features:**
python
# Product category preferences (if available)
df['preferred_category'] = df['product_categories'].str.split(',').str[0]
df['num_categories_shopped'] = df['product_categories'].str.split(',').str.len()

# Diversity of purchases
df['category_diversity'] = df['num_categories_shopped'] / df['purchase_frequency']


### **B. Better Categorical Encoding**

**Target Encoding for membership_type:**
python
from category_encoders import TargetEncoder

# Target encoding (captures relationship with target)
target_enc = TargetEncoder(cols=['membership_type'], smoothing=1.0)
df_encoded = target_enc.fit_transform(df['membership_type'], df['large_purchase'])

# This gives single numerical value per category based on target rate
# E.g., Basic: 0.15, Premium: 0.45, VIP: 0.75


**Weight of Evidence (WoE) Encoding:**
python
import numpy as np

def calculate_woe(df, feature, target):
    """Calculate Weight of Evidence encoding"""
    grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    grouped['non_events'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_event_rate'] = grouped['non_events'] / grouped['non_events'].sum()
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    return grouped['woe'].to_dict()

woe_map = calculate_woe(df, 'membership_type', 'large_purchase')
df['membership_woe'] = df['membership_type'].map(woe_map)


## **Phase 3: Preprocessing Improvements**

### **A. Feature Scaling Optimization**

python
from sklearn.preprocessing import RobustScaler, PowerTransformer

# RobustScaler (better for outliers than StandardScaler)
robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(X_train)

# Or PowerTransformer (makes features more Gaussian)
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_transformed = power_transformer.fit_transform(X_train)


### **B. Outlier Detection and Handling**

python
from sklearn.ensemble import IsolationForest

# Detect outliers
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X_train)

# Option 1: Remove outliers
X_train_clean = X_train[outliers == 1]
y_train_clean = y_train[outliers == 1]

# Option 2: Cap outliers
from scipy.stats import zscore
z_scores = np.abs(zscore(X_train))
X_train_capped = np.where(z_scores > 3,
                           np.sign(X_train) * 3 * X_train.std() + X_train.mean(),
                           X_train)


### **C. Feature Selection**

python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Method 1: Statistical feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)

# Method 2: Recursive Feature Elimination
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Method 3: Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features
top_features = feature_importance.head(10)['feature'].tolist()
X_train_selected = X_train[top_features]


## **Phase 4: Model Architecture Improvements**

### **A. Optimize SVM Hyperparameters**

python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Comprehensive grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4],  # for poly kernel
    'class_weight': ['balanced', None]
}

# Use RandomizedSearch for faster search
random_search = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='f1',  # Optimize for F1 score, not accuracy
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train_balanced, y_train_balanced)
best_model = random_search.best_estimator_

print("Best parameters:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)


### **B. Try Different Kernels**

python
# RBF Kernel (most common for non-linear problems)
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced')

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, C=1, class_weight='balanced')

# Sigmoid Kernel
svm_sigmoid = SVC(kernel='sigmoid', C=1, class_weight='balanced')

# Compare all kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
results = {}

for kernel in kernels:
    model = SVC(kernel=kernel, class_weight='balanced', random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results[kernel] = f1
    print(f"{kernel}: F1 = {f1:.3f}")


### **C. Ensemble Approaches**

python
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Voting Classifier (combine multiple models)
svm_model = SVC(kernel='rbf', C=10, probability=True, class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)

voting_clf = VotingClassifier(
    estimators=[('svm', svm_model), ('rf', rf_model), ('lr', lr_model)],
    voting='soft'  # Use probability-based voting
)
voting_clf.fit(X_train_balanced, y_train_balanced)

# Bagging SVM (multiple SVM models)
bagging_svm = BaggingClassifier(
    base_estimator=SVC(kernel='rbf', class_weight='balanced'),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)
bagging_svm.fit(X_train_balanced, y_train_balanced)


## **Phase 5: Evaluation Strategy**

### **A. Use Proper Metrics**

python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Comprehensive evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Focus on Class 1 metrics
print("\nClass 1 (Large Purchase) Metrics:")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

if y_pred_proba is not None:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"Average Precision: {average_precision_score(y_test, y_pred_proba):.3f}")


### **B. Cross-Validation with Stratification**

python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validate on multiple metrics
scoring = ['f1', 'precision', 'recall', 'roc_auc']

for metric in scoring:
    scores = cross_val_score(model, X_train_balanced, y_train_balanced,
                            cv=skf, scoring=metric)
    print(f"{metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")


### **C. Threshold Optimization**

python
# Find optimal threshold for classification
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_threshold))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    return optimal_threshold, optimal_f1

# Use with SVM decision function
decision_scores = model.decision_function(X_test)
optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, decision_scores)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Optimal F1: {optimal_f1:.3f}")

# Apply optimal threshold
y_pred_optimized = (decision_scores >= optimal_threshold).astype(int)


## **Phase 6: Alternative Models to Consider**

python
# 1. Gradient Boosting (often outperforms SVM)
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 2. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# 3. Neural Network
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train_balanced, y_train_balanced)

# Compare all models
models = {
    'SVM (Improved)': improved_svm,
    'XGBoost': xgb_model,
    'Random Forest': rf_model,
    'Neural Network': nn_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}\n")


## **Complete Implementation Pipeline**

python
# Full improved pipeline
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=15))
])

# Create full pipeline with SMOTE
full_pipeline = ImbPipeline([
    ('preprocessing', preprocessing_pipeline),
    ('smote', SMOTE(sampling_strategy=0.8, random_state=42)),
    ('classifier', SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced'))
])

# Train
full_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = full_pipeline.predict(X_test)
print("Improved Model Performance:")
print(classification_report(y_test, y_pred))


## **Expected Improvements**

With these strategies, you should see:

**Before:**
- Class 1 F1-Score: 0.00
- Class 1 Precision: 0.00
- Class 1 Recall: 0.00
- Accuracy: 0.745 (misleading)

**After (Expected):**
- Class 1 F1-Score: 0.65-0.80
- Class 1 Precision: 0.60-0.75
- Class 1 Recall: 0.70-0.85
- Balanced Accuracy: 0.75-0.85

## **Summary of Strategy**

**Priority 1 (Critical):**
1. Address class imbalance with SMOTE + class weights
2. Switch to F1-score as primary metric

**Priority 2 (High Impact):**
3. Add behavioral and interaction features
4. Improve categorical encoding (target encoding)
5. Hyperparameter optimization with GridSearch

**Priority 3 (Refinement):**
6. Feature selection to remove noise
7. Try different kernels (RBF likely best)
8. Consider ensemble methods

**Priority 4 (Alternative):**
9. Test Gradient Boosting (XGBoost, LightGBM)
10. Threshold optimization for final predictions

This comprehensive approach should transform the model from completely failing (F1=0.00) to actually being useful for identifying large purchasers.`,
            marks: 10,
          },
        ],
      },
    ],
  },
];
