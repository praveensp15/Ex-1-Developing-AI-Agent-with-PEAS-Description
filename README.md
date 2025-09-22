# Ex-1-Developing-AI-Agent-with-PEAS-Description
### Name:praveen s

### Register Number:2305001027

### Aim:
To find the PEAS description for the given AI problem and develop an AI agent.

### Theory :
PEAS stands for:
'''
P-Performance measure

E-Environment

A-Actuators

S-Sensors
'''

It’s a framework used to define the task environment for an AI agent clearly.

### Pick an AI Problem

```

1. Self-driving car

2. Chess playing agent

3. Vacuum cleaning robot

4. Email spam filter

5. Personal assistant (like Siri or Alexa)
```

### Email spam filter
### Algorithm:
Step 1-Input: Collect dataset of emails labeled as spam/ham.

Step 2-Preprocessing: Clean text (lowercase, remove stopwords, punctuation).

Step 3-Feature Extraction: Convert text to numerical form using TF-IDF.

Step 4-Split Data: Train–test split.

Step 5-Train Model: Apply Naive Bayes classifier on training data.

Step 6-Prediction: For a new email, compute probabilities:

If 
𝑃
(
spam
∣
message
)
>
𝑃
(
ham
∣
message
)
P(spam∣message)>P(ham∣message) → classify as Spam

Step 7-Else → classify as Ham

Step 8-Evaluation: Check performance with Accuracy, Precision, Recall, F1-score.

Step 9-Output: Spam = 1, Ham = 0.

### Program:
```
df = pd.read_csv("/content/Ai ex 1 spam email .zip", encoding="latin-1")
df = df[['v1', 'v2']] 
df.columns = ['label', 'message']


df['label_num'] = df.label.map({'ham':1, 'spam':2})


X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.5, random_state=43
)


vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


sample = ["Congratulations! You've won a $1000 Walmart gift card. Click here!",
          "Hey, are we still meeting for lunch tomorrow?"]
sample_tfidf = vectorizer.transform(sample)
print("\nPrediction:", model.predict(sample_tfidf))
```
### Sample Output:

<img width="550" height="393" alt="Screenshot 2025-09-22 135236" src="https://github.com/user-attachments/assets/abbeffe7-8f3d-44a5-b79d-4114101b482a" />


### Result:
Thus the given program for Email spam filter was implemented and executed successfully.
