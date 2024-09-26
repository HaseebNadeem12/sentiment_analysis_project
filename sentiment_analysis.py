# Libraries
import pandas as pd
import nltk
import re

# Data Preprocessing Libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-> Checking that data (stopwords, tokenizer) is downloaded properly or not
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Porter Stemmer and WordNet Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#-> Data cleaning function with all the steps
def preprocess_text(text):

    #-> checking that data contain text or not
    if isinstance(text, str):
        # 1. Convert all text to lowercase
        text = text.lower()

        # 2. Remove URLs, if any
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

        # 3. Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)

        # 4. Tokenize text
        word_tokens = word_tokenize(text)

        # 5. Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in word_tokens if word not in stop_words]

        # 6. Apply stemming and lemmatization
        stemmed_words = [stemmer.stem(word) for word in words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

        # Join back into a single string
        return " ".join(lemmatized_words)

    # return empty string if input is not valid
    return " "

# Function to classify sentiment based on compound score
def classify_sentiment(score):
    if score['compound'] >= 0.05:
        return 1  # positive sentiment as 1
    elif score['compound'] <= -0.05:
        return -1  # negative sentiment as -1
    else:
        return 0  # neutral sentiment as 0


# loading the data file of mobile review
my_file = pd.read_csv("C:/Users/COMTECH COMPUTER/Desktop/redmi6.csv", encoding='ISO-8859-1')

# Apply preprocessing to the 'Comments' column
my_file['Comments'] = my_file['Comments'].apply(preprocess_text)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Generate sentiment scores and classify sentiment for each comment
my_file['Sentiment_Score'] = my_file['Comments'].apply(lambda x: sia.polarity_scores(x))
my_file['Sentiment_Label'] = my_file['Sentiment_Score'].apply(classify_sentiment)

# Split the data into features (X) and target labels (y)
# TF-IDF Vectorizer for feature extraction
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(my_file['Comments']).toarray()
y = my_file['Sentiment_Label']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#-> Checking the model outcome, by predicting the unseen data
y_predictions = model.predict(X_test)
# print(y_test.head(15), "\t", y_predictions)

#-> Checking the model's accuracy
accuracy = accuracy_score(y_test, y_predictions)
#-> Calculating the model's accuracy with the help of formula
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predictions))

# Topic Modeling with LDA
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
count_matrix = count_vectorizer.fit_transform(my_file['Comments'])

#-> Initialization of LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(count_matrix)

# Display the topics
n_top_words = 10
feature_names = count_vectorizer.get_feature_names_out()

# Create a dictionary that maps each topic index to a descriptive label
topic_labels = {
    0: "Camera and Battery",
    1: "General Quality and Experience",
    2: "Price and Camera Quality",
    3: "Phone Brand and Features",
    4: "Battery Life and Value for Money" }

# Function to assign the dominant topic (theme) to each comment
def assign_theme(comment):
    # Transform the comment into the topic space
    comment_vector = count_vectorizer.transform([comment])
    topic_probabilities = lda.transform(comment_vector)
    # Get the index of the most probable topic
    dominant_topic = topic_probabilities.argmax()
    # Return the descriptive label for the dominant topic
    return topic_labels.get(dominant_topic, "Other")  # Default to "Other" if no match

# Apply the theme assignment to each comment
my_file['Theme'] = my_file['Comments'].apply(assign_theme)

# Iterate through each topic in the LDA model
for topic_idx in range(len(lda.components_)):
    print(topic_labels[topic_idx])
    # Get the words related to the topic
    topic = lda.components_[topic_idx]

    # Sort the words in the topic based on their importance (largest to smallest)
    # Reverse the order to get the most important words first
    sorted_word_indices = topic.argsort()[::-1]

    # Print the top N words in the topic
    top_words = []
    for i in sorted_word_indices[:n_top_words]:
        top_words.append(feature_names[i])

    # Print the top words for this topic
    print(top_words)

# Save the updated DataFrame with sentiment scores and labels
my_file.to_csv("C:/Users/COMTECH COMPUTER/Desktop/new_redmi6_with_sentiments.csv", index=False)

# Save the topics and sentiment results to a structured JSON format
import json

results = []
for idx, row in my_file.iterrows():
    result = {
        "Comment"        : row['Comments'],
        "Sentiment_Label": row['Sentiment_Label'],
        "Sentiment_Score": row['Sentiment_Score']['compound'],
        "Rating"         : row['Rating'] }
    results.append(result)

# Save to a JSON file
with open('sentiment_analysis_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Sentiment analysis results saved to sentiment_analysis_results.json")


