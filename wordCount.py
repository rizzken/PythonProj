import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv
import re

# Load data
df = pd.read_csv('ims_galaxy.csv')
df = df.astype(str)
# Combine all descriptions
text = ' '.join(df['Description'])


# Specify the filename
filename = 'most_common_new.csv'

# Read the data from the CSV file
galaxy_stop_words = []
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader)  # Skip the header row
    for row in reader:
        gword = row[0]
        galaxy_stop_words.append((gword))


# Preprocessing
stop_words = set(stopwords.words('english'))
stop_words.update(galaxy_stop_words)
# Efficiently add custom stopwords
ps = PorterStemmer()

# Precompile regex patterns for stop words
stop_word_patterns = [re.compile(f'^{sw}.*') for sw in stop_words]

def preprocess(text, stop_word_patterns):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and stem
    processed_tokens = []
    for word in tokens:
        if not any(pattern.match(word) for pattern in stop_word_patterns) and not word.isdigit():
            processed_tokens.append(ps.stem(word))
    return processed_tokens

# Preprocess text using precompiled patterns
words = preprocess(text, stop_word_patterns)

# Word frequency
word_freq = Counter(words)

# Get the 20 most common words
most_common = word_freq.most_common(15)
print (most_common)
'''
# Specify the filename
filename = 'most_common.csv'

# Write the data to a CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['word', 'frequency'])  # Write the header
    writer.writerows(most_common)  # Write the data
'''
# Prepare data for plotting
words, counts = zip(*most_common)

# Truncate words to first 5 characters for visualization
truncated_words = [word[:10] for word in words]

plt.figure(figsize=(12, 6))
plt.bar(truncated_words, counts)
plt.title('Top 15 Most Common Words in Bug Descriptions')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show plot
plt.show()

# Truncate words to first 5 characters
truncated_word_freq = {word[:10]: freq for word, freq in word_freq.items()}

# Create word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(truncated_word_freq)

# Display the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Top Words (Truncated to 10 Characters)')
plt.show()