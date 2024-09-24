from nltk.stem.porter import PorterStemmer
import nltk
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the NLTK data directory within the script directory
nltk_data_dir = os.path.join(script_dir, 'nltk_data')

# Create the nltk_data directory if it doesn't exist
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK to use the custom directory
nltk.data.path.append(nltk_data_dir)

# Print the current NLTK data paths
# print("Current NLTK data paths:", nltk.data.path)

# Download 'punkt' tokenizer and 'punkt_tab' to the specified directory
# try:
#     nltk.download('punkt', download_dir=nltk_data_dir)
#     print("Punkt tokenizer downloaded successfully.")
# except Exception as e:
#     print("Error downloading punkt:", e)

# try:
#     nltk.download('punkt_tab', download_dir=nltk_data_dir)
#     print("Punkt_tab tokenizer downloaded successfully.")
# except Exception as e:
#     print("Error downloading punkt_tab:", e)

# Initialize the Porter Stemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """Tokenizes the input sentence into words."""
    return nltk.word_tokenize(sentence)


def stem(word):
    """Returns the stem of a given word."""
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Creates a bag-of-words representation.

    Parameters:
    - tokenized_sentence: A list of tokenized words from a sentence.
    - all_words: A list of all possible words in the vocabulary.

    Returns:
    A dictionary with words as keys and 1s or 0s as values, indicating presence.
    """
    # Initialize a bag of words representation
    bag = {word: 0 for word in all_words}  # All words initialized to 0

    for word in tokenized_sentence:
        if word in bag:
            bag[word] = 1  # Set to 1 if the word is found

    return bag


# Example usage
if __name__ == "__main__":
    # Input sentence
    a = "How long does shipping take?"
    print("Original Sentence:", a)

    # Tokenization
    a_tokenized = tokenize(a)
    print("Tokenized Sentence:", a_tokenized)

    # Example of stemming
    # stemmed_words = [stem(word) for word in a_tokenized]
    # print("Stemmed Tokens:", stemmed_words)

    # # Example vocabulary
    # vocabulary = ["how", "long", "shipping", "take", "does"]

    # # Bag of words representation
    # bag = bag_of_words(stemmed_words, vocabulary)
    # print("Bag of Words Representation:", bag)
