from nltk.stem.porter import PorterStemmer
import nltk
import os
import numpy as np

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the NLTK data directory within the script directory
nltk_data_dir = os.path.join(script_dir, 'nltk_data')

# Create the nltk_data directory if it doesn't exist
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK to use the custom directory
nltk.data.path.append(nltk_data_dir)

#################### //uncomment to download punkt file, if not exists #############################

# # Print the current NLTK data paths
# print("Current NLTK data paths:", nltk.data.path)

# # Download 'punkt' tokenizer and 'punkt_tab' to the specified directory
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

####################################################################################################

# Initialize the Porter Stemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """Tokenizes the input sentence into words."""
    return nltk.word_tokenize(sentence)


def stem(word):
    """Returns the stem of a given word."""
    return stemmer.stem(word.lower())


# def bag_of_words(tokenized_sentence, all_words):
#     """
#     Creates a bag-of-words representation.

#     Parameters:
#     - tokenized_sentence: A list of tokenized words from a sentence.
#     - all_words: A list of all possible words in the vocabulary.

#     Returns:
#     A dictionary with words as keys and the count as values, indicating presence.
#     """
#     # Initialize a bag of words representation
#     bag = {word: 0 for word in all_words}  # All words initialized to 0

#     for word in tokenized_sentence:
#         if word in bag:
#             bag[word] += 1  # Set to 1 if the word is found

#     return bag

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # print("inside tokenized ", tokenized_sentence)

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            # print(w)
            bag[idx] = 1.0

    return bag


# Example usage
if __name__ == "__main__":
    # Input sentence
    a = "How long does shipping take? how how"
    print("Original Sentence:", a)

    # Tokenization
    a_tokenized = tokenize(a)
    print("Tokenized Sentence:", a_tokenized)

    # Example of stemming
    # stemmed_words = ["organize", "organizing", "organizer"]
    stemmed_words = [stem(word) for word in a_tokenized]
    print("Stemmed Tokens:", stemmed_words)

    # Example vocabulary
    vocabulary = ["how", "long", "ship", "take", "does"]

    # Bag of words representation
    bag = bag_of_words(stemmed_words, vocabulary)
    print("Bag of Words Representation:", bag)
