import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
import sys
import spacy
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# Function to load spaCy model
import spacy
import sys
import os

def load_spacy_model(model_name):
    try:
        # Try loading the model directly
        return spacy.load(model_name)
    except OSError:
        try:
            # Construct the path for packaged applications
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_path, 'spacy/data', model_name, model_name + '-3.7.1')
            return spacy.load(model_path)
        except Exception as e:
            print(f"Error loading spaCy model from {model_path}: {e}")
            raise

nlp = load_spacy_model('en_core_sm')
def is_transition_word(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if 'conjunction' in synset.lexname():
            return True
    return False

def is_pronoun(word):
    tagged_word = nltk.pos_tag([word])
    if len(tagged_word) > 0:
        pos_tag = tagged_word[0][1]
        if pos_tag in ['PRP', 'PRP$', 'WP', 'WP$']:
            return True
    return False

def calculate_cohesion(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 100, []  # If there's only one sentence, cohesion is perfect by default

    stop_words = set(stopwords.words('english') + list(string.punctuation))
    cohesion_score = 0
    low_cohesion_sentences = []

    for i in range(len(sentences) - 1):
        doc1 = nlp(sentences[i])
        doc2 = nlp(sentences[i + 1])

        # Extract last sentence non-stop words as tokens
        end_tokens_first_sentence = [token for token in doc1 if token.text.lower() not in stop_words]

        # Extract first sentence 5 last non-stop words as tokens
        start_tokens_next_sentence = [token for token in doc2 if token.text.lower() not in stop_words][:5]

        # Process the first sentence to only have key words
        key_ideas_first_sentence = [token.lemma_ for token in end_tokens_first_sentence if token.pos_ in {'NOUN', 'VERB', 'ADJ'}]

        # Process start of next sentence to only have key words
        key_ideas_next_sentence = [token.lemma_ for token in start_tokens_next_sentence if token.pos_ in {'NOUN', 'VERB', 'ADJ'}]

        # Calculate semantic similarity between the key ideas at the end of one sentence and the start of the next
        if key_ideas_first_sentence and key_ideas_next_sentence:
            similarities = [nlp(word1).similarity(nlp(word2)) for word1 in key_ideas_first_sentence for word2 in key_ideas_next_sentence]
            # Check for transition words and pronoun usage
            transition_presence = any(is_transition_word(token.text.lower()) for token in start_tokens_next_sentence)
            pronoun_continuity = any(is_pronoun(token.text.lower()) for token in start_tokens_next_sentence) and any(is_pronoun(token.text.lower()) for token in end_tokens_first_sentence)
            # Include transition presence and pronoun continuity in the calculation of sentence_coherence
            transition_factor = 0.5 if transition_presence else 0
            pronoun_factor = 0.5 if pronoun_continuity else 0
            similarity_factor = max(similarities) # Highest similarity between the two sentences
            sentence_coherence = min((similarity_factor + transition_factor + pronoun_factor), 1)
        else:
            sentence_coherence = 0

        cohesion_score += sentence_coherence

        if sentence_coherence < 0.8:  # Adjusted threshold for low cohesion
            low_cohesion_sentences.append((i, i+1))

    adjusted_cohesion_score = min(((cohesion_score / (len(sentences) - 1)) * 100), 100)
    return adjusted_cohesion_score, low_cohesion_sentences

def calculate_coherence_and_identify_themes(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 100, [], []  # If there's only one sentence, coherence is perfect by default and no themes to identify

    stop_words = set(stopwords.words('english') + list(string.punctuation))
    words = [nlp(word.lower()) for sentence in sentences for word in word_tokenize(sentence) if word.lower() not in stop_words and nlp(word.lower())[0].pos_ in {'NOUN'}]

    # Find the most common words based on word frequency
    word_freq = {}
    for word in words:
        lemma = nlp(word.text.lower())[0].lemma_
        if lemma in word_freq:
            word_freq[lemma] += 1
        else:
            word_freq[lemma] = 1

    # Identify themes based on word frequency
    sorted_lemmas = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    themes = [lemma for lemma, count in sorted_lemmas if count > 1][:5]  # Only consider lemmas that appear more than once

    # Calculate coherence score based on the presence of theme-related words in each sentence
    sentence_theme_presence = [any(max(nlp(theme_word).similarity(nlp(word.lower())) for theme_word in themes) > 0.8 or nlp(word.lower())[0].pos_ == 'PRON' for word in word_tokenize(sentence)[:5]) for sentence in sentences]
    coherence_score = (sum(sentence_theme_presence) / len(sentences)) * 100
    low_coherence_sentences = [i for i, has_theme in enumerate(sentence_theme_presence) if not has_theme]  # Identify sentences with low coherence

    return coherence_score, low_coherence_sentences, themes
import tkinter as tk
from tkinter import scrolledtext
from tkinter import font as tkFont

def evaluate_text(text):
    sentences = sent_tokenize(text)
    cohesion, low_cohesion_pairs = calculate_cohesion(text)
    coherence, low_coherence_sentences, themes = calculate_coherence_and_identify_themes(text)
    
    feedback = "=================\n"
    feedback += f"Cohesion Grade: {cohesion:.2f}%, Coherence Grade: {coherence:.2f}%\n"
    feedback += "=================\n"
    if low_cohesion_pairs:
        feedback += "Low cohesion between the following sentence pairs:\n"
        for pair in low_cohesion_pairs:
            feedback += f"Sentence {pair[0]+1} -> {pair[1]+1}: {sentences[pair[0]][-25:]}... {sentences[pair[1]][:25]}\n"
    feedback += "=================\n"
    if low_coherence_sentences:
        feedback += "Sentences with low coherence to the themes:\n"
        for index in low_coherence_sentences:
            feedback += f"Sentence {index+1}: {sentences[index][:25]}...\n"
    feedback += "=================\n"
    if themes:
        feedback += "Identified themes: " + ", ".join(themes) + "\n"
    feedback += "=================\n"

    return feedback

def submit_text():
    user_input = text_input.get("1.0", tk.END)
    result = evaluate_text(user_input)
    result_display.config(state=tk.NORMAL)
    result_display.delete('1.0', tk.END)
    result_display.insert(tk.END, result)
    result_display.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Text Evaluation Tool")

# Define font styles
default_font = tkFont.Font(family="Helvetica", size=12)
large_font = tkFont.Font(family="Helvetica", size=14)

text_input_label = tk.Label(root, text="Enter your text:", font=large_font)
text_input_label.pack()

text_input = scrolledtext.ScrolledText(root, height=10, font=default_font)
text_input.pack()

submit_button = tk.Button(root, text="Evaluate", command=submit_text, font=large_font)
submit_button.pack()

result_display = scrolledtext.ScrolledText(root, height=15, state=tk.DISABLED, font=default_font)
result_display.pack()

root.mainloop()

