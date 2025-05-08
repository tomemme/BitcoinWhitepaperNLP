# -*- coding: utf-8 -*-
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import textstat
import matplotlib.pyplot as plt  # For visualizations

nltk.download('punkt')
nltk.download('punkt_tab')

# Function to extract text from a PDF
def get_bitcoin_whitepaper_text(file_path):
    pdf_file = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    pdf_file.close()
    return text

# Clean text while preserving some punctuation for analysis
def preprocess_text(text, keep_punctuation=False):
    text = re.sub(r'\s+', ' ', text).strip()
    if not keep_punctuation:
        text = re.sub(r'[^\w\s.]', '', text)  # Keep only words, spaces, periods
    return text.lower()

# Enhanced language analysis with stylometric features
def analyze_language(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_freq = Counter(words)
    
    # N-grams (bigrams: 2-word phrases)
    bigrams = list(ngrams(words, 2))
    bigram_freq = Counter(bigrams).most_common(10)
    
    # Function words (common in stylometry)
    function_words = {'the', 'of', 'to', 'and', 'in', 'is', 'a', 'that', 'for', 'on'}
    func_word_usage = {w: word_freq[w] for w in function_words if w in word_freq}
    
    # Spelling variants (British vs. American)
    spelling_clues = {
        'British': len(re.findall(r'our\b|favour|ise\b', text)),  # e.g., "favour", "realise"
        'American': len(re.findall(r'or\b|favor|ize\b', text))    # e.g., "favor", "realize"
    }
    
    # Syntactic complexity: proportion of long sentences
    long_sentences = len([s for s in sentences if len(word_tokenize(s)) > 20])
    complexity = long_sentences / len(sentences) if sentences else 0
    
    return {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "most_common": word_freq.most_common(10),
        "avg_sentence_length": sum(len(word_tokenize(sent)) for sent in sentences) / len(sentences),
        "readability": textstat.flesch_reading_ease(text),
        "interesting_words": [w for w in set(words) if len(w) > 7 and word_freq[w] > 1][:10],
        "word_freq": word_freq,
        "bigrams": bigram_freq,
        "function_words": func_word_usage,
        "spelling_clues": spelling_clues,
        "syntactic_complexity": complexity
    }

# Visualize word frequencies
def plot_word_freq(word_freq, title="Word Frequency"):
    words, counts = zip(*word_freq.most_common(10))
    plt.bar(words, counts)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function with enhanced output
def main():
    file_path = "/Users/temmerling/projects/btcWPnlp/bitcoin.pdf"  # Update as needed
    print(f"Analyzing Bitcoin white paper from {file_path}...")
    text = get_bitcoin_whitepaper_text(file_path)
    clean_text = preprocess_text(text)  # For basic analysis
    raw_text = preprocess_text(text, keep_punctuation=True)  # For spelling/style
    
    # Run analysis
    analysis = analyze_language(clean_text)
    
    # Forensic linguistic output
    print("\nGreetings, linguistic detectives! Let’s unravel the Bitcoin white paper’s secrets.")
    print(f"Lexicon: {analysis['total_words']} words, {analysis['unique_words']} unique.")
    print(f"Top words: {analysis['most_common']}")
    print(f"Avg. sentence length: {round(analysis['avg_sentence_length'], 1)} words.")
    print(f"Readability (Flesch): {round(analysis['readability'], 1)}—technical yet clear.")
    print(f"Top bigrams: {analysis['bigrams']}")
    print(f"Function word usage: {analysis['function_words']}")
    print(f"Spelling clues: British-like: {analysis['spelling_clues']['British']}, American-like: {analysis['spelling_clues']['American']}")
    print(f"Syntactic complexity (long sentences): {round(analysis['syntactic_complexity'] * 100, 1)}%")
    print("\nIntriguing terms:")
    print("Word".ljust(20) + "Frequency".ljust(10))
    for word, freq in [(w, analysis['word_freq'][w]) for w in analysis['interesting_words']]:
        print(word.ljust(20) + str(freq).ljust(10))
    
    # Plot
    plot_word_freq(analysis['word_freq'])
    
    # Linguistic insights
    print("\nStylometric Notes:")
    if analysis['spelling_clues']['British'] > analysis['spelling_clues']['American']:
        print("- Hints of British English (e.g., 'favour') suggest a non-American influence.")
    print(f"- Heavy use of technical terms like {analysis['most_common'][7][0]} indicates a cryptography or programming background.")
    print(f"- Bigrams like {analysis['bigrams'][0][0]} reflect a focus on {analysis['bigrams'][0][0][1]}.")

if __name__ == "__main__":
    main()