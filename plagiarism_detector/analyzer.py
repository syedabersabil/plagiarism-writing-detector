"""Text Analysis Module for Writing Style and Vocabulary Detection"""

import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('stopwords')


class TextAnalyzer:
    """Comprehensive text analysis for style and vocabulary"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def analyze(self, text):
        """Perform comprehensive text analysis"""
        return {
            'basic_stats': self.get_basic_stats(text),
            'vocabulary': self.analyze_vocabulary(text),
            'complexity': self.analyze_complexity(text),
            'style_markers': self.detect_style_markers(text),
            'readability': self.calculate_readability(text)
        }
    
    def get_basic_stats(self, text):
        """Extract basic text statistics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        clean_words = [w for w in words if w.isalnum()]
        
        return {
            'char_count': len(text),
            'word_count': len(clean_words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in clean_words) / len(clean_words) if clean_words else 0,
            'avg_sentence_length': len(clean_words) / len(sentences) if sentences else 0
        }
    
    def analyze_vocabulary(self, text):
        """Analyze vocabulary diversity and complexity"""
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum()]
        
        if not clean_words:
            return {'vocabulary_level': 'N/A', 'diversity': 0, 'unique_words': 0}
        
        unique_words = set(clean_words)
        word_freq = Counter(clean_words)
        
        # Calculate Type-Token Ratio (vocabulary diversity)
        ttr = len(unique_words) / len(clean_words)
        
        # Vocabulary level based on word length
        avg_word_len = sum(len(w) for w in clean_words) / len(clean_words)
        if avg_word_len < 4.5:
            level = 'Basic'
        elif avg_word_len < 5.5:
            level = 'Intermediate'
        else:
            level = 'Advanced'
        
        # Most common words
        common_words = word_freq.most_common(10)
        
        return {
            'vocabulary_level': level,
            'type_token_ratio': round(ttr, 3),
            'unique_words': len(unique_words),
            'total_words': len(clean_words),
            'avg_word_length': round(avg_word_len, 2),
            'most_common_words': common_words,
            'vocabulary_richness': round(ttr * 100, 2)
        }
    
    def analyze_complexity(self, text):
        """Analyze sentence and syntactic complexity"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum()]
        
        # Sentence length variance
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        sent_variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths) if sent_lengths else 0
        
        # Word complexity
        syllable_count = sum(self.count_syllables(w) for w in clean_words if w.isalpha())
        
        return {
            'avg_sentence_length': round(avg_sent_len, 2),
            'sentence_length_variance': round(sent_variance, 2),
            'syllable_count': syllable_count,
            'avg_syllables_per_word': round(syllable_count / len(clean_words), 2) if clean_words else 0,
            'sentence_count': len(sentences),
            'max_sentence_length': max(sent_lengths) if sent_lengths else 0,
            'min_sentence_length': min(sent_lengths) if sent_lengths else 0
        }
    
    def calculate_readability(self, text):
        """Calculate Flesch-Kincaid and other readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum()]
        
        if not sentences or not clean_words:
            return {'flesch_kincaid_grade': 0, 'flesch_reading_ease': 0}
        
        # Count syllables
        syllable_count = sum(self.count_syllables(w) for w in clean_words if w.isalpha())
        
        # Flesch-Kincaid Grade
        fk_grade = (
            0.39 * (len(clean_words) / len(sentences)) +
            11.8 * (syllable_count / len(clean_words)) -
            15.59
        )
        
        # Flesch Reading Ease
        fre = (
            206.835 -
            1.015 * (len(clean_words) / len(sentences)) -
            84.6 * (syllable_count / len(clean_words))
        )
        
        return {
            'flesch_kincaid_grade': round(max(0, fk_grade), 2),
            'flesch_reading_ease': round(max(0, min(100, fre)), 2),
            'reading_difficulty': self.interpret_reading_difficulty(fre)
        }
    
    def detect_style_markers(self, text):
        """Detect writing style characteristics"""
        text_lower = text.lower()
        words = word_tokenize(text.lower())
        
        markers = []
        
        # Formality markers
        if any(word in text_lower for word in ['therefore', 'moreover', 'furthermore', 'consequently']):
            markers.append('formal')
        if any(word in text_lower for word in ['gonna', 'wanna', "can't", "don't"]):
            markers.append('informal')
        
        # Emotion markers
        emotional_words = ['amazing', 'terrible', 'beautiful', 'horrible', 'love', 'hate']
        if any(word in text_lower for word in emotional_words):
            markers.append('emotional')
        
        # Technical markers
        if any(word in text_lower for word in ['algorithm', 'function', 'system', 'process']):
            markers.append('technical')
        
        # Passive vs Active voice
        passive_ratio = self.calculate_passive_voice_ratio(text)
        if passive_ratio > 0.3:
            markers.append('passive')
        else:
            markers.append('active')
        
        return markers
    
    def count_syllables(self, word):
        """Estimate syllable count in a word"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        
        return max(1, count)
    
    def calculate_passive_voice_ratio(self, text):
        """Estimate passive voice usage"""
        sentences = sent_tokenize(text)
        passive_count = 0
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Simple heuristic: look for "to be" + past participle
            for i, (word, tag) in enumerate(pos_tags):
                if word.lower() in ['is', 'are', 'was', 'were', 'be', 'been']:
                    if i + 1 < len(pos_tags) and pos_tags[i + 1][1] == 'VBN':
                        passive_count += 1
        
        return passive_count / len(sentences) if sentences else 0
    
    def interpret_reading_difficulty(self, fre_score):
        """Interpret Flesch Reading Ease score"""
        if fre_score >= 90:
            return 'Very Easy (5th grade)'
        elif fre_score >= 80:
            return 'Easy (6th grade)'
        elif fre_score >= 70:
            return 'Fairly Easy (7th grade)'
        elif fre_score >= 60:
            return 'Standard (8th-9th grade)'
        elif fre_score >= 50:
            return 'Fairly Difficult (10th-12th grade)'
        elif fre_score >= 30:
            return 'Difficult (College)'
        else:
            return 'Very Difficult (Graduate)'
    
    def extract_ngrams(self, text, n=3):
        """Extract n-grams from text"""
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum()]
        
        ngrams = []
        for i in range(len(clean_words) - n + 1):
            ngrams.append(' '.join(clean_words[i:i+n]))
        
        return Counter(ngrams).most_common(10)
    
    def get_vocabulary_level(self, text):
        """Quick vocabulary level assessment"""
        analysis = self.analyze_vocabulary(text)
        return analysis['vocabulary_level']
    
    def calculate_ttr(self, text):
        """Calculate Type-Token Ratio"""
        words = word_tokenize(text.lower())
        clean_words = [w for w in words if w.isalnum()]
        return len(set(clean_words)) / len(clean_words) if clean_words else 0
    
    def get_sentence_complexity(self, text):
        """Get sentence complexity metrics"""
        complexity = self.analyze_complexity(text)
        return {
            'avg_length': complexity['avg_sentence_length'],
            'variance': complexity['sentence_length_variance'],
            'difficulty': 'Complex' if complexity['avg_sentence_length'] > 20 else 'Simple'
        }


if __name__ == '__main__':
    analyzer = TextAnalyzer()
    test_text = """Artificial intelligence is transforming the world. 
    Machine learning algorithms enable computers to learn from data. 
    Deep learning networks process information like human brains do."""
    
    results = analyzer.analyze(test_text)
    for key, value in results.items():
        print(f"{key}: {value}")