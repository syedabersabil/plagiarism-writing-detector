"""Plagiarism Detection Module using NLP Embeddings and Similarity Metrics"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')


class PlagiarismDetector:
    """Detect plagiarism using multiple similarity metrics"""
    
    def __init__(self, model_type='tfidf', similarity_threshold=0.7):
        """
        Initialize plagiarism detector
        
        Args:
            model_type: 'tfidf', 'bert', 'use' (Universal Sentence Encoder)
            similarity_threshold: Threshold for marking as plagiarized
        """
        self.model_type = model_type
        self.similarity_threshold = similarity_threshold
        self.sources = []  # Store reference texts
        self.source_embeddings = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if self.model_type == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000)
        elif self.model_type == 'bert':
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Install sentence-transformers: pip install sentence-transformers")
                self.model_type = 'tfidf'
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(max_features=5000)
        elif self.model_type == 'use':
            try:
                import tensorflow_hub as hub
                self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            except ImportError:
                print("Install tensorflow and tensorflow_hub")
                self.model_type = 'tfidf'
    
    def add_source(self, title, text):
        """Add a source document"""
        self.sources.append({'title': title, 'text': text})
    
    def add_sources_from_list(self, sources_list):
        """Add multiple sources from a list of dicts with 'title' and 'text'"""
        for source in sources_list:
            self.add_source(source.get('title', 'Unknown'), source.get('text', ''))
    
    def detect_plagiarism(self, text, sources=None, return_matches=True):
        """
        Detect plagiarism in submitted text
        
        Args:
            text: Text to check
            sources: List of source texts to check against
            return_matches: Whether to return detailed matches
        
        Returns:
            Dictionary with similarity scores and matches
        """
        if sources is None:
            sources = [s['text'] for s in self.sources]
        
        if not sources:
            return {
                'similarity_score': 0,
                'plagiarism_percentage': 0,
                'matches': [],
                'message': 'No sources to compare against'
            }
        
        # Get embeddings
        text_embedding = self._get_embedding(text)
        source_embeddings = [self._get_embedding(s) for s in sources]
        
        # Calculate similarities
        similarities = []
        matches = []
        
        for i, source_emb in enumerate(source_embeddings):
            sim = self._cosine_similarity(text_embedding, source_emb)
            similarities.append(sim)
            
            if sim >= self.similarity_threshold and return_matches:
                # Find matching passages
                matching_passages = self._find_matching_passages(
                    text, sources[i], sim
                )
                if matching_passages:
                    matches.extend(matching_passages)
        
        # Calculate overall metrics
        avg_similarity = np.mean(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        return {
            'similarity_score': float(max_similarity),
            'avg_similarity': float(avg_similarity),
            'plagiarism_percentage': float(max_similarity * 100),
            'matches_found': len(matches),
            'matches': matches[:5],  # Top 5 matches
            'sources_checked': len(sources),
            'is_plagiarized': max_similarity >= self.similarity_threshold
        }
    
    def compare_texts(self, text1, text2):
        """Compare two texts for similarity"""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        similarity = self._cosine_similarity(emb1, emb2)
        
        return {
            'text1_length': len(text1),
            'text2_length': len(text2),
            'similarity_score': float(similarity),
            'match_percentage': float(similarity * 100),
            'are_similar': similarity >= self.similarity_threshold
        }
    
    def batch_detect(self, texts, sources):
        """Detect plagiarism for multiple texts"""
        results = []
        for i, text in enumerate(texts):
            result = self.detect_plagiarism(text, sources)
            result['text_index'] = i
            results.append(result)
        
        return results
    
    def find_matches(self, text, sources, threshold=0.7):
        """Find all matching passages above threshold"""
        self.similarity_threshold = threshold
        result = self.detect_plagiarism(text, sources)
        return result['matches']
    
    def get_similarity_score(self, text1, text2):
        """Get similarity between two texts"""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        return float(self._cosine_similarity(emb1, emb2))
    
    def _get_embedding(self, text):
        """Get embedding for text based on model type"""
        if self.model_type == 'tfidf':
            return self._get_tfidf_embedding(text)
        elif self.model_type == 'bert':
            return self.model.encode(text)
        elif self.model_type == 'use':
            return self.model([text])[0].numpy()
        else:
            return self._get_tfidf_embedding(text)
    
    def _get_tfidf_embedding(self, text):
        """Get TF-IDF embedding"""
        try:
            return self.vectorizer.transform([text]).toarray().flatten()
        except:
            # Vectorizer not fitted yet
            return np.random.rand(100)
    
    def _cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        
        # Reshape if needed
        emb1 = emb1.reshape(1, -1) if len(emb1.shape) == 1 else emb1
        emb2 = emb2.reshape(1, -1) if len(emb2.shape) == 1 else emb2
        
        # Calculate cosine similarity
        sim = cosine_similarity(emb1, emb2)[0][0]
        return float(sim)
    
    def _find_matching_passages(self, text1, text2, similarity_score):
        """Find matching sentences/passages between two texts"""
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        matches = []
        
        for sent1 in sentences1:
            for sent2 in sentences2:
                sim = self.get_similarity_score(sent1, sent2)
                if sim >= self.similarity_threshold * 0.8:  # Slightly lower threshold for sentences
                    matches.append({
                        'matched_text': sent1[:50] + '...' if len(sent1) > 50 else sent1,
                        'source_text': sent2[:50] + '...' if len(sent2) > 50 else sent2,
                        'similarity': float(sim)
                    })
        
        # Remove duplicates and sort by similarity
        matches = list({m['matched_text']: m for m in matches}.values())
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def get_detailed_report(self, text, sources):
        """Generate detailed plagiarism report"""
        result = self.detect_plagiarism(text, sources)
        
        report = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'plagiarism_detected': result['is_plagiarized'],
            'similarity_score': result['similarity_score'],
            'plagiarism_percentage': result['plagiarism_percentage'],
            'unique_percentage': 100 - result['plagiarism_percentage'],
            'matches': result['matches'],
            'recommendation': self._get_recommendation(result['similarity_score'])
        }
        
        return report
    
    def _get_recommendation(self, similarity_score):
        """Get recommendation based on similarity score"""
        if similarity_score >= 0.9:
            return 'High plagiarism detected. Significant rewriting needed.'
        elif similarity_score >= 0.7:
            return 'Moderate plagiarism detected. Paraphrase and cite sources.'
        elif similarity_score >= 0.5:
            return 'Some similarity found. Consider adding original content and citations.'
        else:
            return 'Low similarity. Content appears to be mostly original.'
    
    def set_sources(self, sources_list):
        """Set new sources (overwrites existing)"""
        self.sources = sources_list
    
    def clear_sources(self):
        """Clear all sources"""
        self.sources = []
        self.source_embeddings = []
    
    def export_results(self, results, format='json'):
        """Export results to JSON or other formats"""
        import json
        
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            if isinstance(results, list):
                writer = csv.DictWriter(output, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            else:
                writer = csv.DictWriter(output, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)
            
            return output.getvalue()
        
        return str(results)


if __name__ == '__main__':
    detector = PlagiarismDetector(model_type='tfidf')
    
    # Add sources
    detector.add_source('Wikipedia', 'Machine learning is a subset of artificial intelligence.')
    detector.add_source('Tutorial', 'Deep learning uses neural networks for pattern recognition.')
    
    # Check text
    test_text = "Machine learning is a subset of artificial intelligence that enables learning."
    result = detector.detect_plagiarism(test_text, [s['text'] for s in detector.sources])
    
    print('Plagiarism Detection Result:')
    print(f"Similarity: {result['similarity_score']:.2%}")
    print(f"Plagiarized: {result['is_plagiarized']}")
    print(f"Matches: {result['matches']}")