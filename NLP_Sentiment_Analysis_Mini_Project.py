"""
NLP SENTIMENT ANALYSIS MINI PROJECT
Experiment 8 - Natural Language Processing
Author: Student Name
Date: 2024

This project performs sentiment analysis on text input using lexicon-based NLP.
Features:
- Text preprocessing (tokenization, stopword removal)
- Sentiment classification (Positive/Negative/Neutral)
- Emotion detection (Joy, Sadness, Anger, Fear, Surprise)
- Confidence scoring
- Web interface using Flask
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import re
from datetime import datetime
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ==================== NLP LEXICONS ====================

POSITIVE_WORDS = [
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
    'love', 'loved', 'best', 'awesome', 'perfect', 'beautiful', 'happy', 
    'excited', 'joy', 'brilliant', 'outstanding', 'superb', 'pleased', 
    'delighted', 'satisfied', 'impressive', 'remarkable', 'incredible', 
    'exceptional', 'magnificent', 'terrific', 'fabulous', 'marvelous', 
    'splendid', 'enjoy', 'enjoyed', 'positive', 'recommendation'
]

NEGATIVE_WORDS = [
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated', 
    'poor', 'disappointing', 'disappointed', 'sad', 'angry', 'frustrated', 
    'annoyed', 'upset', 'pathetic', 'useless', 'waste', 'disgusting', 
    'appalling', 'dreadful', 'inferior', 'substandard', 'defective', 
    'faulty', 'broken', 'damaged', 'failed', 'failure', 'regret', 'unhappy',
    'negative', 'dissatisfied'
]

EMOTION_LEXICON = {
    'joy': ['happy', 'joy', 'excited', 'delighted', 'cheerful', 'pleased', 
            'glad', 'enjoy', 'wonderful', 'fantastic', 'amazing'],
    'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 
                'gloomy', 'sorrow', 'grief', 'heartbroken'],
    'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 
              'hate', 'rage', 'outraged', 'hostile'],
    'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified', 
             'fearful', 'panic', 'dread'],
    'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'unexpected', 
                 'stunning', 'wow', 'incredible']
}

STOPWORDS = [
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 
    'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 
    'what', 'when', 'where', 'who', 'why', 'how', 'all', 'each', 'every', 
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'to', 'of', 
    'in', 'for', 'with', 'by', 'from'
]

# ==================== TEXT PREPROCESSING ====================

def preprocess_text(text):
    """
    Preprocess input text for sentiment analysis
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters
    3. Tokenize
    4. Remove stopwords
    5. Filter short tokens
    
    Args:
        text (str): Raw input text
        
    Returns:
        dict: Preprocessed text data including tokens
    """
    # Store original
    original_text = text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = text.split()
    
    # Count original words
    word_count = len(tokens)
    
    # Remove stopwords and short tokens
    filtered_tokens = [
        token for token in tokens 
        if token not in STOPWORDS and len(token) > 2
    ]
    
    return {
        'original': original_text,
        'cleaned': text,
        'tokens': filtered_tokens,
        'word_count': word_count,
        'token_count': len(filtered_tokens)
    }

# ==================== SENTIMENT ANALYSIS ====================

def analyze_sentiment(tokens):
    """
    Perform sentiment analysis on preprocessed tokens
    
    Args:
        tokens (list): List of preprocessed tokens
        
    Returns:
        dict: Sentiment analysis results
    """
    # Initialize scores
    positive_score = 0
    negative_score = 0
    detected_emotions = {}
    
    # Analyze each token
    for token in tokens:
        # Check positive words
        if token in POSITIVE_WORDS:
            positive_score += 1
            
        # Check negative words
        if token in NEGATIVE_WORDS:
            negative_score += 1
            
        # Check emotions
        for emotion, emotion_words in EMOTION_LEXICON.items():
            if token in emotion_words:
                detected_emotions[emotion] = detected_emotions.get(emotion, 0) + 1
    
    # Calculate sentiment
    total_score = positive_score + negative_score
    
    if total_score == 0:
        sentiment = 'neutral'
        confidence = 50.0
    else:
        positive_ratio = positive_score / total_score
        
        if positive_ratio > 0.6:
            sentiment = 'positive'
            confidence = min(60 + (positive_ratio - 0.6) * 150, 98)
        elif positive_ratio < 0.4:
            sentiment = 'negative'
            confidence = min(60 + (0.4 - positive_ratio) * 150, 98)
        else:
            sentiment = 'neutral'
            confidence = 50 + abs(positive_ratio - 0.5) * 100
    
    # Calculate emotion percentages
    emotion_scores = {}
    total_emotions = sum(detected_emotions.values())
    
    if total_emotions > 0:
        for emotion, count in detected_emotions.items():
            emotion_scores[emotion] = min((count / total_emotions) * 100, 95)
    
    # Extract keywords
    positive_keywords = [token for token in tokens if token in POSITIVE_WORDS]
    negative_keywords = [token for token in tokens if token in NEGATIVE_WORDS]
    
    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 2),
        'positive_score': positive_score,
        'negative_score': negative_score,
        'emotions': emotion_scores,
        'keywords': {
            'positive': positive_keywords,
            'negative': negative_keywords
        }
    }

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Serve the main HTML interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for sentiment analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        preprocessed = preprocess_text(text)
        
        # Analyze sentiment
        analysis = analyze_sentiment(preprocessed['tokens'])
        
        # Create result
        result = {
            'id': int(datetime.now().timestamp() * 1000),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text': text,
            'preprocessed': preprocessed,
            'analysis': analysis,
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ==================== HTML TEMPLATE ====================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Sentiment Analysis - Experiment 8</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes spin { to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 1s linear infinite; }
        .line-clamp-2 { 
            display: -webkit-box; 
            -webkit-line-clamp: 2; 
            -webkit-box-orient: vertical; 
            overflow: hidden; 
        }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-3">
                🌟 NLP Sentiment Analysis System
            </h1>
            <p class="text-gray-600 text-lg">Experiment 8 - Natural Language Processing Mini Project</p>
        </div>

        <!-- Statistics Dashboard -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="bg-white rounded-xl shadow-md p-4 border-l-4 border-blue-500">
                <p class="text-gray-500 text-sm">Total Analyzed</p>
                <p class="text-2xl font-bold text-gray-800" id="totalCount">0</p>
            </div>
            <div class="bg-white rounded-xl shadow-md p-4 border-l-4 border-green-500">
                <p class="text-gray-500 text-sm">Positive</p>
                <p class="text-2xl font-bold text-green-600" id="positiveCount">0</p>
            </div>
            <div class="bg-white rounded-xl shadow-md p-4 border-l-4 border-red-500">
                <p class="text-gray-500 text-sm">Negative</p>
                <p class="text-2xl font-bold text-red-600" id="negativeCount">0</p>
            </div>
            <div class="bg-white rounded-xl shadow-md p-4 border-l-4 border-gray-500">
                <p class="text-gray-500 text-sm">Neutral</p>
                <p class="text-2xl font-bold text-gray-600" id="neutralCount">0</p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Input Section -->
            <div class="lg:col-span-2">
                <div class="bg-white rounded-xl shadow-lg p-6 mb-4">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">📄 Text Input</h2>
                    <textarea 
                        id="textInput" 
                        placeholder="Enter text to analyze sentiment... (Try: 'I love this product! It's amazing!')" 
                        class="w-full h-32 p-4 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none resize-none"
                    ></textarea>
                    
                    <div class="flex items-center gap-2 mt-4 flex-wrap">
                        <button id="analyzeBtn" class="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-2 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all">
                            ✨ Analyze
                        </button>
                        <button id="clearBtn" class="flex items-center gap-2 bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition-all">
                            🗑️ Clear
                        </button>
                        <button id="exportBtn" class="flex items-center gap-2 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all ml-auto hidden">
                            💾 Export Results
                        </button>
                    </div>

                    <!-- Sample Texts -->
                    <div class="mt-6">
                        <p class="text-sm text-gray-600 mb-2 font-medium">Quick Samples:</p>
                        <div class="flex flex-wrap gap-2" id="samplesContainer"></div>
                    </div>
                </div>

                <!-- Results Section -->
                <div id="resultsSection" class="bg-white rounded-xl shadow-lg p-6 hidden">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">📊 Analysis Results</h2>
                    <div id="resultsContent"></div>
                </div>
            </div>

            <!-- History Section -->
            <div class="lg:col-span-1">
                <div class="bg-white rounded-xl shadow-lg p-6 sticky top-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">📈 Analysis History</h2>
                    <div id="historyContainer" class="space-y-3 max-h-[600px] overflow-y-auto">
                        <div class="text-center py-8 text-gray-400">
                            <p>⚠️ No analysis yet</p>
                            <p class="text-xs mt-2">Enter text and click Analyze</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="mt-8 text-center text-sm text-gray-500">
            <p>Built with Python Flask • Lexicon-based NLP • Experiment 8</p>
            <p class="mt-1">Features: Text Preprocessing • Sentiment Classification • Emotion Detection</p>
        </div>
    </div>

    <script>
        // Sample texts
        const sampleTexts = [
            "I absolutely love this product! It exceeded all my expectations and the customer service was amazing!",
            "This is the worst experience I've ever had. Completely disappointed and frustrated.",
            "The product is okay. Nothing special but it works as described.",
            "I'm so happy and excited about this purchase! Best decision ever! 😊",
            "Terrible quality, waste of money. Very unhappy with this purchase."
        ];

        // Statistics
        let stats = { totalAnalyzed: 0, positive: 0, negative: 0, neutral: 0 };
        let history = [];

        // Emotion emojis
        const emotionEmojis = { 
            joy: '😊', 
            sadness: '😢', 
            anger: '😠', 
            fear: '😨', 
            surprise: '😮' 
        };

        // Display results
        function displayResults(result) {
            const section = document.getElementById('resultsSection');
            const content = document.getElementById('resultsContent');
            
            const sentimentColors = {
                positive: 'text-green-600 bg-green-50 border-green-200',
                negative: 'text-red-600 bg-red-50 border-red-200',
                neutral: 'text-gray-600 bg-gray-50 border-gray-200'
            };
            
            const sentimentIcons = {
                positive: '😊',
                negative: '😢',
                neutral: '😐'
            };

            let html = `
                <div class="border-2 rounded-lg p-4 mb-4 ${sentimentColors[result.analysis.sentiment]}">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <span class="text-3xl">${sentimentIcons[result.analysis.sentiment]}</span>
                            <div>
                                <p class="text-sm font-medium uppercase tracking-wide">Sentiment</p>
                                <p class="text-2xl font-bold capitalize">${result.analysis.sentiment}</p>
                            </div>
                        </div>
                        <div class="text-right">
                            <p class="text-sm font-medium">Confidence</p>
                            <p class="text-2xl font-bold">${result.analysis.confidence.toFixed(1)}%</p>
                        </div>
                    </div>
                    <div class="w-full bg-white bg-opacity-50 rounded-full h-2">
                        <div class="h-2 rounded-full ${result.analysis.sentiment === 'positive' ? 'bg-green-600' : result.analysis.sentiment === 'negative' ? 'bg-red-600' : 'bg-gray-600'}" 
                             style="width: ${result.analysis.confidence}%"></div>
                    </div>
                </div>

                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                    <h3 class="font-semibold text-blue-900 mb-2">📝 Preprocessing Statistics</h3>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div><span class="text-blue-700">Original Words:</span><span class="font-bold ml-2">${result.preprocessed.word_count}</span></div>
                        <div><span class="text-blue-700">Processed Tokens:</span><span class="font-bold ml-2">${result.preprocessed.token_count}</span></div>
                        <div><span class="text-blue-700">Positive Signals:</span><span class="font-bold ml-2 text-green-600">${result.analysis.positive_score}</span></div>
                        <div><span class="text-blue-700">Negative Signals:</span><span class="font-bold ml-2 text-red-600">${result.analysis.negative_score}</span></div>
                    </div>
                </div>
            `;

            // Emotions
            if (Object.keys(result.analysis.emotions).length > 0) {
                html += `
                    <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
                        <h3 class="font-semibold text-purple-900 mb-3">🎭 Emotions Detected</h3>
                        <div class="space-y-2">
                `;
                
                Object.entries(result.analysis.emotions).forEach(([emotion, score]) => {
                    html += `
                        <div class="flex items-center gap-2">
                            <span class="text-2xl">${emotionEmojis[emotion]}</span>
                            <div class="flex-1">
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-medium capitalize text-purple-900">${emotion}</span>
                                    <span class="text-sm font-bold text-purple-700">${score.toFixed(0)}%</span>
                                </div>
                                <div class="w-full bg-purple-200 rounded-full h-2">
                                    <div class="bg-purple-600 h-2 rounded-full" style="width: ${score}%"></div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += `</div></div>`;
            }

            // Keywords
            if (result.analysis.keywords.positive.length > 0 || result.analysis.keywords.negative.length > 0) {
                html += `<div class="grid grid-cols-2 gap-4">`;
                
                if (result.analysis.keywords.positive.length > 0) {
                    html += `
                        <div class="bg-green-50 border border-green-200 rounded-lg p-3">
                            <h3 class="font-semibold text-green-900 mb-2 text-sm">✅ Positive Keywords</h3>
                            <div class="flex flex-wrap gap-1">
                                ${result.analysis.keywords.positive.map(word => 
                                    `<span class="text-xs bg-green-200 text-green-800 px-2 py-1 rounded">${word}</span>`
                                ).join('')}
                            </div>
                        </div>
                    `;
                }
                
                if (result.analysis.keywords.negative.length > 0) {
                    html += `
                        <div class="bg-red-50 border border-red-200 rounded-lg p-3">
                            <h3 class="font-semibold text-red-900 mb-2 text-sm">❌ Negative Keywords</h3>
                            <div class="flex flex-wrap gap-1">
                                ${result.analysis.keywords.negative.map(word => 
                                    `<span class="text-xs bg-red-200 text-red-800 px-2 py-1 rounded">${word}</span>`
                                ).join('')}
                            </div>
                        </div>
                    `;
                }
                
                html += `</div>`;
            }

            content.innerHTML = html;
            section.classList.remove('hidden');
        }

        // Update history
        function updateHistory() {
            const container = document.getElementById('historyContainer');
            
            if (history.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-8 text-gray-400">
                        <p>⚠️ No analysis yet</p>
                        <p class="text-xs mt-2">Enter text and click Analyze</p>
                    </div>
                `;
                return;
            }

            const sentimentColors = {
                positive: 'border-green-300 bg-green-50',
                negative: 'border-red-300 bg-red-50',
                neutral: 'border-gray-300 bg-gray-50'
            };

            container.innerHTML = history.map((item, index) => `
                <div class="border-2 ${sentimentColors[item.analysis.sentiment]} rounded-lg p-3 hover:shadow-md transition-all cursor-pointer" 
                     onclick="showHistoryItem(${index})">
                    <div class="flex items-start justify-between mb-2">
                        <div class="px-2 py-1 rounded text-xs font-bold ${
                            item.analysis.sentiment === 'positive' ? 'bg-green-600 text-white' : 
                            item.analysis.sentiment === 'negative' ? 'bg-red-600 text-white' : 
                            'bg-gray-600 text-white'
                        }">
                            ${item.analysis.sentiment.toUpperCase()}
                        </div>
                        <span class="text-xs text-gray-500">${item.timestamp}</span>
                    </div>
                    <p class="text-sm text-gray-700 line-clamp-2">${item.text}</p>
                    <div class="flex items-center gap-2 mt-2">
                        <div class="text-xs text-gray-500">${item.analysis.confidence.toFixed(0)}% confident</div>
                        ${Object.keys(item.analysis.emotions).length > 0 ? 
                            `<div class="text-xs flex gap-1">
                                ${Object.keys(item.analysis.emotions).slice(0, 2).map(e => emotionEmojis[e]).join('')}
                            </div>` : ''
                        }
                    </div>
                </div>
            `).join('');

            document.getElementById('exportBtn').classList.remove('hidden');
        }

        // Update statistics
        function updateStats() {
            document.getElementById('totalCount').textContent = stats.totalAnalyzed;
            document.getElementById('positiveCount').textContent = stats.positive;
            document.getElementById('negativeCount').textContent = stats.negative;
            document.getElementById('neutralCount').textContent = stats.neutral;
        }

        // Show history item
        function showHistoryItem(index) {
            displayResults(history[index]);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Analyze text
        async function analyzeText() {
            const text = document.getElementById('textInput').value.trim();
            
            if (!text) {
                alert('Please enter some text to analyze!');
                return;
            }

            const btn = document.getElementById('analyzeBtn');
            btn.innerHTML = '<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div> Analyzing...';
            btn.disabled = true;

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    displayResults(result);
                    history.unshift(result);
                    if (history.length > 10) history.pop();
                    
                    stats.totalAnalyzed++;
                    stats[result.analysis.sentiment]++;
                    
                    updateStats();
                    updateHistory();
                } else {
                    alert('Error: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error analyzing text: ' + error.message);
            } finally {
                btn.innerHTML = '✨ Analyze';
                btn.disabled = false;
            }
        }

        // Export results
        function exportResults() {
            const data = {
                statistics: stats,
                history: history,
                exportDate: new Date().toLocaleString()
            };

            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sentiment_analysis_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Add sample buttons
            const samplesContainer = document.getElementById('samplesContainer');
            sampleTexts.forEach((sample, idx) => {
                const btn = document.createElement('button');
                btn.className = 'text-xs bg-purple-100 text-purple-700 px-3 py-1 rounded-full hover:bg-purple-200 transition-all';
                btn.textContent = `Sample ${idx + 1}`;
                btn.onclick = () => { 
                    document.getElementById('textInput').value = sample; 
                };
                samplesContainer.appendChild(btn);
            });

            // Event listeners
            document.getElementById('analyzeBtn').onclick = analyzeText;
            document.getElementById('clearBtn').onclick = () => { 
                document.getElementById('textInput').value = ''; 
            };
            document.getElementById('exportBtn').onclick = exportResults;
            
            // Enter key to analyze
            document.getElementById('textInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    analyzeText();
                }
            });
        });
    </script>
</body>
</html>
'''

# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    print('='*60)
    print('NLP SENTIMENT ANALYSIS SYSTEM')
    print('Experiment 8 - Natural Language Processing Mini Project')
    print('='*60)
    print()
    print('🚀 Starting Flask server...')
    print('📊 Features:')
    print('   ✓ Text Preprocessing (Tokenization, Stopword Removal)')
    print('   ✓ Sentiment Classification (Positive/Negative/Neutral)')
    print('   ✓ Emotion Detection (Joy, Sadness, Anger, Fear, Surprise)')
    print('   ✓ Confidence Scoring')
    print('   ✓ Real-time Analysis Dashboard')
    print('   ✓ Analysis History & Export')
    print()
    print('🌐 Server running at: http://localhost:5000')
    print('💡 Press Ctrl+C to stop the server')
    print('='*60)
    print()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)