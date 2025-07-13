import os
import json
import re
import requests
import praw
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Ensure the punkt_tab model is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # If punkt_tab is not found, use the standard punkt tokenizer
        nltk.download('punkt_tab', quiet=True)

# Download required NLTK data
download_nltk_data()

# Hugging Face API configuration
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
HF_API_KEY = os.getenv('HF_API_KEY')  # Your Hugging Face API key from .env

def query_hf_api(payload):
    """Send a request to the Hugging Face Inference API"""
    if not HF_API_KEY:
        raise ValueError("Hugging Face API key not found in .env file")
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for summarization"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs, special characters, and extra whitespace
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s.,!?\-]', ' ', text)    # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()        # Normalize whitespace
    
    # Ensure the text is not empty after cleaning
    if not text:
        return ""
    
    # Add period if missing at the end
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Summarize text using Hugging Face Inference API"""
    # Preprocess the text first
    text = preprocess_text(text)
    if not text:
        return "No valid text available for summarization"
    
    # Split into sentences and take first few if too long
    try:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 20:  # If too many sentences, take first 10 and last 5
            text = ' '.join(sentences[:10] + sentences[-5:])
    except Exception as e:
        print(f"Warning: Error in sentence tokenization: {str(e)}")
        # Fallback to simple space-based splitting if tokenization fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 20:
            text = '. '.join(sentences[:10] + sentences[-5:]) + '.'
    
    # Ensure text is not too long (BART has a max of 1024 tokens)
    text = text[:3000]  # Conservative limit to avoid token limit
    
    try:
        # Try with the newer API format first
        output = query_hf_api({
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length,
                "do_sample": False,
                "truncation": True
            }
        })
        
        # Handle different response formats
        if isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], dict):
                if 'summary_text' in output[0]:
                    return output[0]['summary_text']
                elif 'generated_text' in output[0]:
                    return output[0]['generated_text']
            return str(output[0])
        elif isinstance(output, dict):
            if 'summary_text' in output:
                return output['summary_text']
            elif 'generated_text' in output:
                return output['generated_text']
        
        # If we get here, try falling back to a simple heuristic
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 5:
                return ' '.join(sentences[:3] + ["..."] + sentences[-2:])
            return ' '.join(sentences)
        except Exception as e:
            print(f"Warning: Fallback tokenization failed: {str(e)}")
            # Last resort: return first 200 characters
            return text[:200] + ('...' if len(text) > 200 else '')
        
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        # Fallback to simple extraction of first few sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 3:
            return ' '.join(sentences[:3] + ["..."])
        return ' '.join(sentences) if sentences else "Summary not available"

class RedditScraper:
    def __init__(self):
        # Get Reddit API credentials from environment variables
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'UserPersonaGenerator/1.0')
        
        if not client_id or not client_secret:
            print("Warning: Reddit API credentials not found in .env file. Some functionality may be limited.")
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.stop_words = set(stopwords.words('english'))

    def get_user_content(self, username: str, limit: int = 100) -> Dict[str, List[Dict]]:
        """Fetch user's comments and posts."""
        try:
            redditor = self.reddit.redditor(username)
            
            # Get comments
            comments = []
            for comment in redditor.comments.new(limit=limit):
                comments.append({
                    'type': 'comment',
                    'subreddit': str(comment.subreddit),
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc
                })
            
            # Get posts
            posts = []
            for submission in redditor.submissions.new(limit=limit):
                posts.append({
                    'type': 'post',
                    'subreddit': str(submission.subreddit),
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'created_utc': submission.created_utc
                })
            
            return {
                'username': username,
                'comments': comments,
                'posts': posts
            }
        except Exception as e:
            print(f"Error fetching data for user {username}: {str(e)}")
            return None

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)]

    def analyze_content(self, content: Dict) -> Dict:
        """Analyze the content and generate a persona using Hugging Face API."""
        all_text = []
        sources = []
        
        # Process comments
        for i, comment in enumerate(content.get('comments', [])[:30]):  # Limit to 30 items
            text = f"{comment['body']}"
            all_text.append(text)
            sources.append({
                'type': 'comment',
                'subreddit': comment['subreddit'],
                'excerpt': text[:200] + '...' if len(text) > 200 else text
            })
        
        # Process posts
        for i, post in enumerate(content.get('posts', [])[:20]):  # Limit to 20 items
            text = f"{post['title']}. {post['selftext']}"
            all_text.append(text)
            sources.append({
                'type': 'post',
                'subreddit': post['subreddit'],
                'excerpt': text[:200] + '...' if len(text) > 200 else text
            })
        
        combined_text = ' '.join(all_text)
        
        # Extract keywords for interests
        interests = self.extract_keywords(combined_text, 15)
        
        # Generate summary using Hugging Face API
        analysis = summarize_text(combined_text)
        
        # Create basic persona
        persona = {
            'name': content['username'],
            'age_range': self.infer_age(combined_text),
            'location': self.infer_location(combined_text),
            'interests': interests[:10],  # Top 10 interests
            'personality_traits': self.analyze_personality(combined_text),
            'behaviors': self.identify_behaviors(all_text),
            'motivations': self.identify_motivations(all_text),
            'frustrations': self.identify_frustrations(all_text),
            'goals': self.identify_goals(all_text),
            'analysis': analysis,
            'sources': sources[:10]  # Limit to 10 sources
        }
        
        return persona
    
    def infer_age(self, text: str) -> str:
        """Infer age range from text."""
        age_indicators = {
            'teen': ['high school', 'college', 'university', 'school', 'parents', 'mom', 'dad'],
            '20s': ['college', 'university', 'grad school', 'first job', 'apartment'],
            '30s': ['career', 'married', 'spouse', 'husband', 'wife', 'mortgage', 'kids'],
            '40s+': ['kids', 'children', 'career', 'retirement', 'mortgage']
        }
        
        text_lower = text.lower()
        matches = {}
        for age_range, indicators in age_indicators.items():
            matches[age_range] = sum(indicator in text_lower for indicator in indicators)
        
        if not any(matches.values()):
            return 'Unknown'
            
        return max(matches.items(), key=lambda x: x[1])[0]
    
    def infer_location(self, text: str) -> str:
        """Infer location from text."""
        # This is a simple implementation - could be enhanced with NER
        location_keywords = [
            'europe', 'asia', 'africa', 'north america', 'south america', 'australia',
            'uk', 'united kingdom', 'canada', 'australia', 'india', 'germany', 'france'
        ]
        
        text_lower = text.lower()
        for loc in location_keywords:
            if loc in text_lower:
                return loc.title()
        return 'Unknown'
    
    def analyze_personality(self, text: str) -> List[str]:
        """Analyze personality traits from text."""
        traits = []
        text_lower = text.lower()
        
        # Simple keyword matching for personality traits
        if any(word in text_lower for word in ['i think', 'i believe', 'in my opinion']):
            traits.append('Analytical')
        if any(word in text_lower for word in ['help', 'support', 'advice']):
            traits.append('Helpful')
        if any(word in text_lower for word in ['lol', 'haha', 'funny', 'joke']):
            traits.append('Humorous')
        if any(word in text_lower for word in ['sad', 'angry', 'frustrated']):
            traits.append('Passionate')
        if any(word in text_lower for word in ['thanks', 'appreciate', 'grateful']):
            traits.append('Appreciative')
            
        return traits[:5] if traits else ['Neutral']
    
    def identify_behaviors(self, texts: List[str]) -> List[str]:
        """Identify user behaviors from text."""
        behaviors = []
        joined_text = ' '.join(texts).lower()
        
        if any(word in joined_text for word in ['i always', 'usually', 'typically']):
            behaviors.append('Habitual in their approach')
        if any(word in joined_text for word in ['i think', 'in my opinion', 'i believe']):
            behaviors.append('Opinionated and expressive')
        if any(word in joined_text for word in ['question', 'why', 'how', 'what if']):
            behaviors.append('Inquisitive and curious')
        if any(word in joined_text for word in ['i feel', 'i think', 'i believe']):
            behaviors.append('Reflective and self-aware')
            
        return behaviors if behaviors else ['Engages in discussions']
    
    def identify_motivations(self, texts: List[str]) -> List[str]:
        """Identify user motivations from text."""
        motivations = []
        joined_text = ' '.join(texts).lower()
        
        if any(word in joined_text for word in ['learn', 'understand', 'know']):
            motivations.append('Desire for knowledge and learning')
        if any(word in joined_text for word in ['help', 'support', 'advice']):
            motivations.append('Helping others')
        if any(word in joined_text for word in ['share', 'experience', 'story']):
            motivations.append('Sharing experiences')
        if any(word in joined_text for word in ['discuss', 'debate', 'opinion']):
            motivations.append('Engaging in discussions')
            
        return motivations if motivations else ['Participating in community discussions']
    
    def identify_frustrations(self, texts: List[str]) -> List[str]:
        """Identify user frustrations from text."""
        frustrations = []
        joined_text = ' '.join(texts).lower()
        
        if any(word in joined_text for word in ['problem', 'issue', 'trouble']):
            frustrations.append('Technical or situational problems')
        if any(word in joined_text for word in ['confused', 'dont understand', 'not clear']):
            frustrations.append('Lack of clarity or information')
        if any(word in joined_text for word in ['annoying', 'frustrating', 'upset']):
            frustrations.append('General frustrations with situations')
            
        return frustrations if frustrations else ['No specific frustrations identified']
    
    def identify_goals(self, texts: List[str]) -> List[str]:
        """Identify user goals from text."""
        goals = []
        joined_text = ' '.join(texts).lower()
        
        if any(word in joined_text for word in ['learn', 'understand', 'know']):
            goals.append('Gain knowledge')
        if any(word in joined_text for word in ['help', 'support']):
            goals.append('Help others')
        if any(word in joined_text for word in ['share', 'contribute']):
            goals.append('Share experiences')
            
        return goals if goals else ['Engage with community']

    def save_persona(self, persona: Dict, username: str):
        """Save the persona to a text file."""
        filename = f"{username}_persona.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"User Persona: {persona.get('name', username)}\n")
            f.write("="*50 + "\n\n")
            
            # Basic Info
            f.write("DEMOGRAPHICS\n")
            f.write("-"*50 + "\n")
            f.write(f"Name: {persona.get('name', 'N/A')}\n")
            f.write(f"Age Range: {persona.get('age_range', 'N/A')}\n")
            f.write(f"Location: {persona.get('location', 'N/A')}\n")
            f.write(f"Occupation: {persona.get('occupation', 'N/A')}\n\n")
            
            # Interests
            f.write("INTERESTS\n")
            f.write("-"*50 + "\n")
            for interest in persona.get('interests', []):
                f.write(f"- {interest}\n")
            f.write("\n")
            
            # Personality Traits
            f.write("PERSONALITY TRAITS\n")
            f.write("-"*50 + "\n")
            for trait in persona.get('personality_traits', []):
                f.write(f"- {trait}\n")
            f.write("\n")
            
            # Behaviors
            f.write("BEHAVIORS\n")
            f.write("-"*50 + "\n")
            for behavior in persona.get('behaviors', []):
                f.write(f"- {behavior}\n")
            f.write("\n")
            
            # Motivations
            f.write("MOTIVATIONS\n")
            f.write("-"*50 + "\n")
            for motivation in persona.get('motivations', []):
                f.write(f"- {motivation}\n")
            f.write("\n")
            
            # Frustrations
            f.write("FRUSTRATIONS\n")
            f.write("-"*50 + "\n")
            for frustration in persona.get('frustrations', []):
                f.write(f"- {frustration}\n")
            f.write("\n")
            
            # Goals
            f.write("GOALS\n")
            f.write("-"*50 + "\n")
            for goal in persona.get('goals', []):
                f.write(f"- {goal}\n")
            f.write("\n")
            
            # Analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write(f"{persona.get('analysis', 'No analysis available.')}\n\n")
            
            # Sources
            f.write("SOURCES\n")
            f.write("-"*50 + "\n")
            for i, source in enumerate(persona.get('sources', [])[:10], 1):  # Limit to 10 sources
                f.write(f"{i}. [{source.get('type', 'post').upper()}] r/{source.get('subreddit', 'unknown')}\n")
                f.write(f"   Excerpt: {source.get('excerpt', 'No excerpt')}\n\n")
            
        print(f"Persona saved to {filename}")

def main():
    # Initialize the scraper
    scraper = RedditScraper()
    
    # Get Reddit username from user
    username = input("Enter Reddit username (without u/): ").strip()
    
    # Fetch user content
    print(f"Fetching content for user {username}...")
    content = scraper.get_user_content(username)
    
    if not content:
        print("Failed to fetch user content. The username might not exist or the account might be private.")
        return
    
    # Analyze content and generate persona
    print("Analyzing content and generating persona (this may take a few minutes)...")
    try:
        persona = scraper.analyze_content(content)
        
        if not persona:
            print("Failed to generate persona.")
            return
        
        # Save persona to file
        scraper.save_persona(persona, username)
        print("Persona generation complete!")
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        print("This might be due to the model size. Please ensure you have enough memory and try again.")

if __name__ == "__main__":
    main()
