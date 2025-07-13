# Reddit User Persona Generator

This script generates a detailed user persona by analyzing a Reddit user's comments and posts using local natural language processing models.

## Prerequisites

1. Python 3.8 or higher
2. Reddit API credentials (optional, for private subreddits)

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your API credentials:
   ```
   # Reddit API Credentials
   REDDIT_CLIENT_ID=reddit_client_id
   REDDIT_CLIENT_SECRET=reddit_client_secret
   REDDIT_USER_AGENT=AppName/1.0 by YourUsername
   
   # Hugging Face API Key
   HF_API_KEY=huggingface_api_key
   ```

   - **Reddit API Credentials**:
     1. Go to https://www.reddit.com/prefs/apps
     2. Click "Create App" or "Create Another App" at the bottom
     3. Fill in the form (select "script" for the type)
     4. Use the client ID (under the app name) and client secret
   
   - **Hugging Face API Key**:
     1. Go to https://huggingface.co/settings/tokens
     2. Create a new access token
     3. Copy the token and add it to your `.env` file as `HF_API_KEY`

## Usage

Run the script with:
```
python reddit_persona_generator.py
```

When prompted, enter the Reddit username (without the u/ prefix) that you want to analyze.

## Output

The script will generate a text file named `{username}_persona.txt` containing:
- Demographics (age range, location)
- Interests (based on frequent terms)
- Personality traits
- Behaviors
- Motivations
- Frustrations
- Goals
- Summary analysis
- Sources (with citations from the user's posts/comments)

## How It Works

The script uses several techniques to analyze user content:
1. **Keyword Extraction**: Identifies most frequently used terms (excluding common words)
2. **Text Summarization**: Uses BART model to generate summaries of user content
3. **Pattern Matching**: Identifies behaviors, motivations, and personality traits through keyword patterns
4. **Statistical Analysis**: Uses word frequency and context to infer user characteristics

## Notes

- The first run will download the required NLP models (about 1.5GB total)
- Analysis may take several minutes depending on the amount of content
- The script works best with active Reddit users who have posted substantial content
- For private accounts or users with no activity, the script will return limited information
- The quality of the persona depends on the amount and type of content the user has posted

## Performance Considerations

- The script uses local models which require significant memory
- If you encounter memory issues, try reducing the number of posts/comments analyzed in the code
- The summarization is limited to the first 3000 characters of content to prevent excessive memory usage
