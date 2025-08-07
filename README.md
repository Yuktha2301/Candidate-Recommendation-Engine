# 🎯 Candidate Recommendation Engine

An intelligent web application that uses Google's Gemini AI to match job descriptions with candidate resumes, providing similarity scores and AI-generated analysis.

## 🚀 Live Demo

**Streamlit Cloud Deployment**: [Your App URL will be here after deployment]

## ✨ Features

- **AI-Powered Matching**: Uses Gemini 2.0 Flash for intelligent candidate-job matching
- **Multiple File Formats**: Supports PDF and DOCX resume uploads
- **Real-time Processing**: Instant similarity scoring and analysis
- **AI Summaries**: Automatic generation of candidate fit explanations
- **Top 10 Ranking**: Displays the most relevant candidates first
- **Rate Limit Handling**: Graceful error handling for API quotas with local time display

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.0 Flash
- **File Processing**: PyPDF2, docx2txt
- **Data Processing**: Pandas, NumPy
- **Environment**: Python 3.11 (optimized for Streamlit Cloud)

## 📋 Requirements

- Python 3.11 (recommended for Streamlit Cloud)
- Google API Key for Gemini AI
- Internet connection for AI model access

## 🚀 Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/A6h9lash/candidate-recommendation-engine.git
   cd candidate-recommendation-engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**:
   - Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Prepare your repository**:
   - Ensure all files are committed to GitHub
   - Make sure the repository is public (required for free Streamlit Cloud)

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Add environment variable: `GOOGLE_API_KEY=your_api_key`

3. **Access your app**:
   - Your app will be available at the provided URL

## 📖 Usage

1. **Enter Job Description**: Paste the job description in the sidebar
2. **Upload Resumes**: Upload PDF/DOCX files or add text resumes
3. **Process Candidates**: Click "Find Best Candidates" to analyze
4. **View Results**: See similarity scores and AI-generated summaries

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google AI API key (required)
- `DEBUG_MODE`: Set to 'true' for debug logging (optional)

### API Limits

- **Free Tier**: 50 requests per day
- **Rate Limit**: Quota resets at midnight UTC
- **Error Handling**: App gracefully handles rate limits with local time display

## 📁 Project Structure

```
candidate-recommendation-engine/
├── app.py                 # Main application
├── streamlit_app.py       # Streamlit Cloud entry point
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version specification
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🔑 API Key Setup

1. **Get Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

2. **Set Environment Variable**:
   - Local: Add to `.env` file
   - Streamlit Cloud: Add in deployment settings

## 🚨 Rate Limits

- **Daily Limit**: 50 requests (free tier)
- **Reset Time**: Midnight UTC
- **Error Handling**: App stops processing and shows retry time in local timezone
- **Local Time Display**: Shows when to retry in your timezone

## 🚀 Deployment Guide

### Pre-Deployment Checklist

- [ ] Repository is public (required for free Streamlit Cloud)
- [ ] All files are committed to GitHub
- [ ] Google API key is ready and working
- [ ] App runs locally without errors

### Deployment Steps

1. **Prepare GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/candidate-recommendation-engine.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Configure:
     - **Repository**: `A6h9lash/candidate-recommendation-engine`
     - **Branch**: `main`
     - **Main file path**: `streamlit_app.py`
     - **App URL**: `candidate-recommendation-engine` (or your preferred name)

3. **Add Environment Variables**:
   In the "Advanced settings" section:
   - **Key**: `GOOGLE_API_KEY`
   - **Value**: Your Google API key

4. **Deploy**:
   - Click "Deploy!"
   - Wait for build to complete (2-3 minutes)
   - Check logs for any errors

### Post-Deployment Verification

- [ ] App loads without errors
- [ ] Job description input works
- [ ] File uploads function
- [ ] AI processing works
- [ ] Results display correctly
- [ ] Rate limit handling works

## 🔧 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check `requirements.txt` has all dependencies
   - Verify Python version compatibility (Python 3.11 recommended)
   - Check logs for specific errors
   - **Python 3.13 Compatibility**: If you see `distutils` errors, ensure you're using Python 3.11

2. **File Watcher Errors**
   - If you see `inotify instance limit reached` errors, the app is configured to disable file watching
   - This is normal in cloud environments and doesn't affect functionality

2. **API Key Issues**
   - Verify API key is correct
   - Check API key has sufficient quota
   - Ensure key is properly set in environment variables

3. **File Upload Issues**
   - Check file size limits
   - Verify supported file formats
   - Test with different file types

4. **Rate Limit Issues**
   - Monitor API usage
   - Check quota reset times
   - Use sample data for testing

### Debug Commands
```bash
# Test local deployment
streamlit run streamlit_app.py

# Check requirements
pip install -r requirements.txt

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('GOOGLE_API_KEY')[:10] + '...')"
```

## 🎯 Assignment Requirements

This project fulfills the SproutsAI Machine Learning Engineer Internship assignment requirements:

### ✅ Core Requirements
- **Job Description Input**: Accept job descriptions via text input
- **Resume Upload**: Support for PDF and DOCX file uploads, plus text input
- **AI-Powered Matching**: Uses Google's Gemini 2.0 Flash for intelligent similarity scoring
- **Similarity Scoring**: Provides 0-100 similarity scores with detailed reasoning
- **Top Recommendations**: Displays top 5-10 most relevant candidates
- **Real-time Processing**: Live analysis with progress indicators

### ✅ Bonus Features
- **AI-Generated Summaries**: Detailed explanations of why each candidate is a great fit
- **Multiple File Formats**: Seamless handling of PDF, DOCX, and text inputs
- **Rate Limit Handling**: Graceful error handling with local time display
- **Professional UI**: Clean, intuitive interface with progress tracking

### 🛠️ Technical Implementation

#### 1. Gemini AI Integration
- Uses Google's Gemini 2.0 Flash model for intelligent text analysis
- Provides context-aware similarity scoring based on multiple factors
- Generates detailed explanations for each candidate match

#### 2. Similarity Computation
- Gemini AI analyzes job-candidate matches holistically
- Considers skills alignment, experience relevance, education, and technical expertise
- Scores range from 0-100 with detailed reasoning

#### 3. File Processing
- **PDF Files**: Uses PyPDF2 to extract text content
- **DOCX Files**: Uses docx2txt for text extraction
- **Text Input**: Direct text processing
- Handles multiple file formats seamlessly

#### 4. AI Summaries (Bonus Feature)
- Uses Gemini AI to generate personalized candidate analysis
- Explains why each candidate received their specific score
- Provides insights into strengths and areas of alignment

### 📊 Performance Considerations

- Gemini AI provides intelligent, context-aware analysis
- Handles multiple files efficiently with progress tracking
- Real-time processing with detailed feedback
- Responsive UI with progress indicators
- Graceful error handling for API rate limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🆘 Support

For issues or questions:
- Check the logs in Streamlit Cloud dashboard
- Verify your API key is correct
- Ensure you haven't exceeded daily quota
- Try using sample data for testing

---