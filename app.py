import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import PyPDF2
import docx2txt
import io
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import time
import json

# Load environment variables
load_dotenv()

# Enable debug mode for troubleshooting
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Runtime configuration for production deployment
# Note: server.fileWatcherType should be set via command line or config.toml if needed

# Configure page
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'candidates' not in st.session_state:
    st.session_state['candidates'] = []
if 'job_description' not in st.session_state:
    st.session_state['job_description'] = ""
if 'gemini_model' not in st.session_state:
    st.session_state['gemini_model'] = None

def setup_gemini():
    """Setup Google Generative AI with Gemini model."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    return model

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Extract text content from a DOCX file."""
    try:
        text = docx2txt.process(docx_file)
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return ""

def get_similarity_score_with_gemini(job_description: str, resume_text: str, model) -> float:
    """Get similarity score using Gemini AI with improved consistency."""
    try:
        # Clean and normalize the inputs
        job_desc_clean = job_description.strip()
        resume_clean = resume_text.strip()
        
        # Create a more structured and consistent prompt
        prompt = f"""You are an expert HR recruiter evaluating a candidate for a data science position.

JOB DESCRIPTION:
{job_desc_clean}

CANDIDATE RESUME:
{resume_clean}

TASK: Rate the candidate's fit for this position on a scale of 0-100.

SCORING CRITERIA:
- 0-20: Poor match (major gaps in requirements)
- 21-40: Below average match (some relevant experience but significant gaps)
- 41-60: Average match (meets basic requirements with some gaps)
- 61-80: Good match (meets most requirements well)
- 81-100: Excellent match (exceeds requirements or perfect fit)

EVALUATION FACTORS:
1. Technical skills alignment (Python, R, SQL, analytics tools)
2. Experience level and relevance (years of experience in data science/analytics)
3. Education requirements (quantitative degree, advanced degrees)
4. Industry experience (data science, analytics, business problems)
5. Project scope and complexity (leadership, strategic impact)

RESPONSE FORMAT: Provide ONLY a number between 0 and 100, no other text or explanation."""

        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
        # Improved score extraction with better error handling
        try:
            # First try direct float conversion
            score = float(score_text)
        except ValueError:
            # If that fails, try to extract numbers
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                # Take the first number that could be a valid score
                for num in numbers:
                    potential_score = float(num)
                    if 0 <= potential_score <= 100:
                        score = potential_score
                        break
                else:
                    # If no valid score found, use default
                    score = 50.0
            else:
                score = 50.0
        
        # Ensure score is within valid range
        score = max(0, min(100, score))
        
        # Log the result for debugging (only in development)
        if DEBUG_MODE:
            print(f"Debug - Raw response: '{score_text}' -> Score: {score}%")
        
        return score
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg and "quota" in error_msg.lower():
            # Calculate local time equivalent to midnight UTC
            import datetime
            import pytz
            
            utc_now = datetime.datetime.now(pytz.UTC)
            utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
            
            # Convert to local time
            local_tz = datetime.datetime.now().astimezone().tzinfo
            local_midnight = utc_midnight.astimezone(local_tz)
            
            st.error(f"⚠️ API Rate Limit Exceeded: You've reached the daily limit for Gemini API requests (50/day for free tier).")
            st.error(f"🕛 Please retry after {local_midnight.strftime('%I:%M %p')} {local_midnight.strftime('%Z')} ({local_midnight.strftime('%B %d, %Y')})")
            st.info("💡 Tip: You can still use the app with sample data or wait for the quota to reset.")
            raise Exception("API_RATE_LIMIT_EXCEEDED")
        elif "quota" in error_msg.lower():
            st.error("⚠️ API Quota Exceeded: Please check your billing and plan details.")
            raise Exception("API_QUOTA_EXCEEDED")
        else:
            st.error(f"Error getting similarity score: {error_msg}")
            raise Exception(f"API_ERROR: {error_msg}")

def generate_ai_summary_with_gemini(job_description: str, resume_text: str, similarity_score: float, model) -> str:
    """Generate AI summary explaining the match using Gemini."""
    try:
        prompt = f"""
        You are an expert HR recruiter. Analyze why this candidate received a similarity score of {similarity_score:.1f}% for this role.
        
        Job Description:
        {job_description[:1000]}
        
        Candidate Resume:
        {resume_text[:1000]}
        
        Similarity Score: {similarity_score:.1f}%
        
        Please provide a concise summary (2-3 sentences) explaining:
        1. Why this candidate received this specific score
        2. Key strengths that align with the job requirements
        3. Areas where the candidate matches or differs from the role expectations
        
        Keep the response professional and constructive.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg and "quota" in error_msg.lower():
            # Calculate local time equivalent to midnight UTC
            import datetime
            import pytz
            
            utc_now = datetime.datetime.now(pytz.UTC)
            utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
            
            # Convert to local time
            local_tz = datetime.datetime.now().astimezone().tzinfo
            local_midnight = utc_midnight.astimezone(local_tz)
            
            st.error(f"⚠️ API Rate Limit Exceeded: Unable to generate AI summary due to daily quota limit.")
            st.error(f"🕛 Please retry after {local_midnight.strftime('%I:%M %p')} {local_midnight.strftime('%Z')} ({local_midnight.strftime('%B %d, %Y')})")
            raise Exception("API_RATE_LIMIT_EXCEEDED")
        elif "quota" in error_msg.lower():
            st.error("⚠️ API Quota Exceeded: Unable to generate AI summary. Please check your billing details.")
            raise Exception("API_QUOTA_EXCEEDED")
        else:
            st.error(f"Unable to generate AI summary: {error_msg}")
            raise Exception(f"API_ERROR: {error_msg}")

def main():
    # Custom CSS to remove padding and make full width
    st.markdown("""
    <style>
    /* Remove all top padding and margins */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0 !important;
        margin-top: 0 !important;
        max-width: none;
    }
    .stColumn {
        padding: 0 0.5rem;
    }
    /* Remove default top margin from title */
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Remove top margin from the entire app */
    .stApp > header {
        display: none;
    }
    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load the Gemini model
    if 'gemini_model' not in st.session_state or st.session_state.get('gemini_model') is None:
        with st.spinner("Loading Gemini AI model..."):
            st.session_state['gemini_model'] = setup_gemini()
    
    if st.session_state.get('gemini_model') is None:
        st.error("Failed to load Gemini model. Please check your API key.")
        return
    
    # Title at the very top
    st.title("Candidate Recommendation Engine")
    st.markdown("**Developed by Yuktha Bhargavi** | [LinkedIn](https://www.linkedin.com/in/yuktha-bhargavi-b8910a1b4/) | [Github](https://github.com/Yuktha2301)")
    
    # Horizontal info sections below the title
    col_info1, col_info2, col_info3, col_stats = st.columns([3, 3, 3, 2])
    
    with col_info1:
        st.markdown("**Instructions**")
        st.markdown("• Enter Job Description below<br>• Upload Resume Files (PDF/DOCX) or add text<br>• Click 'Find Best Candidates' to process<br>• View Results with similarity scores and AI analysis<br>• Only 25 resumes can be analysed in a day (Gemini API free tier restrictions)", unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("**Features**")
        st.markdown("• Gemini AI-powered similarity matching<br>• Multiple file format support<br>• Automatic AI-generated summaries<br>• Real-time processing", unsafe_allow_html=True)
    
    with col_info3:
        st.markdown("**Setup**")
        st.markdown("**Required:** Google API Key<br>• Get from [Google AI Studio](https://makersuite.google.com/app/apikey)<br>• Set as environment variable: `GOOGLE_API_KEY`", unsafe_allow_html=True)
    
    with col_stats:
        if st.session_state.get('candidates'):
            st.markdown("**Statistics**")
            scores = [r['Raw Score'] for r in st.session_state.get('candidates', [])]
            if scores:
                st.metric("Avg", f"{np.mean(scores):.0f}%")
                st.metric("Max", f"{max(scores):.0f}%")
                st.metric("Total", len(scores))
    
    st.markdown("---")
    
    # Input Section - Horizontal layout
    st.header("Input Section")
    
    # Job Description Input - Full width
    job_description = st.text_area(
        "Job Description:",
        height=150,
        placeholder="Paste the job description here...",
        key="job_description_main"
    )
    
    # Resume Input Section - Horizontal layout
    col_upload, col_text = st.columns([1, 1])
    
    with col_upload:
        st.subheader("Upload Resume Files")
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF, DOCX):",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple resume files",
            key="resume_file_uploader"
        )
    
    with col_text:
        st.subheader("Add Resumes as Text")
        num_text_inputs = st.number_input("Number of text resumes to add:", min_value=0, max_value=10, value=0, key="num_text_resumes")
    
    # Text input areas for resumes - wrap across screen
    if num_text_inputs > 0:
        st.subheader("Text Resume Input:")
        
        # Create columns for text inputs (2 per row)
        text_resumes = []
        cols_per_row = 2
        for row in range((num_text_inputs + cols_per_row - 1) // cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                i = row * cols_per_row + col_idx
                if i < num_text_inputs:
                    with cols[col_idx]:
                        resume_text = st.text_area(
                            f"Resume {i+1}:",
                            height=120,
                            placeholder=f"Paste candidate {i+1} resume here...",
                            key=f"resume_text_{i}"
                        )
                        if resume_text.strip():
                            text_resumes.append({
                                'name': f"Candidate {i+1}",
                                'content': resume_text.strip()
                            })
    else:
        text_resumes = []
    
    # Process button - centered
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        process_button = st.button("Find Best Candidates", type="primary", use_container_width=True, key="process_candidates_btn")
    
    st.markdown("---")
    
    # Results Section
    st.header("Results")
    
    if process_button and job_description.strip():
        with st.spinner("Processing candidates with Gemini AI..."):
            # Process uploaded files
            candidates = []
            
            # Process uploaded files
            for i, uploaded_file in enumerate(uploaded_files):
                file_content = ""
                if uploaded_file.type == "application/pdf":
                    file_content = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    file_content = extract_text_from_docx(uploaded_file)
                
                if file_content:
                    candidates.append({
                        'name': f"File {i+1}: {uploaded_file.name}",
                        'content': file_content
                    })
            
            # Add text resumes
            candidates.extend(text_resumes)
            
            if not candidates:
                st.warning("Please upload files or add text resumes to process.")
                return
            
            # Process candidates with Gemini
            st.info(f"Processing {len(candidates)} candidates with Gemini AI...")
            
            results = []
            progress_bar = st.progress(0)
            api_limit_reached = False
            
            for i, candidate in enumerate(candidates):
                # Update progress
                progress = (i + 1) / len(candidates)
                progress_bar.progress(progress)
                
                try:
                    # Get similarity score
                    similarity_score = get_similarity_score_with_gemini(
                        job_description, 
                        candidate['content'], 
                        st.session_state.get('gemini_model')
                    )
                    
                    # Generate AI summary
                    ai_summary = generate_ai_summary_with_gemini(
                        job_description,
                        candidate['content'],
                        similarity_score,
                        st.session_state.get('gemini_model')
                    )
                    
                    results.append({
                        'Name': candidate['name'],
                        'Similarity Score': f"{similarity_score:.1f}%",
                        'Raw Score': similarity_score,
                        'Content': candidate['content'],
                        'AI Summary': ai_summary
                    })
                    
                except Exception as e:
                    if "API_RATE_LIMIT_EXCEEDED" in str(e) or "API_QUOTA_EXCEEDED" in str(e):
                        api_limit_reached = True
                        st.warning(f"⏸️ Processing stopped at candidate {i+1} due to API rate limit.")
                        break
                    else:
                        st.error(f"Error processing candidate {i+1}: {str(e)}")
                        continue
            
            if api_limit_reached:
                st.error("Processing stopped due to API rate limit. Please try again after the quota resets.")
                if results:
                    st.info(f"Successfully processed {len(results)} candidates before hitting the limit.")
            elif not results:
                st.error("No candidates were processed successfully.")
                return
            else:
                st.success(f"Processed {len(results)} candidates successfully!")
            
            # Sort by similarity score
            results.sort(key=lambda x: x['Raw Score'], reverse=True)
            
            # Display top candidates
            st.subheader("**Top Candidates**")
            
            for i, result in enumerate(results[:10]):  # Show top 10
                with st.expander(f"#{i+1} - {result['Name']} ({result['Similarity Score']})"):
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.markdown(f"**Similarity Score:** {result['Similarity Score']}")
                        
                        # Display AI summary automatically
                        st.markdown("**AI Analysis:**")
                        st.info(result['AI Summary'])
                    
                    with col_b:
                        st.markdown("**Resume Preview:**")
                        preview = result['Content'][:300] + "..." if len(result['Content']) > 300 else result['Content']
                        st.text(preview)
            
            # Store results in session state
            st.session_state['candidates'] = results
            st.session_state['job_description'] = job_description
    
    elif process_button and not job_description.strip():
        st.error("Please enter a job description to proceed.")
    
    # Display previous results if available
    elif st.session_state.get('candidates'):
        st.subheader("Previous Results")
        for i, result in enumerate(st.session_state.get('candidates', [])[:5]):
            st.markdown(f"**{i+1}. {result['Name']}** - {result['Similarity Score']}")

if __name__ == "__main__":
    main() 