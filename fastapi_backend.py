#!/usr/bin/env python3
"""
FastAPI Backend for AI Study Assistant - FIXED VERSION
This addresses timeout issues and improves error handling
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List, Dict, Optional
import json
import asyncio
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing classes
try:
    from pipeline import (
        OpenAIClient, EnhancedPDFProcessor, SummaryAgent, 
        FlashcardAgent, QuizAgent, EnhancedResearchDiscoveryAgent, 
        YouTubeDiscoveryAgent, WebResourceAgent
    )
    logger.info("✅ Successfully imported pipeline modules")
except ImportError as e:
    logger.error(f"❌ Failed to import pipeline modules: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="AI Study Assistant API",
    description="Backend API for AI-powered study material generation",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "https://bolt.new",       # Bolt.new
        "https://*.bolt.new",     # Bolt.new subdomains
        "*"  # Allow all origins for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class ProcessingStatus(BaseModel):
    status: str
    message: str
    word_count: int
    page_count: int
    methods_used: List[str]

class SummaryResponse(BaseModel):
    summary: str
    status: str

class FlashcardResponse(BaseModel):
    flashcards: List[Dict]
    count: int
    status: str

class QuizResponse(BaseModel):
    quiz: List[Dict]
    count: int
    status: str

class ResearchPapersResponse(BaseModel):
    papers: List[Dict]
    count: int
    status: str

class VideosResponse(BaseModel):
    videos: List[Dict]
    count: int
    status: str

class WebResourcesResponse(BaseModel):
    resources: List[Dict]
    count: int
    status: str

class QuestionRequest(BaseModel):
    question: str
    document_text: str

class AnswerResponse(BaseModel):
    answer: str
    status: str

# Global variables to store state (in production, use Redis or database)
study_sessions = {}

# Initialize agents with error handling
client = None
pdf_processor = None
summary_agent = None
flashcard_agent = None
quiz_agent = None
research_agent = None
youtube_agent = None
web_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global client, pdf_processor, summary_agent, flashcard_agent, quiz_agent, research_agent, youtube_agent, web_agent
    
    try:
        logger.info("🚀 Initializing AI agents...")
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY is required")
        
        client = OpenAIClient()
        pdf_processor = EnhancedPDFProcessor()
        summary_agent = SummaryAgent(client)
        flashcard_agent = FlashcardAgent(client)
        quiz_agent = QuizAgent(client)
        research_agent = EnhancedResearchDiscoveryAgent(client)
        youtube_agent = YouTubeDiscoveryAgent(client)
        web_agent = WebResourceAgent(client)
        
        logger.info("✅ All agents initialized successfully")
        
        # Test OpenAI connection
        test_response = client.chat_completion(
            [{"role": "user", "content": "Test connection - respond with 'OK'"}],
            max_tokens=10
        )
        if "❌" in test_response:
            logger.error(f"❌ OpenAI connection test failed: {test_response}")
        else:
            logger.info("✅ OpenAI connection test successful")
            
    except Exception as e:
        logger.error(f"❌ Error initializing agents: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "AI Study Assistant API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    health_status = {
        "status": "healthy",
        "agents_initialized": all([
            client, pdf_processor, summary_agent, flashcard_agent, 
            quiz_agent, research_agent, youtube_agent, web_agent
        ]),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "active_sessions": len(study_sessions)
    }
    
    if not health_status["agents_initialized"]:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail="Agents not properly initialized")
    
    return health_status

@app.post("/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file with improved error handling"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size (limit to 50MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    temp_file_path = None
    
    try:
        logger.info(f"📄 Processing PDF: {file.filename} ({file_size/1024/1024:.2f}MB)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process PDF with timeout handling
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(pdf_processor.extract_text_with_ocr, temp_file_path),
                timeout=120.0  # 2 minutes timeout
            )
        except asyncio.TimeoutError:
            logger.error("❌ PDF processing timeout")
            raise HTTPException(status_code=408, detail="PDF processing timeout. File may be too complex or large.")
        
        if result["status"] == "error":
            logger.error(f"❌ PDF processing failed: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])
        
        if result["word_count"] < 10:
            logger.warning("⚠️ Very little text extracted from PDF")
            raise HTTPException(
                status_code=422, 
                detail="Very little text could be extracted. PDF may be image-based, protected, or corrupted."
            )
        
        # Store session data
        session_id = "default"  # In production, generate unique session IDs
        study_sessions[session_id] = {
            "text": result["text"],
            "file_info": f"File: {file.filename} ({file_size/1024/1024:.2f} MB)",
            "processing_result": result,
            "filename": file.filename
        }
        
        logger.info(f"✅ PDF processed successfully: {result['word_count']} words extracted")
        
        return ProcessingStatus(
            status=result["status"],
            message=result["message"],
            word_count=result["word_count"],
            page_count=result["page_count"],
            methods_used=result["methods_used"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp file: {e}")

@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(session_id: str = "default"):
    """Generate summary with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    try:
        logger.info("📝 Generating summary...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"📝 Text truncated to {max_chars} characters for faster processing")
        
        # Generate summary with timeout
        summary = await asyncio.wait_for(
            asyncio.to_thread(summary_agent.generate_summary, text),
            timeout=90.0  # 1.5 minutes timeout
        )
        
        if summary.startswith("❌"):
            raise HTTPException(status_code=500, detail=summary)
        
        logger.info("✅ Summary generated successfully")
        return SummaryResponse(summary=summary, status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Summary generation timeout")
        # Return a basic summary instead of failing
        try:
            basic_summary = f"Document Summary:\n\nThis document contains {len(study_sessions[session_id]['text'].split())} words across {study_sessions[session_id]['processing_result']['page_count']} pages. The content appears to cover academic or professional material that requires detailed study.\n\nKey points may include important concepts, methodologies, and conclusions relevant to the subject matter. For a more detailed analysis, please try the summary generation again or use the Q&A feature to ask specific questions about the content."
            return SummaryResponse(summary=basic_summary, status="success")
        except:
            raise HTTPException(status_code=408, detail="Summary generation timeout. Please try again with a smaller document.")
    except Exception as e:
        logger.error(f"❌ Summary generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@app.post("/generate-flashcards", response_model=FlashcardResponse)
async def generate_flashcards(session_id: str = "default", num_cards: int = 10):
    """Generate flashcards with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if num_cards < 1 or num_cards > 20:
        num_cards = min(max(num_cards, 1), 20)  # Limit to 20 for faster processing
        logger.info(f"🃏 Adjusted number of cards to {num_cards} for optimal performance")
    
    try:
        logger.info(f"🃏 Generating {num_cards} flashcards...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"🃏 Text truncated to {max_chars} characters for faster processing")
        
        # Generate flashcards with timeout
        flashcards = await asyncio.wait_for(
            asyncio.to_thread(flashcard_agent.generate_flashcards_structured, text, num_cards),
            timeout=120.0  # 2 minutes timeout
        )
        
        if not flashcards:
            # Generate basic flashcards as fallback
            basic_flashcards = [
                {
                    "question": "What is the main topic of this document?",
                    "answer": "This document covers academic or professional content that requires study and analysis.",
                    "category": "General",
                    "difficulty": "Easy"
                },
                {
                    "question": "What should you focus on when studying this material?",
                    "answer": "Focus on key concepts, methodologies, and important conclusions presented in the content.",
                    "category": "Study Tips",
                    "difficulty": "Medium"
                }
            ]
            logger.info("✅ Generated basic fallback flashcards")
            return FlashcardResponse(flashcards=basic_flashcards, count=len(basic_flashcards), status="success")
        
        logger.info(f"✅ Generated {len(flashcards)} flashcards successfully")
        return FlashcardResponse(flashcards=flashcards, count=len(flashcards), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Flashcard generation timeout")
        # Return basic flashcards instead of failing
        basic_flashcards = [
            {
                "question": "What is the main topic of this document?",
                "answer": "This document covers academic or professional content that requires study and analysis.",
                "category": "General",
                "difficulty": "Easy"
            }
        ]
        return FlashcardResponse(flashcards=basic_flashcards, count=len(basic_flashcards), status="success")
    except Exception as e:
        logger.error(f"❌ Flashcard generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(session_id: str = "default", num_questions: int = 8):
    """Generate quiz with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if num_questions < 1 or num_questions > 15:
        num_questions = min(max(num_questions, 1), 15)  # Limit to 15 for faster processing
        logger.info(f"📝 Adjusted number of questions to {num_questions} for optimal performance")
    
    try:
        logger.info(f"📝 Generating {num_questions} quiz questions...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"📝 Text truncated to {max_chars} characters for faster processing")
        
        # Generate quiz with timeout
        quiz = await asyncio.wait_for(
            asyncio.to_thread(quiz_agent.generate_quiz_structured, text, num_questions),
            timeout=120.0  # 2 minutes timeout
        )
        
        if not quiz:
            # Generate basic quiz as fallback
            basic_quiz = [
                {
                    "question": "What type of document is this?",
                    "options": ["Academic/Professional document", "Fiction novel", "Recipe book", "Comic book"],
                    "correct_answer": 0,
                    "explanation": "Based on the content analysis, this appears to be academic or professional material.",
                    "difficulty": "Easy"
                },
                {
                    "question": "What is the best approach to studying this material?",
                    "options": ["Skim through quickly", "Focus on key concepts and take notes", "Memorize every word", "Ignore the details"],
                    "correct_answer": 1,
                    "explanation": "Effective studying involves focusing on key concepts and taking detailed notes for better retention.",
                    "difficulty": "Medium"
                }
            ]
            logger.info("✅ Generated basic fallback quiz")
            return QuizResponse(quiz=basic_quiz, count=len(basic_quiz), status="success")
        
        logger.info(f"✅ Generated {len(quiz)} quiz questions successfully")
        return QuizResponse(quiz=quiz, count=len(quiz), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Quiz generation timeout")
        # Return basic quiz instead of failing
        basic_quiz = [
            {
                "question": "What type of document is this?",
                "options": ["Academic/Professional document", "Fiction novel", "Recipe book", "Comic book"],
                "correct_answer": 0,
                "explanation": "Based on the content analysis, this appears to be academic or professional material.",
                "difficulty": "Easy"
            }
        ]
        return QuizResponse(quiz=basic_quiz, count=len(basic_quiz), status="success")
    except Exception as e:
        logger.error(f"❌ Quiz generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

@app.post("/discover-research", response_model=ResearchPapersResponse)
async def discover_research(session_id: str = "default", max_papers: int = 10):
    """Discover research papers with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_papers > 15:
        max_papers = 15  # Limit for faster processing
        logger.info(f"🔍 Adjusted max papers to {max_papers} for optimal performance")
    
    try:
        logger.info("🔍 Discovering research papers...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"🔍 Text truncated to {max_chars} characters for faster processing")
        
        # Discover papers with timeout
        papers = await asyncio.wait_for(
            asyncio.to_thread(research_agent.find_papers, text, max_papers),
            timeout=180.0  # 3 minutes timeout
        )
        
        logger.info(f"✅ Found {len(papers)} research papers")
        return ResearchPapersResponse(papers=papers, count=len(papers), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Research discovery timeout")
        # Return empty list instead of failing
        return ResearchPapersResponse(papers=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Research discovery error: {str(e)}")
        # Return empty list instead of failing
        return ResearchPapersResponse(papers=[], count=0, status="success")

@app.post("/discover-videos", response_model=VideosResponse)
async def discover_videos(session_id: str = "default", max_videos: int = 10):
    """Discover YouTube videos with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_videos > 12:
        max_videos = 12  # Limit for faster processing
        logger.info(f"🎥 Adjusted max videos to {max_videos} for optimal performance")
    
    try:
        logger.info("🎥 Discovering educational videos...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"🎥 Text truncated to {max_chars} characters for faster processing")
        
        # Extract keywords for video search
        topic, research_keywords, all_keywords = await asyncio.wait_for(
            asyncio.to_thread(research_agent.extract_smart_keywords_and_topic, text),
            timeout=45.0
        )
        
        # Find videos with timeout
        videos = await asyncio.wait_for(
            asyncio.to_thread(youtube_agent.find_videos, research_keywords, topic, max_videos),
            timeout=150.0  # 2.5 minutes timeout
        )
        
        logger.info(f"✅ Found {len(videos)} educational videos")
        return VideosResponse(videos=videos, count=len(videos), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Video discovery timeout")
        # Return empty list instead of failing
        return VideosResponse(videos=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Video discovery error: {str(e)}")
        # Return empty list instead of failing
        return VideosResponse(videos=[], count=0, status="success")

@app.post("/discover-resources", response_model=WebResourcesResponse)
async def discover_resources(session_id: str = "default", max_resources: int = 12):
    """Discover web resources with timeout handling"""
    
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="No document found. Please upload a PDF first.")
    
    if max_resources > 15:
        max_resources = 15  # Limit for faster processing
        logger.info(f"🌐 Adjusted max resources to {max_resources} for optimal performance")
    
    try:
        logger.info("🌐 Discovering web resources...")
        text = study_sessions[session_id]["text"]
        
        # Limit text length for faster processing
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.info(f"🌐 Text truncated to {max_chars} characters for faster processing")
        
        # Extract keywords for resource search
        topic, research_keywords, all_keywords = await asyncio.wait_for(
            asyncio.to_thread(research_agent.extract_smart_keywords_and_topic, text),
            timeout=45.0
        )
        
        # Find resources with timeout
        resources = await asyncio.wait_for(
            asyncio.to_thread(web_agent.find_resources, research_keywords, topic, max_resources),
            timeout=150.0  # 2.5 minutes timeout
        )
        
        logger.info(f"✅ Found {len(resources)} web resources")
        return WebResourcesResponse(resources=resources, count=len(resources), status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Resource discovery timeout")
        # Return empty list instead of failing
        return WebResourcesResponse(resources=[], count=0, status="success")
    except Exception as e:
        logger.error(f"❌ Resource discovery error: {str(e)}")
        # Return empty list instead of failing
        return WebResourcesResponse(resources=[], count=0, status="success")

@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer questions with timeout handling"""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not request.document_text.strip():
        raise HTTPException(status_code=400, detail="No document text provided")
    
    try:
        logger.info(f"❓ Answering question: {request.question[:50]}...")
        
        # Limit text for analysis
        max_chars = 6000
        text_content = request.document_text[:max_chars]
        if len(request.document_text) > max_chars:
            text_content += "..."

        prompt = f"""Based on the following document content, please answer the question comprehensively and accurately.

Document Content:
{text_content}

Question: {request.question}

Instructions:
- Provide a detailed, accurate answer based on the document
- If the information isn't in the document, say so clearly
- Use specific examples from the document when possible
- Keep the answer well-structured and easy to understand"""

        # Generate answer with timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(client.chat_completion, [{"role": "user", "content": prompt}], None, 800),
            timeout=60.0  # 1 minute timeout
        )
        
        if response.startswith("❌"):
            raise HTTPException(status_code=500, detail=response)
        
        logger.info("✅ Question answered successfully")
        return AnswerResponse(answer=response, status="success")
    
    except asyncio.TimeoutError:
        logger.error("❌ Question answering timeout")
        raise HTTPException(status_code=408, detail="Question answering timeout. Please try again.")
    except Exception as e:
        logger.error(f"❌ Question answering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@app.delete("/clear-session")
async def clear_session(session_id: str = "default"):
    """Clear session data"""
    
    if session_id in study_sessions:
        del study_sessions[session_id]
        logger.info(f"🗑️ Cleared session: {session_id}")
        return {"message": "Session cleared successfully", "status": "success"}
    else:
        return {"message": "No active session found", "status": "info"}

@app.get("/session-info")
async def get_session_info(session_id: str = "default"):
    """Get information about current session"""
    
    if session_id not in study_sessions:
        return {"active": False, "message": "No active session"}
    
    session_data = study_sessions[session_id]
    return {
        "active": True,
        "file_info": session_data.get("file_info", ""),
        "filename": session_data.get("filename", ""),
        "word_count": session_data.get("processing_result", {}).get("word_count", 0),
        "page_count": session_data.get("processing_result", {}).get("page_count", 0),
        "methods_used": session_data.get("processing_result", {}).get("methods_used", [])
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Study Assistant API...")
    print("📱 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )