import gradio as gr
import os
import json
import re
import asyncio
import requests
from datetime import datetime
from urllib.parse import quote_plus
from pipeline import (
    OpenAIClient, EnhancedPDFProcessor, SummaryAgent, 
    FlashcardAgent, QuizAgent, EnhancedResearchDiscoveryAgent, 
    YouTubeDiscoveryAgent, WebResourceAgent
)

class StudyAssistantOrchestrator:
    def __init__(self):
        try:
            self.client = OpenAIClient()
            self.processor = EnhancedPDFProcessor()
            self.summary_agent = SummaryAgent(self.client)
            self.flashcard_agent = FlashcardAgent(self.client)
            self.quiz_agent = QuizAgent(self.client)
            
            # Enhanced discovery agents from updated pipeline
            self.research_agent = EnhancedResearchDiscoveryAgent(self.client)
            self.youtube_agent = YouTubeDiscoveryAgent(self.client)
            self.web_resource_agent = WebResourceAgent(self.client)
            
            print("✅ All agents initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            raise
        
        self.last_processed_text = ""
        self.last_file_info = ""
        self.document_keywords = []
        self.document_topic = ""
        self.enhanced_keywords = []  # New: enhanced keywords from pipeline
        
        # States for features
        self.current_quiz_data = None
        self.current_flashcard_data = None
        self.quiz_state = {
            "current_question": 0,
            "answers": {},
            "score": 0,
            "completed": False
        }
        self.flashcard_state = {
            "current_card": 0,
            "show_answer": False
        }
        
        # Discovery data
        self.research_papers = []
        self.youtube_videos = []
        self.web_resources = []

    def process_pdf(self, file, progress=gr.Progress()):
        """Process uploaded PDF and generate all materials including enhanced discoveries"""
        if file is None:
            return self._empty_outputs("❌ No file uploaded.")

        try:
            file_path = file.name if hasattr(file, "name") else file
            
            if not os.path.exists(file_path):
                return self._empty_outputs("❌ File not found. Please try uploading again.")

            # Get file info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.last_file_info = f"📁 File: {os.path.basename(file_path)} ({file_size:.2f} MB)"

            progress(0.1, desc="📄 Extracting text from PDF...")
            
            # Extract text from PDF using enhanced processor
            result = self.processor.extract_text_with_ocr(file_path)

            if result["status"] == "error":
                error_msg = f"❌ {result.get('message', 'Failed to process PDF.')}"
                return self._empty_outputs(error_msg)

            # Store the extracted text
            self.last_processed_text = result["text"]
            
            progress(0.2, desc="🔍 Analyzing document content with enhanced AI...")
            
            # Enhanced keyword extraction using the new pipeline method
            self._extract_enhanced_keywords_and_topic()
            
            progress(0.3, desc="📝 Generating intelligent summary...")
            
            # Generate study materials
            summary = self._generate_summary()
            
            progress(0.4, desc="🃏 Creating interactive flashcards...")
            flashcards_data = self._generate_flashcards()
            
            progress(0.5, desc="📝 Creating adaptive quiz...")
            quiz_data = self._generate_quiz()
            
            progress(0.6, desc="📚 Discovering research papers with AI...")
            
            # Enhanced research paper discovery
            self.research_papers = self._discover_research_papers_enhanced()
            
            progress(0.7, desc="🎥 Finding educational YouTube content...")
            
            # Discover YouTube videos
            self.youtube_videos = self._discover_youtube_videos()
            
            progress(0.8, desc="🌐 Finding web learning resources...")
            
            # Discover web resources
            self.web_resources = self._discover_web_resources()
            
            progress(0.9, desc="🔧 Finalizing and organizing results...")
            
            # Reset states
            self._reset_quiz_state()
            self._reset_flashcard_state()
            
            # Enhanced info message with processing details
            info_msg = f"{self.last_file_info}\n{result['message']}"
            if result.get("methods_used"):
                info_msg += f"\n📊 Processing methods: {', '.join(result['methods_used'])}"
            if result["word_count"] < 50:
                info_msg += "\n⚠️ Limited content extracted. Results may be basic."
            if result.get("error_details"):
                info_msg += f"\n⚠️ Minor issues: {len(result['error_details'])} warnings"

            progress(1.0, desc="✅ Complete!")
            
            return self._format_outputs(summary, flashcards_data, quiz_data, info_msg)

        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(f"Error in process_pdf: {e}")
            return self._empty_outputs(error_msg)

    def _extract_enhanced_keywords_and_topic(self):
        """Extract enhanced keywords using the new pipeline method"""
        if not self.last_processed_text.strip():
            return
        
        try:
            # Use the enhanced research agent's keyword extraction
            topic, research_keywords, all_keywords = self.research_agent.extract_smart_keywords_and_topic(
                self.last_processed_text
            )
            
            self.document_topic = topic
            self.document_keywords = research_keywords  # Most relevant research keywords
            self.enhanced_keywords = all_keywords  # All extracted keywords
            
            print(f"✅ Enhanced extraction - Topic: {self.document_topic}")
            print(f"✅ Research keywords: {self.document_keywords}")
            print(f"✅ All keywords: {len(self.enhanced_keywords)} total")
                
        except Exception as e:
            print(f"❌ Error in enhanced keyword extraction: {e}")
            # Fallback to basic extraction
            self._extract_keywords_and_topic_fallback()

    def _extract_keywords_and_topic_fallback(self):
        """Fallback keyword extraction method"""
        if not self.last_processed_text.strip():
            return
        
        try:
            # Limit text for analysis
            analysis_text = self.last_processed_text[:4000]
            
            prompt = f"""Analyze the following academic content and extract key information.

Content:
{analysis_text}

Return ONLY a JSON object with this exact format:
{{
    "main_topic": "Primary subject area",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

Focus on the most important academic concepts and terms."""
            
            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=1000)
            
            # Clean response to extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON response
            data = json.loads(response)
            self.document_topic = data.get("main_topic", "Academic Study Material")
            self.document_keywords = data.get("keywords", ["study", "learning", "education"])
            self.enhanced_keywords = self.document_keywords.copy()
            
            print(f"✅ Fallback extraction - Topic: {self.document_topic}")
            print(f"✅ Keywords: {self.document_keywords}")
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"❌ Fallback keyword extraction failed: {e}")
            # Ultimate fallback
            self.document_topic = "Academic Study Material"
            self.document_keywords = ["study", "learning", "education"]
            self.enhanced_keywords = self.document_keywords.copy()

    def clear_session(self):
        """Clear the current session"""
        self.last_processed_text = ""
        self.last_file_info = ""
        self.document_keywords = []
        self.document_topic = ""
        self.enhanced_keywords = []
        self.current_quiz_data = []
        self.current_flashcard_data = []
        self.research_papers = []
        self.youtube_videos = []
        self.web_resources = []
        self._reset_quiz_state()
        self._reset_flashcard_state()
        
        return self._empty_outputs("🔄 Session cleared. Ready for new document.")

    def _discover_research_papers_enhanced(self):
        """Enhanced research paper discovery using the new pipeline"""
        if not self.last_processed_text:
            return []
        
        try:
            print("🔍 Enhanced research paper discovery...")
            papers = self.research_agent.find_papers(
                text=self.last_processed_text,
                max_papers=10
            )
            print(f"✅ Found {len(papers)} high-quality research papers")
            return papers
        except Exception as e:
            print(f"❌ Error in enhanced research discovery: {e}")
            return []

    def _discover_youtube_videos(self):
        """Discover related YouTube videos"""
        if not self.document_keywords:
            return []
        
        try:
            print("🎥 Searching for educational YouTube videos...")
            videos = self.youtube_agent.find_videos(
                keywords=self.document_keywords,
                topic=self.document_topic,
                max_videos=10
            )
            print(f"✅ Found {len(videos)} educational YouTube videos")
            return videos
        except Exception as e:
            print(f"❌ Error discovering YouTube videos: {e}")
            return []

    def _discover_web_resources(self):
        """Discover related web resources"""
        if not self.document_keywords:
            return []
        
        try:
            print("🌐 Searching for web learning resources...")
            resources = self.web_resource_agent.find_resources(
                keywords=self.document_keywords,
                topic=self.document_topic,
                max_resources=12
            )
            print(f"✅ Found {len(resources)} web learning resources")
            return resources
        except Exception as e:
            print(f"❌ Error discovering web resources: {e}")
            return []

    def _empty_outputs(self, message):
        """Return empty outputs with error message"""
        return (
            message, # summary
            message, # flashcard_question
            "", # flashcard_answer
            "0/0", # flashcard_progress
            gr.update(visible=False), # flashcard_answer_group
            gr.update(variant="secondary"), # show_answer_btn
            message, # quiz_question
            gr.update(choices=[], value=None), # quiz_options
            "", # quiz_feedback
            "0/0 - Score: 0%", # quiz_progress
            gr.update(visible=False), # quiz_nav_group
            gr.update(interactive=False), # prev_btn
            gr.update(interactive=False), # next_btn
            gr.update(interactive=False), # submit_btn
            message, # doc_info
            "", # answer_output
            message, # research_papers_output
            message, # youtube_videos_output
            message  # web_resources_output
        )

    def _format_outputs(self, summary, flashcards_data, quiz_data, info_msg):
        """Format outputs for initial display"""
        # Flashcard initial state
        if flashcards_data and len(flashcards_data) > 0:
            flashcard_question = f"**{flashcards_data[0]['question']}**"
            if flashcards_data[0].get('category'):
                flashcard_question += f"\n\n*Category: {flashcards_data[0]['category']}*"
            flashcard_progress = f"1/{len(flashcards_data)}"
        else:
            flashcard_question = "No flashcards available"
            flashcard_progress = "0/0"
        
        # Quiz initial state
        if quiz_data and len(quiz_data) > 0:
            quiz_question = quiz_data[0]["question"]
            quiz_options = quiz_data[0]["options"]
            quiz_progress = f"1/{len(quiz_data)} - Score: 0%"
            quiz_nav_visible = True
        else:
            quiz_question = "No quiz available"
            quiz_options = []
            quiz_progress = "0/0 - Score: 0%"
            quiz_nav_visible = False
        
        # Format discovery outputs with enhanced information
        research_output = self._format_research_papers_enhanced()
        youtube_output = self._format_youtube_videos()
        web_resources_output = self._format_web_resources()
        
        return (
            summary, # summary
            flashcard_question, # flashcard_question
            "", # flashcard_answer (hidden initially)
            flashcard_progress, # flashcard_progress
            gr.update(visible=False), # flashcard_answer_group
            gr.update(variant="primary"), # show_answer_btn
            quiz_question, # quiz_question
            gr.update(choices=quiz_options, value=None), # quiz_options
            "", # quiz_feedback
            quiz_progress, # quiz_progress
            gr.update(visible=quiz_nav_visible), # quiz_nav_group
            gr.update(interactive=False), # prev_btn (disabled on first question)
            gr.update(interactive=True), # next_btn
            gr.update(interactive=True), # submit_btn
            info_msg, # doc_info
            "", # answer_output
            research_output, # research_papers_output
            youtube_output, # youtube_videos_output
            web_resources_output # web_resources_output
        )

    def _format_research_papers_enhanced(self):
        """Enhanced formatting for research papers with relevance scores"""
        if not self.research_papers:
            return "🔍 No research papers found. Try uploading a more specific academic document."
        
        output = f"# 📚 Related Research Papers ({len(self.research_papers)} found)\n\n"
        output += f"**Topic Focus:** {self.document_topic}\n"
        output += f"**Key Research Terms:** {', '.join(self.document_keywords[:5])}\n\n"
        
        for i, paper in enumerate(self.research_papers, 1):
            output += f"## {i}. {paper.get('title', 'Untitled Paper')}\n"
            output += f"**Authors:** {paper.get('authors', 'Unknown')}\n"
            output += f"**Year:** {paper.get('year', 'N/A')}\n"
            output += f"**Source:** {paper.get('source', 'Academic Database')}\n"
            
            # Enhanced relevance information
            if paper.get('relevance_label'):
                output += f"**Relevance:** {paper['relevance_label']}\n"
            elif paper.get('relevance_score'):
                score = paper['relevance_score']
                if isinstance(score, (int, float)):
                    percentage = int(score * 100)
                    output += f"**Relevance:** {percentage}% match\n"
            
            # Additional metadata from enhanced pipeline
            if paper.get('citation_count'):
                output += f"**Citations:** {paper['citation_count']}\n"
            if paper.get('fields_of_study'):
                output += f"**Fields:** {', '.join(paper['fields_of_study'][:3])}\n"
            if paper.get('categories'):
                output += f"**Categories:** {', '.join(paper['categories'][:3])}\n"
            
            if paper.get('abstract'):
                output += f"**Abstract:** {paper['abstract']}\n"
            
            if paper.get('url') and paper['url'] != '#':
                output += f"**Link:** [{paper['url']}]({paper['url']})\n"
            
            output += "\n---\n\n"
        
        return output

    def _format_youtube_videos(self):
        """Format YouTube videos for display"""
        if not self.youtube_videos:
            return "🎥 No YouTube videos found. Try uploading a document with more specific topics."
        
        output = f"# 🎥 Educational YouTube Videos ({len(self.youtube_videos)} found)\n\n"
        output += f"**Topic Focus:** {self.document_topic}\n\n"
        
        for i, video in enumerate(self.youtube_videos, 1):
            output += f"## {i}. {video.get('title', 'Untitled Video')}\n"
            output += f"**Channel:** {video.get('channel', 'Unknown Channel')}\n"
            output += f"**Duration:** {video.get('duration', 'N/A')}\n"
            output += f"**Views:** {video.get('views', 'N/A')}\n"
            
            if video.get('description'):
                output += f"**Description:** {video['description']}\n"
            
            if video.get('url') and video['url'] != '#':
                output += f"**Watch:** [{video['url']}]({video['url']})\n"
            
            output += f"**Educational Value:** {video.get('educational_score', 'High')}\n\n"
            output += "---\n\n"
        
        return output

    def _format_web_resources(self):
        """Format web resources for display"""
        if not self.web_resources:
            return "🌐 No web resources found. Try uploading a document with more specific topics."
        
        output = f"# 🌐 Additional Learning Resources ({len(self.web_resources)} found)\n\n"
        output += f"**Topic Focus:** {self.document_topic}\n\n"
        
        for i, resource in enumerate(self.web_resources, 1):
            output += f"## {i}. {resource.get('title', 'Untitled Resource')}\n"
            output += f"**Type:** {resource.get('type', 'Web Resource')}\n"
            output += f"**Source:** {resource.get('source', 'Web')}\n"
            
            if resource.get('description'):
                output += f"**Description:** {resource['description']}\n"
            
            if resource.get('url') and resource['url'] != '#':
                output += f"**Link:** [{resource['url']}]({resource['url']})\n"
            
            output += f"**Quality Score:** {resource.get('quality_score', 'High')}\n\n"
            output += "---\n\n"
        
        return output

    def _generate_summary(self):
        """Generate enhanced summary"""
        if not self.last_processed_text.strip():
            return "❌ No content available to summarize."
        
        try:
            summary = self.summary_agent.generate_summary(self.last_processed_text)
            if summary.startswith("❌"):
                return f"❌ Summary generation failed.\n{summary}"
            return summary
        except Exception as e:
            return f"❌ Summary generation failed: {str(e)}"

    def _generate_flashcards(self):
        """Generate enhanced flashcards"""
        if not self.last_processed_text.strip():
            return []
        
        try:
            flashcards_data = self.flashcard_agent.generate_flashcards_structured(self.last_processed_text)
            self.current_flashcard_data = flashcards_data
            print(f"✅ Generated {len(flashcards_data)} flashcards")
            return flashcards_data
        except Exception as e:
            print(f"❌ Flashcard generation error: {e}")
            self.current_flashcard_data = []
            return []

    def _generate_quiz(self):
        """Generate enhanced quiz"""
        if not self.last_processed_text.strip():
            return []
        
        try:
            quiz_data = self.quiz_agent.generate_quiz_structured(self.last_processed_text)
            self.current_quiz_data = quiz_data
            print(f"✅ Generated {len(quiz_data)} quiz questions")
            return quiz_data
        except Exception as e:
            print(f"❌ Quiz generation error: {e}")
            self.current_quiz_data = []
            return []

    def _reset_quiz_state(self):
        """Reset quiz state"""
        self.quiz_state = {
            "current_question": 0,
            "answers": {},
            "score": 0,
            "completed": False
        }

    def _reset_flashcard_state(self):
        """Reset flashcard state"""
        self.flashcard_state = {
            "current_card": 0,
            "show_answer": False
        }

    # Flashcard functions
    def show_flashcard_answer(self):
        """Show answer for current flashcard"""
        if not self.current_flashcard_data or len(self.current_flashcard_data) == 0:
            return "", gr.update(visible=False), gr.update(variant="secondary")
        
        current_card = self.current_flashcard_data[self.flashcard_state["current_card"]]
        self.flashcard_state["show_answer"] = True
        
        answer_text = f"**Answer:** {current_card['answer']}"
        if current_card.get('hint'):
            answer_text += f"\n\n**💡 Hint:** {current_card['hint']}"
        answer_text += f"\n\n**📊 Difficulty:** {current_card['difficulty']}"
        
        return (
            answer_text,
            gr.update(visible=True),
            gr.update(variant="secondary")
        )

    def next_flashcard(self):
        """Move to next flashcard"""
        if not self.current_flashcard_data or len(self.current_flashcard_data) == 0:
            return self._flashcard_no_data()
        
        self.flashcard_state["current_card"] = min(
            self.flashcard_state["current_card"] + 1,
            len(self.current_flashcard_data) - 1
        )
        self.flashcard_state["show_answer"] = False
        
        return self._update_flashcard_display()

    def prev_flashcard(self):
        """Move to previous flashcard"""
        if not self.current_flashcard_data or len(self.current_flashcard_data) == 0:
            return self._flashcard_no_data()
        
        self.flashcard_state["current_card"] = max(
            self.flashcard_state["current_card"] - 1,
            0
        )
        self.flashcard_state["show_answer"] = False
        
        return self._update_flashcard_display()

    def _update_flashcard_display(self):
        """Update flashcard display"""
        if not self.current_flashcard_data or len(self.current_flashcard_data) == 0:
            return self._flashcard_no_data()
        
        current_card = self.current_flashcard_data[self.flashcard_state["current_card"]]
        progress = f"{self.flashcard_state['current_card'] + 1}/{len(self.current_flashcard_data)}"
        
        question_text = f"**{current_card['question']}**"
        if current_card.get('category'):
            question_text += f"\n\n*Category: {current_card['category']}*"
        
        return (
            question_text,
            "",
            progress,
            gr.update(visible=False),
            gr.update(variant="primary")
        )

    def _flashcard_no_data(self):
        """Return empty flashcard data"""
        return (
            "No flashcards available",
            "",
            "0/0",
            gr.update(visible=False),
            gr.update(variant="secondary")
        )

    # Quiz functions
    def submit_quiz_answer(self, selected_option):
        """Submit answer for current quiz question"""
        if not self.current_quiz_data or len(self.current_quiz_data) == 0 or self.quiz_state["completed"]:
            return self._quiz_no_data()
        
        if selected_option is None:
            return (
                "❌ Please select an answer first.",
                self._get_quiz_progress(),
                gr.update(interactive=False),
                gr.update(interactive=True)
            )
        
        current_q = self.quiz_state["current_question"]
        question_data = self.current_quiz_data[current_q]
        
        # Store answer
        self.quiz_state["answers"][current_q] = selected_option
        
        # Check if correct
        correct_option = question_data["options"][question_data["correct_answer"]]
        is_correct = selected_option == correct_option
        if is_correct:
            self.quiz_state["score"] += 1
        
        # Generate feedback
        feedback = self._generate_quiz_feedback(question_data, selected_option, is_correct)
        
        # Update progress
        progress = self._get_quiz_progress()
        
        return (
            feedback,
            progress,
            gr.update(interactive=True),  # Enable next button
            gr.update(interactive=False)  # Disable submit button
        )

    def next_quiz_question(self):
        """Move to next quiz question"""
        if not self.current_quiz_data or len(self.current_quiz_data) == 0:
            return self._quiz_no_data()
        
        # Move to next question
        self.quiz_state["current_question"] += 1
        
        # Check if completed
        if self.quiz_state["current_question"] >= len(self.current_quiz_data):
            return self._complete_quiz()
        
        return self._update_quiz_display()

    def prev_quiz_question(self):
        """Move to previous quiz question"""
        if not self.current_quiz_data or len(self.current_quiz_data) == 0:
            return self._quiz_no_data()
        
        self.quiz_state["current_question"] = max(
            self.quiz_state["current_question"] - 1,
            0
        )
        
        return self._update_quiz_display()

    def _update_quiz_display(self):
        """Update quiz question display"""
        if not self.current_quiz_data or len(self.current_quiz_data) == 0:
            return self._quiz_no_data()
        
        current_q = self.quiz_state["current_question"]
        question_data = self.current_quiz_data[current_q]
        
        # Get previous answer if exists
        previous_answer = self.quiz_state["answers"].get(current_q, None)
        
        progress = self._get_quiz_progress()
        
        prev_enabled = current_q > 0
        next_enabled = current_q in self.quiz_state["answers"]  # Only if answered
        
        return (
            question_data["question"],
            gr.update(choices=question_data["options"], value=previous_answer),
            "",  # Clear feedback
            progress,
            gr.update(visible=True),
            gr.update(interactive=prev_enabled),
            gr.update(interactive=next_enabled),
            gr.update(interactive=True)
        )

    def _complete_quiz(self):
        """Complete the quiz and show results"""
        self.quiz_state["completed"] = True
        total_questions = len(self.current_quiz_data)
        score_percent = (self.quiz_state["score"] / total_questions) * 100
        
        results = f"""🎉 **Quiz Completed!**

**Final Score: {self.quiz_state['score']}/{total_questions} ({score_percent:.1f}%)**

{"🌟 Excellent work!" if score_percent >= 90 else
 "👍 Good job!" if score_percent >= 70 else
 "📚 Keep studying!" if score_percent >= 50 else
 "💪 Don't give up!"}

Review your answers and study the explanations to improve your understanding."""
        
        return (
            results,
            gr.update(choices=[], value=None),
            "",
            f"Quiz Complete - Final Score: {score_percent:.1f}%",
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    def _generate_quiz_feedback(self, question_data, selected_option, is_correct):
        """Generate feedback for quiz answer"""
        correct_option = question_data["options"][question_data["correct_answer"]]
        
        if is_correct:
            feedback = f"✅ **Correct!** \n\n{question_data.get('explanation', 'Good job!')}"
        else:
            feedback = f"❌ **Incorrect.** \n\nYou selected: {selected_option}\nCorrect answer: {correct_option}\n\n{question_data.get('explanation', 'Study this topic more.')}"
        
        return feedback

    def _get_quiz_progress(self):
        """Get current quiz progress"""
        if not self.current_quiz_data:
            return "0/0 - Score: 0%"
        
        current = self.quiz_state["current_question"] + 1
        total = len(self.current_quiz_data)
        answered = len(self.quiz_state["answers"])
        score_percent = (self.quiz_state["score"] / max(answered, 1)) * 100 if answered > 0 else 0
        return f"{current}/{total} - Score: {score_percent:.1f}%"

    def _quiz_no_data(self):
        """Return empty quiz data"""
        return (
            "No quiz available",
            gr.update(choices=[], value=None),
            "",
            "0/0 - Score: 0%",
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    def handle_question(self, question):
        """Handle Q&A questions about the document with enhanced context"""
        if not question.strip():
            return "❓ Please enter a question."

        if not self.last_processed_text:
            return "❌ Please upload and process a PDF document first."

        try:
            max_chars = 6000
            text_content = self.last_processed_text[:max_chars]
            if len(self.last_processed_text) > max_chars:
                text_content += "..."

            # Enhanced prompt with document topic and keywords
            context_info = f"Document Topic: {self.document_topic}\nKey Concepts: {', '.join(self.document_keywords[:5])}"

            prompt = f"""Based on the following document content, please answer the question comprehensively and accurately.

{context_info}

Document Content:
{text_content}

Question: {question}

Instructions:
- Provide a detailed, accurate answer based on the document
- If the information isn't in the document, say so clearly
- Use specific examples from the document when possible
- Keep the answer well-structured and easy to understand
- Consider the document's main topic and key concepts in your response"""

            response = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=800)
            
            if response.startswith("❌"):
                return f"❌ Failed to generate answer: {response}"
            
            return response

        except Exception as e:
            return f"❌ Error processing question: {str(e)}"


# Initialize the orchestrator
try:
    orchestrator = StudyAssistantOrchestrator()
    print("✅ StudyAssistantOrchestrator initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize orchestrator: {e}")
    raise

def create_interface():
    """Create the enhanced Gradio interface with updated pipeline integration"""
    
    custom_css = """
    .gradio-container {
        max-width: 1600px;
        margin: auto;
    }
    .quiz-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .flashcard-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        min-height: 200px;
    }
    .discovery-container {
        border: 2px solid #f0f0f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #fafafa;
    }
    .score-display {
        font-size: 18px;
        font-weight: bold;
        color: #2196F3;
    }
    .enhanced-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        title="AI Study Assistant Pro - Enhanced", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
        # 🎓 AI Study Assistant Pro - Enhanced Discovery Edition
        
        **Now with Advanced AI Research Discovery & Enhanced Processing**
        
        Upload your PDFs and get comprehensive study materials with:
        - 📝 **Enhanced intelligent summaries** with deep content analysis
        - 🧠 **Interactive flashcards** with categories, hints, and difficulty levels *(FULLY FUNCTIONAL)*
        - 🎯 **Adaptive quiz system** with detailed feedback and progress tracking *(FULLY FUNCTIONAL)*
        - 💬 **Smart Q&A assistance** with enhanced document context *(FULLY FUNCTIONAL)*
        - 📚 **Advanced research paper discovery** from arXiv, Semantic Scholar, PubMed with relevance scoring
        - 🎥 **Curated YouTube educational content** from top educational channels
        - 🌐 **Comprehensive web learning resources** from verified educational platforms
        - 🔍 **Enhanced keyword extraction** with academic focus and smart categorization
        """, elem_classes=["enhanced-info"])

        with gr.Tab("📄 Document Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📎 Upload Your Document",
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🚀 Process & Discover", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 New Session", variant="secondary")
                    
                    doc_info = gr.Textbox(
                        label="📊 Processing Status & Details",
                        lines=10,
                        interactive=False,
                        info="Enhanced processing with detailed extraction methods and keyword analysis"
                    )

                with gr.Column(scale=2):
                    summary_output = gr.Textbox(
                        label="📝 Enhanced Intelligent Summary",
                        lines=30,
                        interactive=False,
                        placeholder="Your enhanced document summary with structured analysis will appear here...",
                        info="AI-powered summary with key concepts, takeaways, and study focus areas"
                    )

        with gr.Tab("🧠 Interactive Flashcards"):
            gr.Markdown("### 🎯 Smart Flashcards with Categories & Difficulty Levels")
            with gr.Column():
                with gr.Row():
                    flashcard_progress = gr.Textbox(
                        label="📊 Progress",
                        interactive=False,
                        scale=1
                    )
                
                flashcard_question = gr.Textbox(
                    label="🤔 Question",
                    lines=4,
                    interactive=False,
                    elem_classes=["flashcard-container"],
                    info="Enhanced flashcards with academic categorization"
                )
                
                with gr.Group(visible=False) as flashcard_answer_group:
                    flashcard_answer = gr.Textbox(
                        label="💡 Answer & Details",
                        lines=6,
                        interactive=False,
                        info="Complete answers with hints and difficulty indicators"
                    )
                
                with gr.Row():
                    prev_flashcard_btn = gr.Button("⬅️ Previous", size="sm")
                    show_answer_btn = gr.Button("👁️ Show Answer", variant="primary")
                    next_flashcard_btn = gr.Button("Next ➡️", size="sm")

        with gr.Tab("🎯 Adaptive Quiz System"):
            gr.Markdown("### 📚 Intelligent Quiz with Detailed Feedback")
            with gr.Column():
                quiz_progress = gr.Textbox(
                    label="📊 Progress & Score",
                    interactive=False,
                    elem_classes=["score-display"],
                    info="Real-time progress tracking with performance analytics"
                )
                
                quiz_question = gr.Textbox(
                    label="❓ Question",
                    lines=3,
                    interactive=False,
                    elem_classes=["quiz-container"],
                    info="AI-generated questions testing understanding and application"
                )
                
                quiz_options = gr.Radio(
                    label="📝 Select Your Answer",
                    choices=[],
                    interactive=True,
                    info="Multiple choice options with varying difficulty levels"
                )
                
                quiz_feedback = gr.Textbox(
                    label="💬 Detailed Feedback",
                    lines=4,
                    interactive=False,
                    info="Comprehensive explanations for both correct and incorrect answers"
                )
                
                with gr.Group(visible=False) as quiz_nav_group:
                    with gr.Row():
                        prev_quiz_btn = gr.Button("⬅️ Previous", interactive=False)
                        submit_answer_btn = gr.Button("✅ Submit Answer", variant="primary")
                        next_quiz_btn = gr.Button("Next ➡️", interactive=False)

        with gr.Tab("💬 Enhanced Q&A Assistant"):
            gr.Markdown("### 🤖 Smart Document Q&A with Context Awareness")
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        placeholder="What would you like to know about this document?",
                        label="🤔 Your Question",
                        lines=3,
                        info="Ask detailed questions about the document content"
                    )
                    ask_btn = gr.Button("🎯 Get Enhanced Answer", variant="primary")
                
                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="🎓 AI Response with Context",
                        lines=15,
                        interactive=False,
                        info="Contextually aware responses using document topic and key concepts"
                    )

        with gr.Tab("📚 Advanced Research Discovery"):
            gr.Markdown("### 🔍 Enhanced Research Papers from Academic Sources")
            gr.Markdown("*Intelligent discovery from arXiv, Semantic Scholar, and PubMed with relevance scoring*")
            
            research_papers_output = gr.Markdown(
                value="Upload and process a document to discover relevant research papers with enhanced AI analysis...",
                elem_classes=["discovery-container"]
            )

        with gr.Tab("🎥 Curated YouTube Content"):
            gr.Markdown("### 📺 Educational YouTube Videos from Top Channels")
            gr.Markdown("*Curated content from Khan Academy, Crash Course, MIT, TED-Ed, and more*")
            
            youtube_videos_output = gr.Markdown(
                value="Upload and process a document to discover educational YouTube content...",
                elem_classes=["discovery-container"]
            )

        with gr.Tab("🌐 Comprehensive Web Resources"):
            gr.Markdown("### 🔗 Learning Resources from Educational Platforms")
            gr.Markdown("*Courses, tutorials, and references from Coursera, edX, W3Schools, MDN, and more*")
            
            web_resources_output = gr.Markdown(
                value="Upload and process a document to discover comprehensive web learning resources...",
                elem_classes=["discovery-container"]
            )

        # Event handlers with enhanced functionality
        file_input.upload(
            orchestrator.process_pdf,
            inputs=[file_input],
            outputs=[
                summary_output, flashcard_question, flashcard_answer, flashcard_progress,
                flashcard_answer_group, show_answer_btn, quiz_question, quiz_options,
                quiz_feedback, quiz_progress, quiz_nav_group, prev_quiz_btn,
                next_quiz_btn, submit_answer_btn, doc_info, answer_output,
                research_papers_output, youtube_videos_output, web_resources_output
            ]
        )

        process_btn.click(
            orchestrator.process_pdf,
            inputs=[file_input],
            outputs=[
                summary_output, flashcard_question, flashcard_answer, flashcard_progress,
                flashcard_answer_group, show_answer_btn, quiz_question, quiz_options,
                quiz_feedback, quiz_progress, quiz_nav_group, prev_quiz_btn,
                next_quiz_btn, submit_answer_btn, doc_info, answer_output,
                research_papers_output, youtube_videos_output, web_resources_output
            ]
        )

        clear_btn.click(
            orchestrator.clear_session,
            outputs=[
                summary_output, flashcard_question, flashcard_answer, flashcard_progress,
                flashcard_answer_group, show_answer_btn, quiz_question, quiz_options,
                quiz_feedback, quiz_progress, quiz_nav_group, prev_quiz_btn,
                next_quiz_btn, submit_answer_btn, doc_info, answer_output,
                research_papers_output, youtube_videos_output, web_resources_output
            ]
        )

        # Flashcard events
        show_answer_btn.click(
            orchestrator.show_flashcard_answer,
            outputs=[flashcard_answer, flashcard_answer_group, show_answer_btn]
        )

        next_flashcard_btn.click(
            orchestrator.next_flashcard,
            outputs=[flashcard_question, flashcard_answer, flashcard_progress, 
                    flashcard_answer_group, show_answer_btn]
        )

        prev_flashcard_btn.click(
            orchestrator.prev_flashcard,
            outputs=[flashcard_question, flashcard_answer, flashcard_progress,
                    flashcard_answer_group, show_answer_btn]
        )

        # Quiz events
        submit_answer_btn.click(
            orchestrator.submit_quiz_answer,
            inputs=[quiz_options],
            outputs=[quiz_feedback, quiz_progress, next_quiz_btn, submit_answer_btn]
        )

        next_quiz_btn.click(
            orchestrator.next_quiz_question,
            outputs=[quiz_question, quiz_options, quiz_feedback, quiz_progress,
                    quiz_nav_group, prev_quiz_btn, next_quiz_btn, submit_answer_btn]
        )

        prev_quiz_btn.click(
            orchestrator.prev_quiz_question,
            outputs=[quiz_question, quiz_options, quiz_feedback, quiz_progress,
                    quiz_nav_group, prev_quiz_btn, next_quiz_btn, submit_answer_btn]
        )

        # Enhanced Q&A events
        question_input.submit(
            orchestrator.handle_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

        ask_btn.click(
            orchestrator.handle_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

    return demo

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ WARNING: No OpenAI API key found!")
        print("💡 Please set OPENAI_API_KEY in your .env file")
        print("Example .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\n🚀 Starting anyway - you can test the interface...")
    
    try:
        demo = create_interface()
        
        print("\n🎓 AI Study Assistant Pro - Enhanced Discovery Edition")
        print("📱 Interface will be available at: http://localhost:7860")
        print("\n✨ ENHANCED Features:")
        print("   ✅ Interactive flashcards with hints, categories, and difficulty levels")
        print("   ✅ Adaptive quiz system with navigation, scoring, and detailed feedback")
        print("   ✅ Smart Q&A with enhanced document context and topic awareness")
        print("   🌟 Advanced research paper discovery with relevance scoring")
        print("   🔍 Enhanced keyword extraction with academic focus")
        print("   🎥 Curated YouTube educational content from top channels")
        print("   🌐 Comprehensive web learning resources from verified platforms")
        print("   📊 Detailed processing status with extraction method reporting")
        print("   💡 Enhanced AI-generated content with improved quality")
        print("   🎯 Better error handling and fallback mechanisms")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("Please check your OpenAI API key and dependencies.")