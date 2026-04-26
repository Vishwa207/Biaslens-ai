from __future__ import annotations

import json
import logging
import os
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

LOGGER = logging.getLogger(__name__)

class GeminiAnalysisSchema(BaseModel):
    bias: str = Field(description="The primary cognitive bias detected, e.g., 'Overgeneralization', 'Catastrophizing', 'Black-and-White Thinking', 'Mind Reading', 'Personalization', 'Emotional Reasoning', or 'Balanced Thinking'")
    explanation: str = Field(description="A compassionate, brief explanation of why this bias is present in the thought.")
    balanced_thought: str = Field(description="A balanced, rational rewrite of the original thought.")
    suggestion: str = Field(description="One practical, constructive next step the user can take.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 that this is the correct bias.")
    matched_terms: list[str] = Field(description="Specific words or phrases from the thought that indicate the bias.")
    sentiment_bucket: str = Field(description="Either 'Negative Thinking' or 'Balanced Thinking'")

class GeminiChatSchema(BaseModel):
    content: str = Field(description="A conversational, compassionate reply to the user's thought. Do not repeat the balanced thought or suggestion here, keep it conversational.")
    bias: str = Field(description="The cognitive bias detected.")
    explanation: str = Field(description="A brief explanation of why this bias was detected.")
    balanced_reframe: str = Field(description="A balanced rewrite of the thought.")
    suggestion: str = Field(description="A practical suggestion.")
    follow_up_question: str = Field(description="An engaging follow-up question to keep the conversation going.")
    keywords: list[str] = Field(description="Key phrases from the user's thought.")

class GeminiEngine:
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.is_ready = HAS_GENAI and bool(self.api_key)
        
        if self.is_ready:
            self.client = genai.Client(api_key=self.api_key)
            LOGGER.info("GeminiEngine initialized with model %s", self.model_name)
        else:
            LOGGER.warning("GeminiEngine is not ready. Missing google-genai or GEMINI_API_KEY.")

    def analyze_thought(self, text: str) -> dict | None:
        if not self.is_ready:
            return None
        
        prompt = f"""
        Act as an empathetic Cognitive Behavioral Therapy (CBT) expert.
        Analyze the following thought for cognitive biases.
        
        Thought: "{text}"
        
        Provide the analysis structured exactly as requested.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiAnalysisSchema,
                    temperature=0.3,
                )
            )
            return response.parsed.model_dump()
        except Exception as e:
            LOGGER.error("Gemini analyze_thought failed: %s", e)
            return None

    def generate_chat_reply(self, text: str, bias: str, recent_messages: list[dict]) -> dict | None:
        if not self.is_ready:
            return None
            
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages[-4:]])
        
        prompt = f"""
        Act as an empathetic Cognitive Behavioral Therapy (CBT) expert chatting with a user.
        
        Recent Conversation:
        {history_text}
        
        User's latest thought: "{text}"
        Detected bias: {bias}
        
        Generate a conversational reply that feels human and helpful. Return the response structured as JSON.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiChatSchema,
                    temperature=0.7,
                )
            )
            return response.parsed.model_dump()
        except Exception as e:
            LOGGER.error("Gemini generate_chat_reply failed: %s", e)
            return None
