from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import random
import re
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .utils import preprocess_text
from .llm import GeminiEngine

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "cognitive_bias_dataset.json"
ARTIFACT_PATH = ROOT_DIR / "model" / "baseline_metrics.json"

NEGATIVE_THINKING = "Negative Thinking"
BALANCED_THINKING = "Balanced Thinking"

EXAMPLE_BANK = {
    "Overgeneralization": [
        "I always fail in everything I try.",
        "Nothing ever works out for me.",
        "Every interview goes badly, so I will never get hired.",
    ],
    "Catastrophizing": [
        "If I make one mistake, my whole future is ruined.",
        "Missing this email means the entire project will collapse.",
        "If I speak up and stumble, it will be a disaster.",
    ],
    "Black-and-White Thinking": [
        "If this is not perfect, then it is a total failure.",
        "Either I do this flawlessly or there is no point trying.",
        "If they do not fully agree with me, they are completely against me.",
    ],
    "Emotional Reasoning": [
        "I feel anxious, so this must be dangerous.",
        "I feel guilty, which means I definitely did something wrong.",
        "I feel like a failure, so I probably am one.",
    ],
    "Mind Reading": [
        "Everyone thinks I am embarrassing.",
        "They probably think I have no idea what I am doing.",
        "My team must think I am not good enough for this role.",
    ],
    "Personalization": [
        "The meeting went badly because of me.",
        "My friend seems quiet today, so I must have upset them.",
        "The team's delay is all my fault.",
    ],
}

EXPLANATION_RULES = {
    "Overgeneralization": {
        "keywords": ["always", "never", "everyone", "nobody", "nothing", "everything", "ever"],
        "explanation": "Absolute language like '{token}' suggests one difficult moment is being stretched across every situation.",
    },
    "Catastrophizing": {
        "keywords": ["ruined", "disaster", "worst", "collapse", "over", "finished", "fall apart"],
        "explanation": "The wording jumps toward the worst-case outcome through high-intensity terms like '{token}'.",
    },
    "Black-and-White Thinking": {
        "keywords": ["perfect", "failure", "completely", "either", "worthless", "totally", "flawless"],
        "explanation": "All-or-nothing framing around '{token}' leaves little room for progress, nuance, or partial success.",
    },
    "Emotional Reasoning": {
        "keywords": ["feel", "feels", "anxious", "afraid", "ashamed", "guilty", "hopeless"],
        "explanation": "Emotion-heavy language like '{token}' suggests the feeling itself is being treated as proof.",
    },
    "Mind Reading": {
        "keywords": ["they think", "everyone thinks", "judge", "hate me", "embarrassing", "laughing at me", "probably think"],
        "explanation": "The sentence assumes other people are thinking something negative without clear evidence.",
    },
    "Personalization": {
        "keywords": ["my fault", "because of me", "i ruined", "blame me", "all on me", "upset them"],
        "explanation": "The statement places too much responsibility on the speaker, especially around '{token}'.",
    },
    BALANCED_THINKING: {
        "keywords": ["learn", "improve", "progress", "next time", "can handle", "not define"],
        "explanation": "The language stays specific and solution-focused instead of overextending one difficult event.",
    },
}

GLOBAL_HIGHLIGHT_TERMS = [
    "always",
    "never",
    "nothing",
    "everything",
    "everyone",
    "nobody",
    "ruined",
    "disaster",
    "perfect",
    "failure",
    "completely",
    "totally",
    "my fault",
    "because of me",
]

SOFTENING_MAP = {
    "always": "sometimes",
    "never": "not always",
    "nothing": "some things",
    "everything": "this situation",
    "everyone": "some people",
    "nobody": "not everyone",
    "completely": "partly",
    "totally": "not entirely",
}

HIGHLIGHT_EXPLANATIONS = {
    "always": "An absolute word that turns one difficult moment into a permanent pattern.",
    "never": "A final-sounding word that makes change or improvement feel impossible.",
    "nothing": "A totalizing word that erases exceptions and small wins.",
    "everything": "Language that makes the setback sound bigger than the specific situation.",
    "everyone": "A broad social assumption that stretches one fear across all people.",
    "nobody": "A sweeping rejection cue that can make the situation sound universally negative.",
    "ruined": "A high-intensity word that pushes the mind toward a worst-case conclusion.",
    "disaster": "A catastrophic cue that increases the perceived threat level of the situation.",
    "collapse": "A severe outcome word that makes the future sound more unstable than the evidence may support.",
    "worst": "Language that pulls attention toward the most extreme possible outcome.",
    "perfect": "A rigid standard that leaves very little room for normal progress or imperfection.",
    "failure": "A pass-or-fail label that can flatten a mixed experience into a total judgment.",
    "completely": "An all-or-nothing amplifier that removes nuance from the thought.",
    "totally": "An intensity word that makes the conclusion sound more absolute.",
    "feel": "An emotion marker that may signal the feeling is being treated as proof.",
    "feels": "An emotion marker that may signal the feeling is being treated as proof.",
    "anxious": "A strong emotional cue that can make the thought feel true because it feels intense.",
    "guilty": "A self-judging emotion that can blur the line between feeling bad and being at fault.",
    "ashamed": "A self-critical emotion that can make the conclusion harsher than the facts justify.",
    "hopeless": "A final-sounding word that can make change feel unavailable.",
    "judge": "A cue that suggests assumed criticism from other people.",
    "they think": "A mind-reading phrase that assumes other people's thoughts without direct evidence.",
    "everyone thinks": "A broad mind-reading phrase that treats fear as social fact.",
    "probably think": "An assumption-based phrase that fills in what others believe without confirmation.",
    "my fault": "A self-blaming phrase that can take on more responsibility than is realistic.",
    "because of me": "A personalization cue that places too much of the outcome on one person.",
    "all on me": "A phrase that frames a shared situation as entirely your responsibility.",
    "blame me": "A self-directed blame cue that can exaggerate personal responsibility.",
}

ASSISTANT_KEYWORD_KEEP = {
    "always",
    "never",
    "everyone",
    "nobody",
    "nothing",
    "everything",
    "fail",
    "failure",
    "ruined",
    "disaster",
    "perfect",
    "worthless",
    "guilty",
    "anxious",
    "afraid",
    "ashamed",
    "judge",
    "fault",
    "blame",
    "wrong",
    "hopeless",
}

ASSISTANT_GENERIC_TOKENS = {
    "make",
    "made",
    "really",
    "thing",
    "things",
    "just",
    "today",
    "went",
    "getting",
    "gets",
    "maybe",
}

NEGATIVE_SENTIMENT_CUES = {
    "fail",
    "failed",
    "failure",
    "ruined",
    "disaster",
    "collapse",
    "worst",
    "hopeless",
    "afraid",
    "anxious",
    "guilty",
    "ashamed",
    "embarrassing",
    "hate",
    "fault",
    "blame",
    "wrong",
    "never",
    "nothing",
    "nobody",
}

POSITIVE_SENTIMENT_CUES = {
    "can",
    "handle",
    "learn",
    "improve",
    "progress",
    "better",
    "try",
    "trying",
    "help",
    "calm",
    "cope",
    "manage",
    "specific",
    "realistic",
    "next",
}

INTENSE_NEGATIVE_CUES = {"ruined", "disaster", "collapse", "hopeless", "worthless", "worst", "finished"}

SENTIMENT_REFLECTIONS = {
    "highly_negative": [
        "The tone feels very intense right now, which can make the thought sound more final than the evidence supports.",
        "There is a lot of emotional pressure in the wording, so the conclusion may feel harsher and more certain than it really is.",
        "The language carries a heavy worst-case tone, which often amplifies the distortion.",
    ],
    "negative": [
        "The tone is pretty hard on you, which can make the conclusion feel broader than it really is.",
        "The wording sounds discouraged, and that often pulls the mind toward stronger conclusions than the facts alone support.",
        "There is a negative emotional charge in the sentence, which can make the thought feel more convincing than it is.",
    ],
    "reflective": [
        "The good sign is that the tone is reflective enough that we can slow it down and examine it.",
        "Even though the thought is uncomfortable, the tone suggests there is room to question it.",
        "This sounds reflective rather than impulsive, which gives us a good starting point for reframing it.",
    ],
    "constructive": [
        "The tone already has some balance in it, which makes this easier to work with.",
        "There is some constructive energy in the wording, even if part of the thought still needs tightening.",
        "The sentence is not purely negative, so there is already a useful foundation to build on.",
    ],
    "neutral": [
        "The tone is fairly matter-of-fact, so the main signal is in the wording itself.",
        "The sentence is relatively even in tone, which helps us focus on the thinking pattern directly.",
        "The emotion is not overwhelming here, so the cognitive pattern stands out more clearly.",
    ],
}

CHAT_RESPONSE_LIBRARY = {
    "Overgeneralization": {
        "keyword_explanations": [
            "The word '{keyword}' makes this sound broader and more permanent than one difficult moment probably is.",
            "Using '{keyword}' turns a painful experience into a rule about your whole situation, which is a classic overgeneralization move.",
            "The phrasing around '{keyword}' stretches this thought beyond the specific evidence in front of you.",
        ],
        "fallback_explanations": [
            "This thought sounds like it is taking one setback and projecting it across a much bigger pattern.",
            "I'm hearing a broad conclusion here instead of a narrow description of what happened this time.",
            "The statement reads as if one hard moment is being used to judge the whole picture.",
        ],
        "suggestions": [
            "Try shrinking the statement to one recent example so it feels accurate instead of absolute.",
            "Replace the sweeping language with a specific description of what happened in this one situation.",
            "Ground the thought in time and context by naming when, where, and how often this actually happened.",
        ],
        "questions": [
            "Can you think of one situation where this was not true?",
            "What evidence shows this happens every time, rather than in only some moments?",
            "If you removed the absolute wording, how would you describe this more precisely?",
        ],
    },
    "Catastrophizing": {
        "keyword_explanations": [
            "The term '{keyword}' pushes the thought toward the worst possible outcome before the facts have fully played out.",
            "Language like '{keyword}' can make the mind treat a setback as a disaster instead of a problem to solve.",
            "The word '{keyword}' makes the future sound more extreme and dangerous than it may actually be.",
        ],
        "fallback_explanations": [
            "This thought seems to jump quickly from a setback to a worst-case ending.",
            "I'm hearing the mind race ahead to disaster before we have enough evidence for that conclusion.",
            "The statement sounds like it is escalating the consequences faster than the facts require.",
        ],
        "suggestions": [
            "Name the most likely outcome alongside the worst-case one so the picture becomes more realistic.",
            "Focus on the next practical step you can take instead of the full chain of feared outcomes.",
            "Break the situation into smaller risks so it feels solvable rather than catastrophic.",
        ],
        "questions": [
            "What is the most likely outcome here, not just the worst one?",
            "If this went imperfectly, what would probably happen next in real life?",
            "What practical step would reduce the risk you are imagining most?",
        ],
    },
    "Black-and-White Thinking": {
        "keyword_explanations": [
            "The cue '{keyword}' makes the situation sound all-good or all-bad, with very little room in between.",
            "Words like '{keyword}' often flatten nuance and push the mind into an all-or-nothing frame.",
            "The phrasing around '{keyword}' suggests the thought is treating anything less than ideal as failure.",
        ],
        "fallback_explanations": [
            "This sounds like an all-or-nothing judgment instead of a nuanced read of the situation.",
            "The thought leaves little room for partial success, mixed outcomes, or gradual progress.",
            "I'm hearing a rigid either-or frame where the situation is probably more mixed than that.",
        ],
        "suggestions": [
            "Look for the middle ground and name what is acceptable even if it is not perfect.",
            "Separate 'not ideal' from 'total failure' so the situation has more nuance.",
            "Try rating the outcome on a scale instead of forcing it into success-or-failure language.",
        ],
        "questions": [
            "What would the middle ground look like here?",
            "If this were 60 percent good instead of perfect, what would still count as progress?",
            "What part of this situation worked, even if another part did not?",
        ],
    },
    "Emotional Reasoning": {
        "keyword_explanations": [
            "The term '{keyword}' suggests the feeling is being treated as proof, rather than one source of information.",
            "Language like '{keyword}' can blur the line between 'I feel this' and 'this must be true.'",
            "The phrase '{keyword}' points to emotion carrying more weight than direct evidence right now.",
        ],
        "fallback_explanations": [
            "This thought seems to treat the emotional experience as the verdict instead of checking it against facts.",
            "I'm hearing the feeling and the conclusion get fused together, which is common in emotional reasoning.",
            "The statement sounds like the emotion is driving the belief more than outside evidence is.",
        ],
        "suggestions": [
            "Acknowledge the feeling first, then list the facts that support it and the facts that do not.",
            "Try separating 'I feel unsafe or upset' from 'this situation is definitely unsafe or hopeless.'",
            "Pause long enough to ask what the evidence says in addition to what the feeling says.",
        ],
        "questions": [
            "What facts support this feeling, and what facts point another way?",
            "If a friend felt this, what evidence would you want them to check?",
            "What part of this thought is emotion, and what part is confirmed fact?",
        ],
    },
    "Mind Reading": {
        "keyword_explanations": [
            "The cue '{keyword}' suggests you may be filling in other people's thoughts without direct evidence.",
            "Language around '{keyword}' makes it sound like their internal judgment is already known, even though it has not been confirmed.",
            "The phrase '{keyword}' points to an assumption about what others think or feel.",
        ],
        "fallback_explanations": [
            "This thought seems to assume other people's judgments without clear proof.",
            "I'm hearing an interpretation of what others think, rather than evidence of what they have actually said or done.",
            "The statement sounds like it is guessing at someone else's internal view of you.",
        ],
        "suggestions": [
            "Shift from assumption to evidence by noticing what they actually said, did, or confirmed.",
            "Leave room for multiple interpretations until you have direct feedback.",
            "If the relationship matters, consider asking a clarifying question instead of predicting their judgment.",
        ],
        "questions": [
            "What direct evidence do you have for what they think?",
            "What are two other explanations for their behavior besides the one your mind picked first?",
            "If you asked for feedback, what do you realistically expect they would say?",
        ],
    },
    "Personalization": {
        "keyword_explanations": [
            "The phrase '{keyword}' suggests you may be taking on more responsibility than is actually yours.",
            "Language like '{keyword}' can make a shared situation feel as if it rests entirely on you.",
            "The cue '{keyword}' points to self-blame that may be larger than the evidence supports.",
        ],
        "fallback_explanations": [
            "This thought sounds like it is placing too much of the outcome on your shoulders alone.",
            "I'm hearing strong self-blame in a situation that likely has multiple causes.",
            "The statement seems to assign you responsibility for more of the situation than you can fully control.",
        ],
        "suggestions": [
            "Separate what was under your control from what belonged to other people, timing, or circumstances.",
            "Try naming your actual contribution without turning it into total responsibility.",
            "Replace global self-blame with a more specific statement about what you did and did not influence.",
        ],
        "questions": [
            "Which parts of this were actually under your control?",
            "What other factors played a role besides your actions?",
            "If someone else described this story, would you hold them 100 percent responsible?",
        ],
    },
    BALANCED_THINKING: {
        "keyword_explanations": [
            "The wording around '{keyword}' stays fairly grounded and specific, which is why this reads as more balanced thinking.",
            "The phrase '{keyword}' suggests you are already leaving room for nuance instead of jumping to extremes.",
            "Using '{keyword}' helps keep the thought practical rather than distorted.",
        ],
        "fallback_explanations": [
            "This thought is relatively grounded and specific instead of overreaching.",
            "I'm hearing a more balanced pattern here, with less distortion and more realism.",
            "The sentence stays closer to facts and next steps than to sweeping conclusions.",
        ],
        "suggestions": [
            "Keep building on that specificity and use it to decide the next helpful action.",
            "You can strengthen this thought even more by pairing it with one concrete next step.",
            "Hold on to the balanced part of this thought and turn it into a small plan.",
        ],
        "questions": [
            "What helped you keep this thought grounded?",
            "What next step would match the balanced tone you already have here?",
            "How could you keep this same level of nuance the next time stress spikes?",
        ],
    },
}


@dataclass(slots=True)
class PredictionResult:
    bias: str
    confidence: float
    confidence_percent: str
    thinking_score: int
    thinking_band: str
    thinking_tone: str
    explanation: str
    balanced_thought: str
    suggestion: str
    matched_terms: list[str]
    highlight_terms: list[str]
    highlight_details: list[dict[str, str]]
    sentiment_bucket: str
    alternative_biases: list[dict[str, Any]]
    processed_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatReply:
    content: str
    bias: str
    explanation: str
    balanced_reframe: str
    suggestion: str
    follow_up_question: str
    keywords: list[str]
    sentiment_tone: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OptionalTransformerOption:
    """Optional transformer upgrade path that is loaded only when available."""

    available = False

    def __init__(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self.pipeline = pipeline
            self.available = True
        except Exception:
            self.pipeline = None
            self.available = False


class RuleBasedExplainer:
    def explain(self, original_text: str, bias: str) -> tuple[str, list[str]]:
        rules = EXPLANATION_RULES.get(bias, EXPLANATION_RULES[BALANCED_THINKING])
        matches = self._ordered_matches(original_text, rules["keywords"])

        if matches:
            return rules["explanation"].format(token=matches[0]), matches[:4]

        if bias == BALANCED_THINKING:
            return "The sentence stays specific and measured without strong distortion markers.", matches

        return f"The sentence contains linguistic patterns that align with {bias.lower()}.", matches

    @staticmethod
    def _ordered_matches(text: str, candidates: list[str]) -> list[str]:
        lower_text = text.lower()
        matches = [candidate for candidate in candidates if candidate in lower_text]
        return sorted(dict.fromkeys(matches), key=lambda candidate: lower_text.find(candidate))


class BalancedRewriter:
    def rewrite(self, original_text: str, bias: str, highlight_terms: list[str]) -> str:
        cleaned_text = self._normalize_text(original_text)
        lower_text = cleaned_text.lower()
        softened_text = self._soften_absolutes(cleaned_text)

        if bias == "Overgeneralization":
            return self._rewrite_overgeneralization(lower_text, softened_text)
        if bias == "Catastrophizing":
            return self._rewrite_catastrophizing(lower_text)
        if bias == "Black-and-White Thinking":
            return self._rewrite_black_and_white(lower_text)
        if bias == "Emotional Reasoning":
            return self._rewrite_emotional_reasoning(lower_text)
        if bias == "Mind Reading":
            return self._rewrite_mind_reading(lower_text)
        if bias == "Personalization":
            return self._rewrite_personalization(lower_text)
        if bias == BALANCED_THINKING:
            return self._rewrite_balanced(highlight_terms)

        if highlight_terms:
            return "This thought feels strong right now, but it may be more accurate to slow it down and describe the situation more specifically."
        return f"{softened_text} I can stay specific, check the evidence, and focus on what is actually true right now."

    def _rewrite_overgeneralization(self, lower_text: str, softened_text: str) -> str:
        if any(token in lower_text for token in ("fail", "failed", "failure", "mess up", "messed up", "wrong")):
            return random.choice([
                "I struggled in this specific instance, but it doesn't define my overall ability. I can extract a lesson from this.",
                "I may have struggled here, but that does not mean I always fail. I can learn from it and improve with practice.",
                "This attempt didn't go perfectly, but one setback isn't a permanent pattern. Next time, I can try a different approach."
            ])
        if any(token in lower_text for token in ("nothing", "never", "ever")):
            return random.choice([
                "This feels discouraging right now, but not everything will go badly. One hard moment does not dictate every outcome.",
                "It feels like nothing works right now, but I know that's not factually true. There are exceptions I can focus on.",
                "While this specific situation is frustrating, saying 'never' ignores the times things have actually gone well."
            ])
        if any(token in lower_text for token in ("everyone", "nobody", "everything")):
            return random.choice([
                "This is one part of the picture, not the whole story. I can look for specific evidence instead of making it universal.",
                "It's easy to assume everyone feels a certain way, but I only have evidence for this specific moment.",
                "I am stretching one negative experience to cover everything. Let me scale it back to just the facts of this situation."
            ])
        return f"{softened_text} This is one moment, not a permanent pattern, and I can respond constructively from here."

    @staticmethod
    def _rewrite_catastrophizing(lower_text: str) -> str:
        if any(token in lower_text for token in ("ruined", "disaster", "collapse", "worst", "finished", "over")):
            return random.choice([
                "This situation feels serious, but it is probably not a disaster. I can look at the most likely outcome, and take a practical step.",
                "I'm assuming the absolute worst-case scenario. Realistically, there are many ways this could play out that are manageable.",
                "Even if things go poorly, I have handled difficult situations before. This is a problem to solve, not a catastrophe."
            ])
        return random.choice([
            "This feels intense right now, but I do not have to jump to the worst-case outcome. I can focus on what is most likely.",
            "My mind is racing ahead to danger. Let me pause and ask what is happening right now, in this exact moment."
        ])

    @staticmethod
    def _rewrite_black_and_white(lower_text: str) -> str:
        if any(token in lower_text for token in ("perfect", "failure", "worthless", "flawless")):
            return random.choice([
                "It does not have to be perfect to have value. A mixed or imperfect result can still be meaningful progress.",
                "I'm holding myself to an impossible standard. Doing it 'okay' is still better than not doing it at all.",
                "There is a lot of space between perfect and worthless. What is the middle ground here?"
            ])
        return random.choice([
            "This situation is probably not all good or all bad. I can make room for the middle ground.",
            "I'm seeing this in extremes. What would a more balanced, realistic view look like?"
        ])

    @staticmethod
    def _rewrite_emotional_reasoning(lower_text: str) -> str:
        if any(token in lower_text for token in ("anxious", "afraid", "guilty", "ashamed", "hopeless")):
            return random.choice([
                "My feelings are real, but they are not the whole truth. I can acknowledge the emotion and still check the facts.",
                "I feel bad right now, but that doesn't mean the situation is inherently bad. Feelings are just signals, not facts.",
                "It's okay to feel this way. However, I shouldn't let the intensity of the feeling dictate my entire conclusion."
            ])
        return "How I feel matters, but feelings alone do not prove the conclusion. I can pause and look at the evidence before deciding what this means."

    @staticmethod
    def _rewrite_mind_reading(lower_text: str) -> str:
        if any(token in lower_text for token in ("they think", "everyone thinks", "probably think", "judge", "hate me")):
            return random.choice([
                "I do not know what others think yet. I can look for evidence or ask for feedback instead of assuming the worst.",
                "I am projecting my own insecurities onto other people. I can't read minds.",
                "People are usually too focused on themselves to judge me as harshly as I am judging myself right now."
            ])
        return "I may be guessing what other people think, but I do not know that for sure. It would be fairer to stay open to other explanations."

    @staticmethod
    def _rewrite_personalization(lower_text: str) -> str:
        if any(token in lower_text for token in ("my fault", "because of me", "all on me", "blame me", "i ruined")):
            return random.choice([
                "I may have influenced part of this situation, but I am not responsible for everything. Other people and circumstances matter too.",
                "I'm taking all the blame for a complex situation. Let me realistically divide the responsibility.",
                "It's easy to blame myself, but there are external factors here that were completely out of my control."
            ])
        return "I can own my part without taking ownership of the entire outcome. This situation likely has more than one cause."

    @staticmethod
    def _rewrite_balanced(highlight_terms: list[str]) -> str:
        if highlight_terms:
            return "This thought already has some balance. Keeping it specific and grounded will make it even more helpful."
        return "This thought is already fairly balanced. I can keep focusing on specific facts and the next helpful step."

    @staticmethod
    def _soften_absolutes(text: str) -> str:
        adjusted = text
        for source, target in SOFTENING_MAP.items():
            adjusted = re.sub(rf"\b{re.escape(source)}\b", target, adjusted, flags=re.IGNORECASE)
        adjusted = re.sub(r"\s+", " ", adjusted).strip().rstrip(".!?")
        if not adjusted:
            return "This situation feels difficult right now."
        return adjusted[0].upper() + adjusted[1:] + "."

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized or "This situation feels difficult right now."


class SuggestionEngine:
    def suggest(self, original_text: str, bias: str, highlight_terms: list[str]) -> str:
        if bias == "Overgeneralization":
            return "Name one specific example and one exception so the thought becomes more accurate instead of absolute."
        if bias == "Catastrophizing":
            return "Write down the most likely outcome and the first practical step you can take if things become difficult."
        if bias == "Black-and-White Thinking":
            return "Rate the situation on a scale instead of calling it a total success or failure."
        if bias == "Emotional Reasoning":
            return "Separate what you feel from what you know by listing the evidence for and against the thought."
        if bias == "Mind Reading":
            return "Look for direct evidence or ask a clarifying question before deciding what someone else thinks."
        if bias == "Personalization":
            return "List what was under your control and what belonged to other people or circumstances."
        if highlight_terms:
            return f"Slow down around the word '{highlight_terms[0]}' and restate the thought in more specific, testable language."
        return "Keep the thought specific and pair it with one concrete next action you can take."


class AssistantResponseEngine:
    def build_reply(
        self,
        text: str,
        prediction: PredictionResult,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> ChatReply:
        keywords = self.extract_keywords(text, prediction)
        sentiment_tone = self.detect_sentiment_tone(text, prediction, keywords)
        recent_assistant_text = [
            item.get("content", "")
            for item in (recent_messages or [])
            if item.get("role") == "assistant" and item.get("content")
        ]

        explanation = self._build_explanation(prediction, keywords, sentiment_tone, recent_assistant_text)
        balanced_reframe = self._build_balanced_reframe(prediction, recent_assistant_text)
        suggestion = self._build_suggestion(prediction, keywords, recent_assistant_text)
        follow_up_question = self._build_follow_up(prediction, keywords, recent_assistant_text)
        content = self._build_content(
            prediction,
            explanation,
            balanced_reframe,
            suggestion,
            follow_up_question,
            recent_assistant_text,
        )

        return ChatReply(
            content=content,
            bias=prediction.bias,
            explanation=explanation,
            balanced_reframe=balanced_reframe,
            suggestion=suggestion,
            follow_up_question=follow_up_question,
            keywords=keywords,
            sentiment_tone=sentiment_tone,
        )

    def extract_keywords(self, text: str, prediction: PredictionResult, *, limit: int = 5) -> list[str]:
        lower_text = text.lower()
        keywords: list[str] = []

        for term in [*prediction.highlight_terms, *prediction.matched_terms]:
            cleaned = term.strip().lower()
            if cleaned and cleaned in lower_text and cleaned not in keywords:
                keywords.append(cleaned)
            if len(keywords) >= limit:
                return keywords

        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        for token in tokens:
            if token in keywords:
                continue
            if len(token) < 3:
                continue
            if token in ASSISTANT_GENERIC_TOKENS:
                continue
            if any(" " in keyword and token in keyword.split() for keyword in keywords):
                continue
            if token in ENGLISH_STOP_WORDS and token not in ASSISTANT_KEYWORD_KEEP:
                continue
            keywords.append(token)
            if len(keywords) >= limit:
                return keywords

        return keywords

    def detect_sentiment_tone(self, text: str, prediction: PredictionResult, keywords: list[str]) -> str:
        lower_text = text.lower()
        tokens = re.findall(r"[a-zA-Z']+", lower_text)
        negative_hits = sum(1 for token in tokens if token in NEGATIVE_SENTIMENT_CUES)
        positive_hits = sum(1 for token in tokens if token in POSITIVE_SENTIMENT_CUES)
        intense_hits = sum(1 for token in tokens if token in INTENSE_NEGATIVE_CUES)
        reflective_hits = sum(
            1 for marker in ("maybe", "perhaps", "wonder", "could", "might", "evidence", "why") if marker in lower_text
        )
        absolute_hits = sum(1 for keyword in keywords if keyword in {"always", "never", "nothing", "everything", "everyone", "nobody"})

        if prediction.sentiment_bucket == BALANCED_THINKING and positive_hits >= negative_hits:
            return "constructive"
        if intense_hits or negative_hits + absolute_hits >= 4:
            return "highly_negative"
        if negative_hits > positive_hits:
            return "negative"
        if positive_hits > negative_hits:
            return "constructive"
        if reflective_hits or "?" in text:
            return "reflective"
        return "negative" if prediction.sentiment_bucket == NEGATIVE_THINKING else "neutral"

    def _build_explanation(
        self,
        prediction: PredictionResult,
        keywords: list[str],
        sentiment_tone: str,
        recent_assistant_text: list[str],
    ) -> str:
        library = CHAT_RESPONSE_LIBRARY.get(prediction.bias, CHAT_RESPONSE_LIBRARY[BALANCED_THINKING])
        keyword = self._primary_keyword(keywords)
        if keyword:
            explanation_pool = [template.format(keyword=keyword) for template in library["keyword_explanations"]]
        else:
            explanation_pool = list(library["fallback_explanations"])

        explanation = self._choose_variant(explanation_pool, recent_assistant_text)
        tone_reflection = self._choose_variant(SENTIMENT_REFLECTIONS[sentiment_tone], recent_assistant_text)
        if len(keywords) > 1:
            extra_keywords = self._format_keywords(keywords[1:3])
            return f"{explanation} I also noticed {extra_keywords}, which reinforces that pattern. {tone_reflection}"
        return f"{explanation} {tone_reflection}"

    def _build_suggestion(
        self,
        prediction: PredictionResult,
        keywords: list[str],
        recent_assistant_text: list[str],
    ) -> str:
        library = CHAT_RESPONSE_LIBRARY.get(prediction.bias, CHAT_RESPONSE_LIBRARY[BALANCED_THINKING])
        suggestion_pool = [*library["suggestions"], prediction.suggestion]
        suggestion = self._choose_variant(suggestion_pool, recent_assistant_text)
        keyword = self._primary_keyword(keywords)
        if keyword and prediction.bias == "Overgeneralization":
            return f"{suggestion} Replacing '{keyword}' with more specific language would make the thought more accurate."
        if keyword and prediction.bias == "Catastrophizing":
            return f"{suggestion} Notice how '{keyword}' raises the threat level in the sentence."
        return suggestion

    @staticmethod
    def _build_balanced_reframe(prediction: PredictionResult, recent_assistant_text: list[str]) -> str:
        candidates = [
            f"Here is a more balanced way to look at it: \n\n> {prediction.balanced_thought}",
            f"If we reframe that slightly, it might sound like this: \n\n> {prediction.balanced_thought}",
            f"A steadier, more grounded perspective could be: \n\n> {prediction.balanced_thought}",
            f"Try telling yourself this instead: \n\n> {prediction.balanced_thought}",
            f"Consider this alternative angle: \n\n> {prediction.balanced_thought}",
        ]
        recent_blob = " ".join(recent_assistant_text)
        fresh_candidates = [candidate for candidate in candidates if candidate not in recent_blob]
        pool = fresh_candidates or candidates
        return random.choice(pool)

    def _build_content(
        self,
        prediction: PredictionResult,
        explanation: str,
        balanced_reframe: str,
        suggestion: str,
        follow_up_question: str,
        recent_assistant_text: list[str],
    ) -> str:
        templates = [
            f"I'm noticing a pattern of **{prediction.bias}** here. {explanation}\n\n{balanced_reframe}\n\n**Next step:** {suggestion}\n\n*What do you think? {follow_up_question}*",
            f"This sounds a lot like **{prediction.bias}**. {explanation}\n\n{balanced_reframe}\n\nTo move forward, I'd suggest to {suggestion[0].lower() + suggestion[1:] if suggestion else ''}.\n\n*Let me ask you: {follow_up_question}*",
            f"Based on your wording, this leans toward **{prediction.bias}**. {explanation}\n\n{balanced_reframe}\n\n**Try this:** {suggestion}\n\n*{follow_up_question}*",
            f"I see some **{prediction.bias}** creeping in. {explanation}\n\n{balanced_reframe}\n\nHere is a practical step: {suggestion}\n\n*{follow_up_question}*",
        ]
        return self._choose_variant(templates, recent_assistant_text)

    def _build_follow_up(
        self,
        prediction: PredictionResult,
        keywords: list[str],
        recent_assistant_text: list[str],
    ) -> str:
        library = CHAT_RESPONSE_LIBRARY.get(prediction.bias, CHAT_RESPONSE_LIBRARY[BALANCED_THINKING])
        question = self._choose_variant(library["questions"], recent_assistant_text)
        keyword = self._primary_keyword(keywords)
        if keyword and prediction.bias == "Overgeneralization":
            return f"Where does the word '{keyword}' feel least true, and how would you restate the thought without it?"
        if keyword and prediction.bias == "Mind Reading":
            return f"What direct evidence supports the idea behind '{keyword}', and what evidence points to a different explanation?"
        return question

    @staticmethod
    def _primary_keyword(keywords: list[str]) -> str | None:
        return keywords[0] if keywords else None

    @staticmethod
    def _format_keywords(keywords: list[str]) -> str:
        cleaned = [f"'{keyword}'" for keyword in keywords if keyword]
        if not cleaned:
            return "other wording cues"
        if len(cleaned) == 1:
            return cleaned[0]
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    @staticmethod
    def _choose_variant(candidates: list[str], recent_assistant_text: list[str]) -> str:
        recent_blob = " ".join(recent_assistant_text)
        fresh_candidates = [candidate for candidate in candidates if candidate and candidate not in recent_blob]
        pool = fresh_candidates or [candidate for candidate in candidates if candidate]
        return random.choice(pool) if pool else ""


class CognitiveBiasModel:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or DATASET_PATH
        self.explainer = RuleBasedExplainer()
        self.rewriter = BalancedRewriter()
        self.suggester = SuggestionEngine()
        self.assistant = AssistantResponseEngine()
        self.llm_engine = GeminiEngine()
        self.transformer_option = OptionalTransformerOption()
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=preprocess_text,
                        tokenizer=str.split,
                        token_pattern=None,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
                ),
            ]
        )
        self.dataset = self._load_dataset()
        self.metrics = self._train_and_evaluate()

    def _load_dataset(self) -> list[dict[str, str]]:
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        LOGGER.info("Loaded %s labeled examples from %s", len(dataset), self.dataset_path)
        return dataset

    def _rule_bias_scores(self, text: str) -> dict[str, float]:
        lower_text = text.lower()
        scores: dict[str, float] = defaultdict(float)
        for bias, rule in EXPLANATION_RULES.items():
            for keyword in rule["keywords"]:
                if keyword in lower_text:
                    scores[bias] += 0.18
        return scores

    def _train_and_evaluate(self) -> dict[str, Any]:
        texts = [item["text"] for item in self.dataset]
        labels = [item["label"] for item in self.dataset]

        self.pipeline.fit(texts, labels)
        predictions = [self.predict(text).bias for text in texts]
        classes = [str(class_name) for class_name in self.pipeline.classes_]

        metrics = {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "precision": round(float(precision_score(labels, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(labels, predictions, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(labels, predictions, average="weighted", zero_division=0)), 4),
            "confusion_labels": classes,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
            "dataset_size": len(self.dataset),
            "bias_distribution": dict(Counter(labels)),
            "benchmark_type": "curated internal benchmark",
            "model_name": "TF-IDF + Linear SVM",
            "transformer_ready": self.transformer_option.available,
        }

        ARTIFACT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOGGER.info("Model trained and evaluation metrics stored at %s", ARTIFACT_PATH)
        return metrics

    def predict(self, text: str) -> PredictionResult:
        if self.llm_engine.is_ready:
            llm_result = self.llm_engine.analyze_thought(text)
            if llm_result:
                confidence = float(llm_result.get("confidence", 0.85))
                sentiment_bucket = llm_result.get("sentiment_bucket", BALANCED_THINKING)
                highlight_terms = llm_result.get("matched_terms", [])
                thinking_score = self._thinking_score(confidence, sentiment_bucket, highlight_terms)
                
                return PredictionResult(
                    bias=llm_result.get("bias", BALANCED_THINKING),
                    confidence=round(confidence, 4),
                    confidence_percent=f"{round(confidence * 100, 1)}%",
                    thinking_score=thinking_score,
                    thinking_band=self._thinking_band(thinking_score),
                    thinking_tone=self._thinking_tone(thinking_score),
                    explanation=llm_result.get("explanation", ""),
                    balanced_thought=llm_result.get("balanced_thought", ""),
                    suggestion=llm_result.get("suggestion", ""),
                    matched_terms=highlight_terms,
                    highlight_terms=highlight_terms,
                    highlight_details=self._highlight_details(llm_result.get("bias", ""), highlight_terms),
                    sentiment_bucket=sentiment_bucket,
                    alternative_biases=[],
                    processed_text=preprocess_text(text),
                )
                
        probabilities = self.pipeline.predict_proba([text])[0]
        class_names = list(self.pipeline.classes_)
        adjusted_scores = {str(class_name): float(score) for class_name, score in zip(class_names, probabilities)}

        for bias, bonus in self._rule_bias_scores(text).items():
            if bias in adjusted_scores:
                adjusted_scores[bias] += bonus

        ranked = sorted(
            (
                {
                    "bias": str(class_name),
                    "confidence": round(float(score), 4),
                }
                for class_name, score in adjusted_scores.items()
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        winning_label = ranked[0]["bias"]
        confidence = min(0.99, ranked[0]["confidence"])
        explanation, matched_terms = self.explainer.explain(text, winning_label)
        highlight_terms = self._highlight_terms(text, winning_label, matched_terms)
        highlight_details = self._highlight_details(winning_label, highlight_terms)
        balanced_thought = self.rewriter.rewrite(text, winning_label, highlight_terms)
        suggestion = self.suggester.suggest(text, winning_label, highlight_terms)
        sentiment_bucket = BALANCED_THINKING if winning_label == BALANCED_THINKING else NEGATIVE_THINKING
        thinking_score = self._thinking_score(confidence, sentiment_bucket, highlight_terms)
        thinking_band = self._thinking_band(thinking_score)

        return PredictionResult(
            bias=winning_label,
            confidence=round(confidence, 4),
            confidence_percent=f"{round(confidence * 100, 1)}%",
            thinking_score=thinking_score,
            thinking_band=thinking_band,
            thinking_tone=self._thinking_tone(thinking_score),
            explanation=explanation,
            balanced_thought=balanced_thought,
            suggestion=suggestion,
            matched_terms=matched_terms,
            highlight_terms=highlight_terms,
            highlight_details=highlight_details,
            sentiment_bucket=sentiment_bucket,
            alternative_biases=ranked[1:4],
            processed_text=preprocess_text(text),
        )

    def build_chat_reply(
        self,
        text: str,
        prediction: PredictionResult,
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> ChatReply:
        if self.llm_engine.is_ready:
            llm_chat = self.llm_engine.generate_chat_reply(text, prediction.bias, recent_messages or [])
            if llm_chat:
                return ChatReply(
                    content=llm_chat.get("content", ""),
                    bias=llm_chat.get("bias", prediction.bias),
                    explanation=llm_chat.get("explanation", ""),
                    balanced_reframe=llm_chat.get("balanced_reframe", ""),
                    suggestion=llm_chat.get("suggestion", ""),
                    follow_up_question=llm_chat.get("follow_up_question", ""),
                    keywords=llm_chat.get("keywords", []),
                )
        return self.assistant.build_reply(text, prediction, recent_messages)

    def analytics_snapshot(
        self,
        history: list[dict[str, Any]],
        saved_items: list[dict[str, Any]],
        chat_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        bias_counter = Counter(item["bias"] for item in history)
        sentiment_counter = Counter(item["sentiment_bucket"] for item in history)
        trigger_counter = Counter(term for item in history for term in item.get("highlight_terms", []))
        balanced_count = sentiment_counter.get(BALANCED_THINKING, 0)
        negative_count = max(0, len(history) - balanced_count)
        avg_confidence = round(
            (sum(float(item.get("confidence", 0.0)) for item in history) / len(history)) * 100,
            1,
        ) if history else 0.0
        avg_thinking_score = round(
            sum(int(item.get("thinking_score", 0)) for item in history) / len(history),
            1,
        ) if history else 0.0
        most_common_bias = bias_counter.most_common(1)[0][0] if bias_counter else "No analyses yet"
        most_used_word = trigger_counter.most_common(1)[0][0] if trigger_counter else "No clear trigger yet"
        risk_score = self._risk_score(len(history), balanced_count, avg_confidence, most_common_bias)
        risk_level = self._risk_level(risk_score)
        weekly_progress = self._weekly_progress(history)

        usage_statistics = [
            {"label": "Analyses", "value": len(history)},
            {"label": "Saved Drafts", "value": len(saved_items)},
            {"label": "Chat Turns", "value": len(chat_messages or [])},
            {"label": "Avg Confidence", "value": avg_confidence},
        ]

        return {
            "bias_distribution": [{"label": label, "count": count} for label, count in bias_counter.most_common()],
            "sentiment_distribution": [
                {"label": NEGATIVE_THINKING, "count": negative_count},
                {"label": BALANCED_THINKING, "count": balanced_count},
            ],
            "negative_vs_balanced": [
                {"label": NEGATIVE_THINKING, "count": negative_count},
                {"label": BALANCED_THINKING, "count": balanced_count},
            ],
            "usage_statistics": usage_statistics,
            "history_count": len(history),
            "balanced_ratio": round((balanced_count / len(history)) * 100, 1) if history else 0.0,
            "avg_confidence": avg_confidence,
            "avg_thinking_score": avg_thinking_score,
            "most_common_bias": most_common_bias,
            "most_used_word": most_used_word,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_summary": self._risk_summary(risk_score, most_common_bias),
            "trend_summary": self._trend_summary(history, most_common_bias),
            "timeline": self._timeline(history),
            "top_triggers": [{"label": label, "count": count} for label, count in trigger_counter.most_common(5)],
            "smart_insight_message": self._smart_insight_message(
                history_count=len(history),
                most_common_bias=most_common_bias,
                most_used_word=most_used_word,
                risk_level=risk_level,
            ),
            "grouped_history": self._group_history(history),
            "weekly_progress": weekly_progress,
            "evaluation": self.metrics,
        }

    def enrich_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.enrich_analysis_item(item) for item in history]

    def enrich_analysis_item(self, item: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(item)
        text = str(enriched.get("text", "") or "")
        bias = str(enriched.get("bias", BALANCED_THINKING) or BALANCED_THINKING)
        matched_terms = list(enriched.get("matched_terms") or [])
        highlight_terms = list(enriched.get("highlight_terms") or self._highlight_terms(text, bias, matched_terms))
        thinking_score = int(enriched.get("thinking_score", 0) or 0)
        confidence = float(enriched.get("confidence", 0.0) or 0.0)
        sentiment_bucket = str(enriched.get("sentiment_bucket", NEGATIVE_THINKING) or NEGATIVE_THINKING)

        if thinking_score <= 0:
            thinking_score = self._thinking_score(confidence, sentiment_bucket, highlight_terms)

        enriched["highlight_terms"] = highlight_terms
        enriched["highlight_details"] = self._highlight_details(bias, highlight_terms)
        enriched["balanced_thought"] = enriched.get("balanced_thought") or self.rewriter.rewrite(text, bias, highlight_terms)
        enriched["suggestion"] = enriched.get("suggestion") or self.suggester.suggest(text, bias, highlight_terms)
        enriched["thinking_score"] = thinking_score
        enriched["thinking_band"] = self._thinking_band(thinking_score)
        enriched["thinking_tone"] = self._thinking_tone(thinking_score)
        return enriched

    def _highlight_terms(self, text: str, bias: str, matched_terms: list[str]) -> list[str]:
        rules = EXPLANATION_RULES.get(bias, EXPLANATION_RULES[BALANCED_THINKING])
        candidates = [*matched_terms, *rules["keywords"], *GLOBAL_HIGHLIGHT_TERMS]
        lower_text = text.lower()
        found = [candidate for candidate in candidates if candidate in lower_text]
        return sorted(dict.fromkeys(found), key=lambda candidate: lower_text.find(candidate))[:6]

    def _highlight_details(self, bias: str, highlight_terms: list[str]) -> list[dict[str, str]]:
        details: list[dict[str, str]] = []
        for term in highlight_terms:
            normalized = term.lower()
            explanation = HIGHLIGHT_EXPLANATIONS.get(
                normalized,
                f"This phrase adds language that strengthens a {bias.lower()} pattern.",
            )
            details.append({"term": term, "explanation": explanation})
        return details

    def _trend_summary(self, history: list[dict[str, Any]], most_common_bias: str) -> str:
        if not history:
            return "Run your first analysis to unlock your personal thinking trend summary."

        if len(history) < 3:
            return "A few more analyses will reveal whether your thinking is becoming more balanced over time."

        recent = history[:5]
        recent_balanced = sum(1 for item in recent if item["sentiment_bucket"] == BALANCED_THINKING)
        recent_ratio = (recent_balanced / len(recent)) * 100

        if recent_ratio >= 60:
            return "Your recent entries are trending more balanced, with more nuanced and evidence-based language."

        if most_common_bias == "Overgeneralization":
            return "Overgeneralization appears most often. Watch for words like 'always' or 'never' and replace them with specifics."
        if most_common_bias == "Catastrophizing":
            return "Recent entries lean toward worst-case predictions. Try naming the next realistic step instead of the worst possible outcome."
        if most_common_bias == "Mind Reading":
            return "Mind reading is recurring. Grounding yourself in evidence or direct feedback may reduce assumption-heavy thoughts."
        if most_common_bias == "Personalization":
            return "You may be taking on too much blame in recent entries. Separate what is yours to own from what is outside your control."

        return f"Recent entries still lean toward {most_common_bias.lower()}. More specific and balanced language may improve the pattern."

    def _timeline(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: Counter[str] = Counter()
        for item in reversed(history[-10:]):
            parsed = self._parse_created_at(item)
            label = parsed.strftime("%b %d") if parsed else item.get("timestamp", "Now")
            buckets[label] += 1
        return [{"label": label, "count": count} for label, count in buckets.items()]

    def _thinking_score(self, confidence: float, sentiment_bucket: str, highlight_terms: list[str]) -> int:
        confidence_penalty = confidence * 32
        trigger_penalty = min(22, len(highlight_terms) * 5)
        sentiment_penalty = 15 if sentiment_bucket == NEGATIVE_THINKING else -8
        score = round(100 - (confidence_penalty + trigger_penalty + sentiment_penalty))
        return max(18, min(96, score))

    @staticmethod
    def _thinking_band(score: int) -> str:
        if score >= 75:
            return "Low Risk"
        if score >= 50:
            return "Medium Risk"
        return "High Risk"

    @staticmethod
    def _thinking_tone(score: int) -> str:
        if score >= 75:
            return "low"
        if score >= 50:
            return "medium"
        return "high"

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 70:
            return "High Risk"
        if score >= 45:
            return "Medium Risk"
        return "Low Risk"

    def _risk_score(self, history_count: int, balanced_count: int, avg_confidence: float, most_common_bias: str) -> int:
        if history_count == 0:
            return 18

        imbalance = 100 - round((balanced_count / history_count) * 100, 1)
        bias_penalty = {
            "Catastrophizing": 12,
            "Overgeneralization": 10,
            "Mind Reading": 9,
            "Personalization": 8,
        }.get(most_common_bias, 6)
        score = round((imbalance * 0.58) + (avg_confidence * 0.24) + bias_penalty)
        return max(12, min(94, score))

    @staticmethod
    def _risk_summary(score: int, most_common_bias: str) -> str:
        if score >= 70:
            return f"Recent patterns suggest elevated distortion risk, led by {most_common_bias.lower()}."
        if score >= 45:
            return f"Your recent thoughts show moderate risk signals, especially around {most_common_bias.lower()}."
        return "Recent patterns look relatively balanced, with lower cognitive distortion risk."

    def _group_history(self, history: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in history:
            bias = item.get("bias", "Unknown")
            current = grouped.setdefault(
                bias,
                {
                    "bias": bias,
                    "count": 0,
                    "latest_text": item.get("text", ""),
                    "latest_timestamp": item.get("timestamp", "Now"),
                    "latest_created_at": item.get("created_at", ""),
                    "thinking_band": item.get("thinking_band", "Medium Risk"),
                },
            )
            current["count"] += 1
            current_created_at = str(current.get("latest_created_at", "") or "")
            next_created_at = str(item.get("created_at", "") or "")
            if not current_created_at or next_created_at >= current_created_at:
                current["latest_text"] = item.get("text", "")
                current["latest_timestamp"] = item.get("timestamp", "Now")
                current["latest_created_at"] = next_created_at
                current["thinking_band"] = item.get("thinking_band", current["thinking_band"])

        ordered = sorted(
            grouped.values(),
            key=lambda group: (str(group.get("latest_created_at", "")), int(group.get("count", 0))),
            reverse=True,
        )[:limit]
        return [
            {
                "bias": item["bias"],
                "count": item["count"],
                "latest_text": item["latest_text"],
                "latest_timestamp": item["latest_timestamp"],
                "thinking_band": item["thinking_band"],
            }
            for item in ordered
        ]

    def _weekly_progress(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {
                "series": [],
                "improvement": 0.0,
                "direction": "steady",
                "summary": "Analyze thoughts across multiple weeks to unlock your progress trend.",
            }

        weekly_buckets: dict[datetime, dict[str, int]] = {}
        for item in history:
            parsed = self._parse_created_at(item)
            if not parsed:
                continue
            week_start = (parsed - timedelta(days=parsed.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            bucket = weekly_buckets.setdefault(week_start, {"total": 0, "negative": 0})
            bucket["total"] += 1
            if item.get("sentiment_bucket") != BALANCED_THINKING:
                bucket["negative"] += 1

        ordered_weeks = sorted(weekly_buckets.items())[-4:]
        series = [
            {
                "label": week_start.strftime("%b %d"),
                "value": round((stats["negative"] / stats["total"]) * 100, 1) if stats["total"] else 0.0,
            }
            for week_start, stats in ordered_weeks
        ]

        if len(series) < 2:
            return {
                "series": series,
                "improvement": 0.0,
                "direction": "steady",
                "summary": "Add entries across another week to compare whether negative thinking is decreasing over time.",
            }

        previous_value = float(series[-2]["value"])
        current_value = float(series[-1]["value"])
        improvement = round(previous_value - current_value, 1)
        if improvement > 0:
            direction = "improved"
            summary = f"Negative-thinking language improved by {improvement}% compared with last week."
        elif improvement < 0:
            direction = "declined"
            summary = f"Negative-thinking language increased by {abs(improvement)}% compared with last week."
        else:
            direction = "steady"
            summary = "Your negative-thinking percentage is steady week over week."

        return {
            "series": series,
            "improvement": improvement,
            "direction": direction,
            "summary": summary,
        }

    @staticmethod
    def _smart_insight_message(
        *,
        history_count: int,
        most_common_bias: str,
        most_used_word: str,
        risk_level: str,
    ) -> str:
        if history_count == 0:
            return "Analyze a thought to unlock a personalized language insight."
        if most_used_word != "No clear trigger yet":
            base = f"You frequently use '{most_used_word}', which can reinforce {most_common_bias.lower()}."
        else:
            base = f"{most_common_bias} is your most frequent recent pattern."

        if risk_level == "High Risk":
            return f"{base} Slowing the thought down before acting on it could help reduce distortion."
        if risk_level == "Medium Risk":
            return f"{base} Rewriting the thought in more specific language may lower the risk level."
        return f"{base} Your recent language is showing more balance overall."

    @staticmethod
    def _parse_created_at(item: dict[str, Any]) -> datetime | None:
        created_at = item.get("created_at")
        if not created_at:
            return None
        try:
            return datetime.fromisoformat(str(created_at))
        except Exception:
            return None
