"""Unit tests for LangGraph agents."""

import pytest

from agents.planner import planner_node, _detect_repository, _classify_intent
from agents.critic import critic_node
from agents.formatter import formatter_node
from agents.state import ArchonState


class TestPlannerAgent:
    def test_detect_langchain(self):
        assert _detect_repository("How does LLMChain work?") == "langchain"

    def test_detect_crewai(self):
        assert _detect_repository("How to use CrewAI agents?") == "crewai"

    def test_classify_debug(self):
        assert _classify_intent("Fix this traceback error") == "debug"

    def test_classify_code_generation(self):
        assert _classify_intent("Write code to implement a retriever") == "code_generation"

    def test_planner_node(self):
        state: ArchonState = {"user_query": "Explain ChatPromptTemplate", "agent_trace": []}
        result = planner_node(state)
        assert "plan" in result
        assert result["repository"] == "langchain"
        assert len(result["agent_trace"]) == 1


class TestCriticAgent:
    def test_critic_detects_hallucination(self):
        state: ArchonState = {
            "generated_code": "prompt = ChatPromptTemplate(system_message='test')",
            "generated_explanation": "Use system_message param",
            "citations": [],
            "agent_trace": [],
        }
        result = critic_node(state)
        assert result["critic_score"] < 0.8
        assert len(result["hallucination_flags"]) > 0

    def test_critic_passes_good_code(self):
        state: ArchonState = {
            "generated_code": "from langchain.prompts import ChatPromptTemplate\nprompt = ChatPromptTemplate.from_messages([])",
            "generated_explanation": "Modern API usage",
            "citations": [{"source": "test.py", "type": "code", "content": "x"}],
            "agent_trace": [],
        }
        result = critic_node(state)
        assert result["critic_score"] > 0.5


class TestFormatterAgent:
    def test_formatter_computes_confidence(self):
        state: ArchonState = {
            "generated_explanation": "Test response",
            "critic_score": 0.8,
            "citations": [{"source": "a.py"}],
            "execution_result": {"success": True},
            "hallucination_flags": [],
            "agent_trace": [],
        }
        result = formatter_node(state)
        assert result["confidence_score"] > 0.5
        assert "final_response" in result
