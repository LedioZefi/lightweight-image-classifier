"""
OpenRouter API client for drafting README sections and explaining predictions.

This module provides utilities to interact with OpenRouter API for:
- Generating README documentation sections
- Explaining image classification predictions
"""

import os
import json
import requests
from typing import Optional, Dict, Any


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "meta-llama/llama-2-7b-chat"  # Default model

    def draft_readme_section(self, section_title: str, context: str) -> str:
        """
        Draft a README section using OpenRouter.

        Args:
            section_title: Title of the section (e.g., "Installation", "Usage")
            context: Context about the project to include in the prompt

        Returns:
            Generated README section text
        """
        if not self.api_key:
            return f"# {section_title}\n\n[OpenRouter API key not configured. Set OPENROUTER_API_KEY env var.]\n"

        prompt = f"""Write a concise README section for a lightweight image classifier project.

Section Title: {section_title}
Project Context: {context}

Requirements:
- Keep it brief and practical
- Include code examples if relevant
- Use markdown formatting
- Focus on clarity for developers

Generate only the section content, no title."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error generating section: {str(e)}]\n"

    def explain_prediction(
        self, class_name: str, confidence: float, top_classes: Dict[str, float]
    ) -> str:
        """
        Generate an explanation for a classification prediction.

        Args:
            class_name: Predicted class name
            confidence: Confidence score (0-1)
            top_classes: Dict of top predicted classes and their scores

        Returns:
            Human-readable explanation of the prediction
        """
        if not self.api_key:
            return f"Predicted: {class_name} ({confidence:.2%})"

        top_classes_str = "\n".join(
            [f"  - {cls}: {score:.2%}" for cls, score in top_classes.items()]
        )

        prompt = f"""Provide a brief, friendly explanation of this image classification result.

Predicted Class: {class_name}
Confidence: {confidence:.2%}
Top Predictions:
{top_classes_str}

Keep the explanation to 1-2 sentences. Be conversational and informative."""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 150,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Predicted: {class_name} ({confidence:.2%}) [Error generating explanation: {str(e)}]"

