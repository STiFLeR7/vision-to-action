"""
Gemini API Integration for Cognition Layer

LLM-based reasoning over structured CV outputs.
Operates on CPU, never receives raw images.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv, find_dotenv

# Ensure .env is loaded for API keys. Honor GEMINI_DOTENV_PATH if provided.
_DOTENV_PATH = os.getenv("GEMINI_DOTENV_PATH") or find_dotenv()
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH, override=False)

# Use absolute imports so the module works when executed via scripts/
from ingestion.schemas import Detection, PerceptionEvent


class GeminiReasoner:
    """
    Gemini-based cognition for structured CV outputs.
    
    Design principles (from SoW):
    - Never receives raw images
    - Operates only on structured outputs
    - CPU-only execution
    - Stateless by default
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        dotenv_path: Optional[str] = None
    ):
        """
        Initialize Gemini reasoner.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env var)
            model_name: Model to use
            temperature: Generation temperature
            max_output_tokens: Maximum response tokens
        """
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

        if api_key is None:
            # Load .env with explicit override to prevent environment pollution
            load_dotenv(dotenv_path='.env', override=True)
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

        self.api_key = api_key
        print(f"Loaded API key: {self.api_key[:4]}... (length: {len(self.api_key)})")
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        print(f"Initialized Gemini reasoner (REST): {model_name}")
    
    def _format_detection_summary(self, detections: List[Detection]) -> str:
        """Format detections for LLM consumption"""
        if not detections:
            return "No detections found."
        
        summary = f"Detected {len(detections)} objects:\n\n"
        
        # Group by class
        by_class = {}
        for det in detections:
            if det.class_name not in by_class:
                by_class[det.class_name] = []
            by_class[det.class_name].append(det)
        
        for class_name, dets in by_class.items():
            avg_conf = sum(d.confidence for d in dets) / len(dets)
            summary += f"- {class_name}: {len(dets)} instances (avg confidence: {avg_conf:.2f})\n"
            
            # List individual detections
            for i, det in enumerate(dets, 1):
                bbox = det.bbox
                summary += f"  {i}. bbox=({bbox.x1:.0f}, {bbox.y1:.0f}, {bbox.x2:.0f}, {bbox.y2:.0f}), "
                summary += f"confidence={det.confidence:.2f}\n"
        
        return summary
    
    def summarize_event(
        self,
        event: PerceptionEvent,
        include_metadata: bool = True
    ) -> str:
        """
        Generate natural language summary of a perception event.
        
        Args:
            event: PerceptionEvent to summarize
            include_metadata: Include processing metadata
            
        Returns:
            Natural language summary
        """
        detection_summary = self._format_detection_summary(event.detections)
        
        prompt = f"""You are analyzing structured computer vision detection results.

Event ID: {event.event_id}
Timestamp: {event.timestamp}
Image: {event.image_metadata.source_path} ({event.image_metadata.width}x{event.image_metadata.height})

Detection Results:
{detection_summary}

Please provide a clear, concise summary of what was detected in this image. Focus on:
1. What objects were found
2. Their spatial distribution
3. Confidence levels
4. Any notable patterns

Keep the summary factual and actionable."""

        if include_metadata and event.processing_metadata:
            prompt += f"\n\nProcessing metadata: {event.processing_metadata}"
        
        return self._call_gemini(prompt)
    
    def summarize_scene(self, event: PerceptionEvent) -> str:
        """Alias for summarize_event for compatibility."""
        return self.summarize_event(event, include_metadata=False)
    
    def explain_anomaly(
        self,
        event: PerceptionEvent,
        anomaly_description: str
    ) -> str:
        """
        Explain a detected anomaly.
        
        Args:
            event: PerceptionEvent with anomaly
            anomaly_description: Description of the anomaly
            
        Returns:
            Explanation and recommendations
        """
        detection_summary = self._format_detection_summary(event.detections)
        
        prompt = f"""You are analyzing an anomaly in computer vision detection results.

Event ID: {event.event_id}
Timestamp: {event.timestamp}

Anomaly Description:
{anomaly_description}

Detection Results:
{detection_summary}

Please provide:
1. A clear explanation of what the anomaly might indicate
2. Potential causes
3. Risk assessment (low/medium/high)
4. Recommended actions

Be precise and actionable."""

        return self._call_gemini(prompt)
    
    def generate_inspection_report(
        self,
        events: List[PerceptionEvent],
        report_type: str = "daily"
    ) -> str:
        """
        Generate an inspection report from multiple events.
        
        Args:
            events: List of PerceptionEvents
            report_type: Type of report (daily, weekly, incident)
            
        Returns:
            Formatted inspection report
        """
        if not events:
            return "No events to report."
        
        # Aggregate statistics
        total_detections = sum(len(e.detections) for e in events)
        
        all_classes = {}
        for event in events:
            for det in event.detections:
                if det.class_name not in all_classes:
                    all_classes[det.class_name] = {'count': 0, 'confidences': []}
                all_classes[det.class_name]['count'] += 1
                all_classes[det.class_name]['confidences'].append(det.confidence)
        
        stats = "Detection Statistics:\n\n"
        for class_name, data in all_classes.items():
            avg_conf = sum(data['confidences']) / len(data['confidences'])
            stats += f"- {class_name}: {data['count']} detections, avg confidence: {avg_conf:.2f}\n"
        
        prompt = f"""You are generating a {report_type} inspection report from computer vision system outputs.

Report Period: {events[0].timestamp.strftime('%Y-%m-%d')} to {events[-1].timestamp.strftime('%Y-%m-%d')}
Total Events: {len(events)}
Total Detections: {total_detections}

{stats}

Please generate a professional inspection report with:
1. Executive Summary
2. Key Findings
3. Trends and Patterns
4. Anomalies or Concerns
5. Recommendations

Format the report in markdown."""

        return self._call_gemini(prompt)
    
    def analyze_temporal_pattern(
        self,
        events: List[PerceptionEvent],
        time_window: str = "1 hour"
    ) -> str:
        """
        Analyze temporal patterns in detection events.
        
        Args:
            events: Chronologically ordered events
            time_window: Time window description
            
        Returns:
            Pattern analysis
        """
        if len(events) < 2:
            return "Insufficient events for temporal analysis."
        
        # Build timeline
        timeline = []
        for event in events:
            timeline.append({
                'timestamp': event.timestamp.strftime('%H:%M:%S'),
                'detections': len(event.detections),
                'classes': [d.class_name for d in event.detections]
            })
        
        timeline_str = "\n".join([
            f"{t['timestamp']}: {t['detections']} detections - {', '.join(set(t['classes']))}"
            for t in timeline
        ])
        
        prompt = f"""You are analyzing temporal patterns in computer vision detections.

Time Window: {time_window}
Number of Events: {len(events)}

Timeline:
{timeline_str}

Please analyze:
1. Detection frequency patterns
2. Changes over time
3. Repeated occurrences
4. Unusual temporal behavior
5. Predictions or alerts

Provide actionable insights."""

        return self._call_gemini(prompt)
    
    def batch_summarize(
        self,
        events: List[PerceptionEvent],
        max_events: int = 10
    ) -> List[str]:
        """
        Summarize multiple events in batch.
        
        Args:
            events: List of events to summarize
            max_events: Maximum events to process in one call
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i in range(0, len(events), max_events):
            batch = events[i:i+max_events]
            for event in batch:
                summary = self.summarize_event(event)
                summaries.append(summary)
                time.sleep(0.1)  # Rate limiting
        
        return summaries

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini via RESTful API."""
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise RuntimeError(f"Unexpected Gemini response: {data}")
