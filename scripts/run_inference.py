#!/usr/bin/env python3
"""
End-to-end inference pipeline for Vision-to-Action system.

Demonstrates:
1. Perception (detection)
2. Ingestion (structured storage)
3. Cognition (Gemini summary)
4. Action (n8n webhook trigger)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Vision-to-Action imports
from cv.detection import YOLOv8Detector
from cv.analysis.ai_detection import AIImageDetector
from ingestion.schemas import (
    PerceptionEvent, Detection, BoundingBox,
    ImageMetadata, TaskType
)
from ingestion.storage import JSONStorageBackend, generate_event_id
from cognition.gemini import GeminiReasoner
from configs.config_manager import get_config_manager


def run_inference_pipeline(
    image_path: str,
    enable_cognition: bool = True,
    enable_orchestration: bool = False,
    confidence_threshold: float = 0.7
):
    """
    Run full inference pipeline.
    
    Args:
        image_path: Path to input image
        enable_cognition: Enable Gemini reasoning
        enable_orchestration: Trigger n8n workflows
        confidence_threshold: Detection confidence threshold
    """
    print(f"\n{'='*60}")
    print(f"Vision-to-Action Inference Pipeline")
    print(f"{'='*60}\n")
    
    # Load configuration
    config_manager = get_config_manager()
    system_config = config_manager.get_system_config()
    
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape
    
    # 0. AI DETECTION: Check if image is AI-generated
    print(f"\n🤖 AI DETECTION")
    print(f"   Analyzing image authenticity...")
    
    ai_detector = AIImageDetector()
    ai_result = ai_detector.analyze(image_path)
    
    print(f"\n   {ai_result['verdict']}:")
    print(f"   • AI Confidence: {ai_result['ai_confidence']:.1%}")
    print(f"   • Original Confidence: {ai_result['original_confidence']:.1%}")
    print(f"   • Signals: {ai_result['explanation']}")
    
    # 1. PERCEPTION: Object Detection
    print(f"\n1️⃣  PERCEPTION LAYER")
    print(f"   Initializing YOLOv8 detector...")
    
    detector = YOLOv8Detector(
        variant="nano",
        device=system_config.hardware['target_device'],
        confidence_threshold=system_config.models['detection']['confidence_threshold'],
        input_size=tuple(system_config.models['detection']['input_size']),
        use_fp16=(system_config.hardware['precision'] == 'fp16')
    )
    
    print(f"   Running detection...")
    result = detector.detect(image_rgb, preprocess=True)
    
    print(f"\n   ✓ Found {len(result.boxes)} objects:")
    for i, (box, score, cls_name) in enumerate(zip(result.boxes, result.scores, result.class_names), 1):
        print(f"      {i}. {cls_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    
    # 2. INGESTION: Structured Storage
    print(f"\n2️⃣  INGESTION LAYER")
    print(f"   Creating structured event record...")
    
    # Convert to schema
    detections = []
    for box, score, cls_id, cls_name in zip(result.boxes, result.scores, result.classes, result.class_names):
        detections.append(Detection(
            bbox=BoundingBox(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])),
            confidence=float(score),
            class_id=int(cls_id),
            class_name=cls_name
        ))
    
    event = PerceptionEvent(
        event_id=generate_event_id(),
        timestamp=datetime.now(timezone.utc),
        task_type=TaskType.DETECTION,
        image_metadata=ImageMetadata(
            source_path=image_path,
            width=w,
            height=h,
            channels=c,
            dtype=str(image_rgb.dtype)
        ),
        detections=detections,
        processing_metadata=result.metadata
    )
    
    # Store event
    storage = JSONStorageBackend()
    filepath = storage.store_event(event)
    print(f"   ✓ Event stored: {filepath}")
    print(f"   Event ID: {event.event_id}")
    
    # 3. COGNITION: Gemini Reasoning
    if enable_cognition:
        print(f"\n3️⃣  COGNITION LAYER")
        print(f"   Initializing Gemini reasoner...")
        
        try:
            reasoner = GeminiReasoner()
            
            print(f"   Generating natural language summary...")
            summary = reasoner.summarize_event(event)
            
            print(f"\n   📝 GEMINI SUMMARY:")
            print(f"   {'-'*55}")
            for line in summary.split('\n'):
                print(f"   {line}")
            print(f"   {'-'*55}")
            
        except Exception as e:
            print(f"   ⚠️  Cognition failed: {e}")
    
    # 4. ACTION: n8n Orchestration
    if enable_orchestration:
        print(f"\n4️⃣  ACTION LAYER")
        print(f"   Orchestrating workflows via n8n...")
        
        try:
            from orchestration.n8n.notifier import N8NOrchestrator
            orchestrator = N8NOrchestrator(config_manager)
            
            # Workflow 02: AI Detection Alert
            if ai_result['verdict'] == 'AI-Generated' or ai_result['ai_confidence'] > 0.5:
                print(f"   ⚡ Triggering AI Detection Alert (Email)...")
                orchestrator.trigger_detection_alert_email({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "confidence_score": ai_result['ai_confidence'],
                    "signals": ai_result.get('signals', {}), # Assuming signals are part of result
                    "verdict": ai_result['verdict']
                })

            # Check for high-confidence detections
            high_confidence_detections = [d for d in detections if d.confidence >= confidence_threshold]
            
            if high_confidence_detections:
                print(f"   ⚡ Processing {len(high_confidence_detections)} detections for Smart Routing...")
                
                for det in high_confidence_detections:
                    payload = {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "detection_class": det.class_name, # Alias for smart routing
                        "location": "Main Camera Feed", # Placeholder
                        "bbox": {
                            "x1": det.bbox.x1, "y1": det.bbox.y1,
                            "x2": det.bbox.x2, "y2": det.bbox.y2
                        },
                        "image_path": image_path,
                        "device": system_config.hardware['target_device']
                    }
                    
                    # Workflow 03: Smart Routing
                    orchestrator.trigger_smart_routing(payload)
                    
                    # Workflow 06: Comprehensive Alert (Logging)
                    orchestrator.trigger_comprehensive_alert(payload)

            else:
                print(f"   No high-confidence detections to route (threshold: {confidence_threshold})")
                
        except Exception as e:
            print(f"   ⚠️  Orchestration failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"{'='*60}\n")
    
    return event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vision-to-Action inference pipeline")
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--no-cognition', action='store_true', help='Disable Gemini reasoning')
    parser.add_argument('--enable-orchestration', action='store_true', help='Enable n8n workflows')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for actions')
    
    args = parser.parse_args()
    
    run_inference_pipeline(
        image_path=args.image,
        enable_cognition=not args.no_cognition,
        enable_orchestration=args.enable_orchestration,
        confidence_threshold=args.threshold
    )
