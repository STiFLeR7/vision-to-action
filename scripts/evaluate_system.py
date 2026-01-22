"""
Comprehensive evaluation script for Vision-to-Action system.
Tests the complete pipeline on a dataset and tracks all metrics.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
import traceback

# Bootstrap project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from cv.detection.yolov8_detector import YOLOv8Detector
from cv.analysis.ai_detection import AIImageDetector
from ingestion.storage.json_storage import JSONStorageBackend
from ingestion.schemas.perception_schema import PerceptionEvent, Detection
from cognition.gemini.reasoner import GeminiReasoner


class SystemEvaluator:
    """Comprehensive evaluation of the vision-to-action pipeline."""
    
    def __init__(self, test_dir: Path, eval_dir: Path):
        self.test_dir = test_dir
        self.eval_dir = eval_dir
        self.eval_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        print("Initializing system components...")
        self.detector = YOLOv8Detector()
        self.ai_detector = AIImageDetector()
        self.storage = JSONStorageBackend()
        self.reasoner = GeminiReasoner()
        
        # Metrics storage
        self.metrics = {
            "test_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_dir": str(test_dir),
                "total_images": 0,
                "hardware": "NVIDIA RTX 3050 6GB",
            },
            "perception_layer": {
                "total_detections": 0,
                "images_with_detections": 0,
                "images_without_detections": 0,
                "fps": [],
                "latency_ms": [],
                "confidence_scores": [],
                "detections_per_image": [],
            },
            "ingestion_layer": {
                "events_stored": 0,
                "storage_latency_ms": [],
                "avg_event_size_bytes": [],
            },
            "cognition_layer": {
                "summaries_generated": 0,
                "latency_ms": [],
                "token_counts": [],
                "failures": 0,
            },
            "ai_detection": {
                "total_analyzed": 0,
                "ai_generated_count": 0,
                "original_count": 0,
                "avg_ai_confidence": [],
                "latency_ms": [],
            },
            "end_to_end": {
                "total_latency_ms": [],
                "success_rate": 0.0,
                "workflow_trigger_accuracy": 0.0,
            },
            "per_image_results": []
        }
    
    def evaluate_image(self, image_path: Path) -> Dict[str, Any]:
        """Run full pipeline on single image and track metrics."""
        result = {
            "image": image_path.name,
            "success": False,
            "error": None,
            "ai_detection": {},
            "perception": {},
            "ingestion": {},
            "cognition": {},
            "timings": {}
        }
        
        try:
            # Start end-to-end timer
            e2e_start = time.perf_counter()
            
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # ========== AI DETECTION ==========
            ai_start = time.perf_counter()
            ai_result = self.ai_detector.analyze(str(image_path))
            ai_time = (time.perf_counter() - ai_start) * 1000
            
            result["ai_detection"] = {
                "verdict": ai_result["verdict"],
                "is_ai_generated": ai_result["is_ai_generated"],
                "ai_confidence": ai_result["ai_confidence"],
                "original_confidence": ai_result["original_confidence"],
                "latency_ms": ai_time,
                "explanation": ai_result["explanation"]
            }
            
            # Update AI detection metrics
            self.metrics["ai_detection"]["total_analyzed"] += 1
            self.metrics["ai_detection"]["latency_ms"].append(ai_time)
            self.metrics["ai_detection"]["avg_ai_confidence"].append(ai_result["ai_confidence"])
            
            if ai_result["is_ai_generated"]:
                self.metrics["ai_detection"]["ai_generated_count"] += 1
            else:
                self.metrics["ai_detection"]["original_count"] += 1
            
            # ========== PERCEPTION LAYER ==========
            perception_start = time.perf_counter()
            detection_result = self.detector.detect(img)
            perception_time = (time.perf_counter() - perception_start) * 1000
            
            # Convert DetectionResult to list of Detection objects
            detections = []
            for i in range(len(detection_result.boxes)):
                from ingestion.schemas.perception_schema import BoundingBox
                bbox = detection_result.boxes[i]
                detections.append(Detection(
                    bbox=BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), 
                                     x2=int(bbox[2]), y2=int(bbox[3])),
                    confidence=float(detection_result.scores[i]),
                    class_id=int(detection_result.classes[i]),
                    class_name=detection_result.class_names[i]
                ))
            
            result["perception"] = {
                "num_detections": len(detections),
                "latency_ms": perception_time,
                "fps": 1000 / perception_time if perception_time > 0 else 0,
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    } for d in detections
                ]
            }
            
            # Update perception metrics
            self.metrics["perception_layer"]["total_detections"] += len(detections)
            if len(detections) > 0:
                self.metrics["perception_layer"]["images_with_detections"] += 1
                self.metrics["perception_layer"]["confidence_scores"].extend(
                    [d.confidence for d in detections]
                )
            else:
                self.metrics["perception_layer"]["images_without_detections"] += 1
            
            self.metrics["perception_layer"]["latency_ms"].append(perception_time)
            self.metrics["perception_layer"]["fps"].append(result["perception"]["fps"])
            self.metrics["perception_layer"]["detections_per_image"].append(len(detections))
            
            # ========== INGESTION LAYER ==========
            ingestion_start = time.perf_counter()
            from ingestion.schemas.perception_schema import ImageMetadata, TaskType
            
            event = PerceptionEvent(
                event_id=f"eval_{image_path.stem}_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                task_type=TaskType.DETECTION,
                image_metadata=ImageMetadata(
                    source_path=str(image_path),
                    width=img.shape[1],
                    height=img.shape[0],
                    channels=img.shape[2] if len(img.shape) == 3 else 1,
                    dtype=str(img.dtype)
                ),
                detections=detections,
                processing_metadata={"evaluation": True}
            )
            event_path = self.storage.store_event(event)
            ingestion_time = (time.perf_counter() - ingestion_start) * 1000
            
            # Get event file size
            event_size = Path(event_path).stat().st_size
            
            result["ingestion"] = {
                "event_id": event.event_id,
                "event_path": event_path,
                "latency_ms": ingestion_time,
                "size_bytes": event_size
            }
            
            self.metrics["ingestion_layer"]["events_stored"] += 1
            self.metrics["ingestion_layer"]["storage_latency_ms"].append(ingestion_time)
            self.metrics["ingestion_layer"]["avg_event_size_bytes"].append(event_size)
            
            # ========== COGNITION LAYER ==========
            cognition_start = time.perf_counter()
            try:
                summary = self.reasoner.summarize_scene(event)
                cognition_time = (time.perf_counter() - cognition_start) * 1000
                
                result["cognition"] = {
                    "summary": summary,
                    "latency_ms": cognition_time,
                    "success": True
                }
                
                self.metrics["cognition_layer"]["summaries_generated"] += 1
                self.metrics["cognition_layer"]["latency_ms"].append(cognition_time)
            except Exception as e:
                cognition_time = (time.perf_counter() - cognition_start) * 1000
                result["cognition"] = {
                    "error": str(e),
                    "latency_ms": cognition_time,
                    "success": False
                }
                self.metrics["cognition_layer"]["failures"] += 1
            
            # ========== END-TO-END ==========
            e2e_time = (time.perf_counter() - e2e_start) * 1000
            result["timings"]["end_to_end_ms"] = e2e_time
            result["success"] = True
            
            self.metrics["end_to_end"]["total_latency_ms"].append(e2e_time)
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"   ✗ Error processing {image_path.name}: {e}")
        
        return result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test images."""
        test_images = list(self.test_dir.glob("*"))
        test_images = [p for p in test_images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        self.metrics["test_info"]["total_images"] = len(test_images)
        
        print(f"\n{'='*60}")
        print(f"Starting Evaluation on {len(test_images)} Images")
        print(f"{'='*60}\n")
        
        for idx, image_path in enumerate(test_images, 1):
            print(f"[{idx}/{len(test_images)}] Processing: {image_path.name}")
            result = self.evaluate_image(image_path)
            self.metrics["per_image_results"].append(result)
            
            if result["success"]:
                print(f"   ✓ AI: {result['ai_detection']['verdict']}, "
                      f"Detections: {result['perception']['num_detections']}, "
                      f"E2E: {result['timings']['end_to_end_ms']:.1f}ms")
            else:
                print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        
        return self.metrics
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate statistics."""
        # Perception aggregates
        if self.metrics["perception_layer"]["latency_ms"]:
            self.metrics["perception_layer"]["avg_latency_ms"] = np.mean(
                self.metrics["perception_layer"]["latency_ms"]
            )
            self.metrics["perception_layer"]["avg_fps"] = np.mean(
                self.metrics["perception_layer"]["fps"]
            )
        
        if self.metrics["perception_layer"]["confidence_scores"]:
            self.metrics["perception_layer"]["avg_confidence"] = np.mean(
                self.metrics["perception_layer"]["confidence_scores"]
            )
            self.metrics["perception_layer"]["min_confidence"] = np.min(
                self.metrics["perception_layer"]["confidence_scores"]
            )
            self.metrics["perception_layer"]["max_confidence"] = np.max(
                self.metrics["perception_layer"]["confidence_scores"]
            )
        
        # Ingestion aggregates
        if self.metrics["ingestion_layer"]["storage_latency_ms"]:
            self.metrics["ingestion_layer"]["avg_latency_ms"] = np.mean(
                self.metrics["ingestion_layer"]["storage_latency_ms"]
            )
            self.metrics["ingestion_layer"]["avg_size_bytes"] = np.mean(
                self.metrics["ingestion_layer"]["avg_event_size_bytes"]
            )
        
        # Cognition aggregates
        if self.metrics["cognition_layer"]["latency_ms"]:
            self.metrics["cognition_layer"]["avg_latency_ms"] = np.mean(
                self.metrics["cognition_layer"]["latency_ms"]
            )
            self.metrics["cognition_layer"]["success_rate"] = (
                self.metrics["cognition_layer"]["summaries_generated"] /
                (self.metrics["cognition_layer"]["summaries_generated"] + 
                 self.metrics["cognition_layer"]["failures"])
            )
        
        # AI detection aggregates
        if self.metrics["ai_detection"]["latency_ms"]:
            self.metrics["ai_detection"]["avg_latency_ms"] = np.mean(
                self.metrics["ai_detection"]["latency_ms"]
            )
            self.metrics["ai_detection"]["avg_confidence"] = np.mean(
                self.metrics["ai_detection"]["avg_ai_confidence"]
            )
            self.metrics["ai_detection"]["ai_generated_pct"] = (
                self.metrics["ai_detection"]["ai_generated_count"] / 
                self.metrics["ai_detection"]["total_analyzed"] * 100
            )
        
        # End-to-end aggregates
        if self.metrics["end_to_end"]["total_latency_ms"]:
            self.metrics["end_to_end"]["avg_latency_ms"] = np.mean(
                self.metrics["end_to_end"]["total_latency_ms"]
            )
            self.metrics["end_to_end"]["min_latency_ms"] = np.min(
                self.metrics["end_to_end"]["total_latency_ms"]
            )
            self.metrics["end_to_end"]["max_latency_ms"] = np.max(
                self.metrics["end_to_end"]["total_latency_ms"]
            )
            
            # Success rate
            successful = sum(1 for r in self.metrics["per_image_results"] if r["success"])
            self.metrics["end_to_end"]["success_rate"] = successful / len(self.metrics["per_image_results"])
        
        # Workflow trigger accuracy (based on detection success)
        self.metrics["end_to_end"]["workflow_trigger_accuracy"] = (
            self.metrics["perception_layer"]["images_with_detections"] / 
            self.metrics["test_info"]["total_images"]
            if self.metrics["test_info"]["total_images"] > 0 else 0.0
        )
        
        # Manual effort reduction (estimated based on automation)
        # Assumption: Manual review takes 30 seconds per image, automated is <5 seconds
        manual_time = self.metrics["test_info"]["total_images"] * 30000  # ms
        automated_time = sum(self.metrics["end_to_end"]["total_latency_ms"])
        self.metrics["end_to_end"]["manual_effort_reduction_pct"] = (
            (manual_time - automated_time) / manual_time * 100
            if manual_time > 0 else 0.0
        )
    
    def save_results(self):
        """Save evaluation results to JSON files."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save full metrics
        metrics_path = self.eval_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save summary report
        summary_path = self.eval_dir / f"summary_{timestamp}.json"
        summary = {
            "timestamp": self.metrics["test_info"]["timestamp"],
            "hardware": self.metrics["test_info"]["hardware"],
            "total_images": self.metrics["test_info"]["total_images"],
            "perception": {
                "avg_latency_ms": self.metrics["perception_layer"].get("avg_latency_ms", 0),
                "avg_fps": self.metrics["perception_layer"].get("avg_fps", 0),
                "total_detections": self.metrics["perception_layer"]["total_detections"],
                "avg_confidence": self.metrics["perception_layer"].get("avg_confidence", 0),
            },
            "ingestion": {
                "events_stored": self.metrics["ingestion_layer"]["events_stored"],
                "avg_latency_ms": self.metrics["ingestion_layer"].get("avg_latency_ms", 0),
            },
            "cognition": {
                "summaries_generated": self.metrics["cognition_layer"]["summaries_generated"],
                "avg_latency_ms": self.metrics["cognition_layer"].get("avg_latency_ms", 0),
                "success_rate": self.metrics["cognition_layer"].get("success_rate", 0),
            },
            "ai_detection": {
                "total_analyzed": self.metrics["ai_detection"]["total_analyzed"],
                "ai_generated_count": self.metrics["ai_detection"]["ai_generated_count"],
                "original_count": self.metrics["ai_detection"]["original_count"],
                "ai_generated_pct": self.metrics["ai_detection"].get("ai_generated_pct", 0),
                "avg_latency_ms": self.metrics["ai_detection"].get("avg_latency_ms", 0),
            },
            "end_to_end": {
                "avg_latency_ms": self.metrics["end_to_end"].get("avg_latency_ms", 0),
                "success_rate": self.metrics["end_to_end"]["success_rate"],
                "workflow_trigger_accuracy": self.metrics["end_to_end"]["workflow_trigger_accuracy"],
                "manual_effort_reduction_pct": self.metrics["end_to_end"].get("manual_effort_reduction_pct", 0),
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Metrics saved to: {metrics_path}")
        print(f"✓ Summary saved to: {summary_path}")
        
        return metrics_path, summary_path
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"📊 Test Information:")
        print(f"   Total Images: {self.metrics['test_info']['total_images']}")
        print(f"   Hardware: {self.metrics['test_info']['hardware']}")
        print(f"   Timestamp: {self.metrics['test_info']['timestamp']}\n")
        
        print(f"👁️  Perception Layer:")
        print(f"   Total Detections: {self.metrics['perception_layer']['total_detections']}")
        print(f"   Images with Detections: {self.metrics['perception_layer']['images_with_detections']}")
        print(f"   Avg Latency: {self.metrics['perception_layer'].get('avg_latency_ms', 0):.2f} ms")
        print(f"   Avg FPS: {self.metrics['perception_layer'].get('avg_fps', 0):.2f}")
        print(f"   Avg Confidence: {self.metrics['perception_layer'].get('avg_confidence', 0):.3f}\n")
        
        print(f"📥 Ingestion Layer:")
        print(f"   Events Stored: {self.metrics['ingestion_layer']['events_stored']}")
        print(f"   Avg Latency: {self.metrics['ingestion_layer'].get('avg_latency_ms', 0):.2f} ms")
        print(f"   Avg Event Size: {self.metrics['ingestion_layer'].get('avg_size_bytes', 0):.0f} bytes\n")
        
        print(f"🧠 Cognition Layer:")
        print(f"   Summaries Generated: {self.metrics['cognition_layer']['summaries_generated']}")
        print(f"   Failures: {self.metrics['cognition_layer']['failures']}")
        print(f"   Success Rate: {self.metrics['cognition_layer'].get('success_rate', 0):.1%}")
        print(f"   Avg Latency: {self.metrics['cognition_layer'].get('avg_latency_ms', 0):.2f} ms\n")
        
        print(f"🤖 AI Detection:")
        print(f"   Total Analyzed: {self.metrics['ai_detection']['total_analyzed']}")
        print(f"   AI-Generated: {self.metrics['ai_detection']['ai_generated_count']} ({self.metrics['ai_detection'].get('ai_generated_pct', 0):.1f}%)")
        print(f"   Original/Real: {self.metrics['ai_detection']['original_count']}")
        print(f"   Avg Latency: {self.metrics['ai_detection'].get('avg_latency_ms', 0):.2f} ms\n")
        
        print(f"🔄 End-to-End:")
        print(f"   Avg Latency: {self.metrics['end_to_end'].get('avg_latency_ms', 0):.2f} ms")
        print(f"   Success Rate: {self.metrics['end_to_end']['success_rate']:.1%}")
        print(f"   Workflow Trigger Accuracy: {self.metrics['end_to_end']['workflow_trigger_accuracy']:.1%}")
        print(f"   Manual Effort Reduction: {self.metrics['end_to_end'].get('manual_effort_reduction_pct', 0):.1f}%\n")


def main():
    test_dir = PROJECT_ROOT / "data" / "test"
    eval_dir = PROJECT_ROOT / "eval"
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    evaluator = SystemEvaluator(test_dir, eval_dir)
    evaluator.run_evaluation()
    evaluator.save_results()
    evaluator.print_summary()


if __name__ == "__main__":
    main()
