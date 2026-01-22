# n8n Workflow Templates for Vision-to-Action

This directory contains n8n workflow templates for agentic orchestration.

## Workflows

### 1. Detection Alert Workflow (`detection_alert_workflow.json`)

**Trigger:** High-confidence detection events

**Actions:**
- Check confidence threshold (> 0.7)
- Format and send email alert
- Send webhook to external systems
- Log to database
- Return webhook response

**Usage:**
```bash
curl -X POST http://localhost:5678/webhook/vision-to-action-detection \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_20260122_120000_001",
    "timestamp": "2026-01-22T12:00:00Z",
    "class_name": "person",
    "confidence": 0.95,
    "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
    "image_path": "data/images/frame_001.jpg",
    "device": "cuda:0"
  }'
```

### 2. Anomaly Escalation Workflow (`anomaly_escalation_workflow.json`)

**Trigger:** Critical or high-severity anomalies

**Actions:**
- Check severity level (high/critical)
- Format critical alert email
- Send Slack notification
- Log escalation to database
- Return webhook response

**Usage:**
```bash
curl -X POST http://localhost:5678/webhook/vision-to-action-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "anomaly_20260122_120000_abc",
    "event_id": "evt_20260122_120000_001",
    "timestamp": "2026-01-22T12:00:00Z",
    "anomaly_type": "repeated_detection",
    "severity": "critical",
    "description": "Person detected in restricted area 3 times in 60 seconds",
    "gemini_explanation": "High-frequency detection suggests persistent presence...",
    "recommended_actions": "1. Verify location access\n2. Review camera footage\n3. Dispatch security"
  }'
```

## Setup

### Import Workflows

1. Open n8n interface (http://localhost:5678)
2. Click "Workflows" → "Import from File"
3. Select workflow JSON file
4. Configure credentials:
   - Email (SMTP settings)
   - Webhooks (external endpoints)
   - Database (PostgreSQL connection)
   - Slack (webhook URL)

### Configure Endpoints

Update the following in workflow nodes:

- **Email:** Set SMTP server and credentials
- **Webhooks:** Update external API URLs
- **Database:** Configure PostgreSQL connection
- **Slack:** Add your Slack webhook URL

## Integration with Vision-to-Action

From Python code:

```python
import requests

def trigger_detection_alert(event_data):
    """Trigger n8n detection alert workflow"""
    response = requests.post(
        "http://localhost:5678/webhook/vision-to-action-detection",
        json=event_data
    )
    return response.json()

def trigger_anomaly_escalation(anomaly_data):
    """Trigger n8n anomaly escalation workflow"""
    response = requests.post(
        "http://localhost:5678/webhook/vision-to-action-anomaly",
        json=anomaly_data
    )
    return response.json()
```

## Customization

Each workflow can be customized for:
- Different confidence thresholds
- Additional notification channels
- Custom processing logic
- Integration with other services

## Governance Integration

These workflows integrate with imgshape v4 governance:
- Triggered by imgshape compatibility failures
- Quality issue escalations
- Dataset validation alerts
