import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from configs.config_manager import get_config_manager

class N8NOrchestrator:
    """
    Handles communication with n8n webhook workflows.
    """
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_orchestration_config()
        self.base_url = self.config['n8n']['base_url']
        self.logger = logging.getLogger("N8NOrchestrator")
        
    def _send_webhook(self, endpoint_key: str, payload: Dict[str, Any]) -> bool:
        """
        Generic method to send data to a configured n8n endpoint.
        """
        try:
            # Get specific path for the workflow
            # Expected config structure: n8n.workflows.endpoint_key.path
            workflow_config = self.config.get('workflows', {}).get(endpoint_key)
            if not workflow_config:
                self.logger.warning(f"No configuration found for workflow: {endpoint_key}")
                return False
                
            path = workflow_config.get('path')
            if not path:
                self.logger.warning(f"No path configured for workflow: {endpoint_key}")
                return False

            url = f"{self.base_url}{path}"
            
            # Enrich payload with standard metadata if missing
            if 'timestamp' not in payload:
                payload['timestamp'] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Triggering {endpoint_key} workflow at {url}")
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            
            self.logger.info(f"Successfully triggered {endpoint_key}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to trigger {endpoint_key}: {e}")
            return False

    def trigger_detection_alert_discord(self, event_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 01: Detection Alert (Discord)"""
        return self._send_webhook('detection_alert_discord', event_data)

    def trigger_detection_alert_email(self, event_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 02: Detection Alert (Email)"""
        return self._send_webhook('detection_alert_email', event_data)

    def trigger_smart_routing(self, event_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 03: Smart Routing"""
        return self._send_webhook('smart_routing', event_data)

    def trigger_temporal_analysis(self, batch_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 04: Temporal Patterns"""
        return self._send_webhook('temporal_patterns', batch_data)

    def trigger_anomaly_escalation(self, anomaly_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 05: Anomaly Escalation"""
        return self._send_webhook('anomaly_escalation', anomaly_data)

    def trigger_comprehensive_alert(self, event_data: Dict[str, Any]) -> bool:
        """Trigger Workflow 06: Comprehensive Detection Alert"""
        return self._send_webhook('detection_alert', event_data)
