from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MockN8NHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data)
            logging.info(f"Received Webhook: {self.path}")
            logging.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "mock": True}).encode('utf-8'))
            
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON payload")
            self.send_response(400)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=MockN8NHandler, port=5678):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f"Starting Mock n8n Server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping Mock n8n Server...")

if __name__ == '__main__':
    run()
