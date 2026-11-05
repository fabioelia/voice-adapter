"""Serve this folder on http://localhost:8000 with cross-origin isolation headers.

Cross-origin isolation lets ONNX Runtime Web use multi-threaded WebAssembly (SharedArrayBuffer),
which makes CPU generation several times faster. COEP "credentialless" keeps CDN and Hugging Face
downloads working (supported in Chrome and Edge; Safari ignores it and stays single-threaded).

    python3 tools/serve.py            # then open http://localhost:8000/index.html
"""
import http.server, functools, os, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        super().end_headers()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f'Serving {root} on http://localhost:{port}/ (cross-origin isolated)')
http.server.ThreadingHTTPServer(('localhost', port), functools.partial(Handler, directory=root)).serve_forever()
