"""Cliente de Ollama por HTTP local. No requiere servicios cloud."""
import json
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

ALLOWED_MODELS = ('llama3.2:3b', 'llama3.2:1b')


class OllamaClient:
    def __init__(self, model='llama3.2:3b'):
        if model not in ALLOWED_MODELS:
            raise ValueError('Modelo fuera de la lista autorizada.')
        self.model = model
        self.base_url = 'http://127.0.0.1:11434'
        self.opener = build_opener(ProxyHandler({}))

    def _request(self, path, body=None, timeout=10):
        payload = json.dumps(body).encode('utf-8') if body is not None else None
        request = Request(self.base_url + path, data=payload,
                          headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError('No se completó la conexión con Ollama. Comprueba que esté abierto, '
                               'que el modelo esté descargado y que haya memoria disponible.') from exc

    def status(self):
        version = self._request('/api/version').get('version', 'desconocida')
        models = self._request('/api/tags').get('models', [])
        names = [m.get('name') for m in models]
        return {'version': version, 'modelo': self.model,
                'disponible': self.model in names, 'instalados': names}

    def chat(self, messages, schema=None):
        body = {'model': self.model, 'messages': messages, 'stream': False,
                'options': {'temperature': 0, 'num_predict': 1600 if schema else 650, 'num_ctx': 8192}}
        if schema:
            body['format'] = schema
        result = self._request('/api/chat', body, timeout=180)
        text = result.get('message', {}).get('content')
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Ollama devolvió una respuesta vacía.')
        return text
