"""Registro reproducible sin rutas personales, datos ni descarga de modelos."""
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from urllib.request import build_opener, ProxyHandler

report = {'python': platform.python_version(), 'sistema': platform.system(),
          'release': platform.release(), 'arquitectura': platform.machine(), 'paquetes': {}}
for name in ('streamlit','pandas','plotly','numpy','scikit-learn','pyarrow'):
    try: report['paquetes'][name] = version(name)
    except PackageNotFoundError: report['paquetes'][name] = 'no instalado'
try:
    opener = build_opener(ProxyHandler({}))
    def get(path):
        with opener.open('http://127.0.0.1:11434/api/'+path, timeout=5) as response:
            return json.load(response)
    report['ollama'] = {'version':get('version').get('version'),
                        'modelos':[{'nombre':m.get('name'),'digest':m.get('digest')}
                                   for m in get('tags').get('models',[])]}
except (OSError, ValueError) as exc:
    report['ollama'] = {'estado':'no disponible', 'tipo_error':type(exc).__name__}
print(json.dumps(report, ensure_ascii=False, indent=2))
