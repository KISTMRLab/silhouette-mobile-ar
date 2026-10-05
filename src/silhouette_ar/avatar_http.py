"""Serve bundled character assets and pinned renderer modules for local demos."""
from pathlib import Path
from urllib.parse import unquote,urlsplit

def serve_avatar_asset(handler,static):
    path=unquote(urlsplit(handler.path).path)
    companion=path=='/static/recorded-motion.html'
    module=path in {'/static/avatar.js','/static/speech.js','/static/gesture-library.js','/static/application-gesture.js'}
    if not companion and not module and not path.startswith(('/static/avatars/','/static/vendor/','/static/examples/')):return False
    root=Path(static).resolve();target=(root/path.removeprefix('/static/')).resolve()
    if not target.is_relative_to(root) or (companion and target.name!='recorded-motion.html') or (not companion and target.suffix not in {'.glb','.json','.md','.js','.png'}) or not target.is_file():
        handler.send_error(404);return True
    data=target.read_bytes();kind={'.glb':'model/gltf-binary','.js':'text/javascript','.json':'application/json','.png':'image/png','.md':'text/plain','.html':'text/html; charset=utf-8'}[target.suffix]
    handler.send_response(200);handler.send_header('Content-Type',kind);handler.send_header('Content-Length',str(len(data)));handler.end_headers();handler.wfile.write(data);return True
