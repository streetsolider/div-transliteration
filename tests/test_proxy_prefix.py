"""The page is served behind a reverse proxy in production (Swarm ingress,
Traefik, nginx), sometimes under a path prefix. Every URL it fetches — the
ready poll, /transliterate, the two Thaana fonts — used to be hardcoded
root-relative, so under /anything/ the fonts 404'd and the ready poll never
returned 200, leaving the "Loading AI model" overlay up forever. That looked
to users like the model not loading on the server, while the same image
worked locally at http://localhost:5001/.

These tests pin the fix: url_for in the template plus ProxyFix(x_prefix=1),
so the prefix arrives either as X-Forwarded-Prefix from the proxy or as
SCRIPT_NAME from the WSGI server (gunicorn passes the env var through).
"""
import os
import re
import sys

os.environ['SKIP_MODEL_PRELOAD'] = '1'  # must precede the app import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as transliteration  # noqa: E402

FETCHED = ['ready', 'transliterate', 'static/fonts/Faruma.ttf', 'static/fonts/Bolhu-Bold.ttf']


def _page_urls(html):
    found = set()
    for font, endpoint in re.findall(r"url\('([^']+\.ttf)'\)|fetch\('([^']+)'", html):
        found.add(font or endpoint)
    return found


def _client():
    return transliteration.app.test_client()


def test_unprefixed_deployment_keeps_root_relative_urls():
    html = _client().get('/').get_data(as_text=True)
    assert _page_urls(html) == {'/' + p for p in FETCHED}


def test_x_forwarded_prefix_is_applied_to_every_fetched_url():
    html = _client().get('/', headers={'X-Forwarded-Prefix': '/translit'}).get_data(as_text=True)
    assert _page_urls(html) == {'/translit/' + p for p in FETCHED}


def test_script_name_from_the_wsgi_server_is_applied_too():
    html = _client().get('/', environ_overrides={'SCRIPT_NAME': '/translit'}).get_data(as_text=True)
    assert _page_urls(html) == {'/translit/' + p for p in FETCHED}


def test_ready_is_503_until_both_models_are_loaded():
    # The healthcheck relies on this: urllib.request.urlopen raises on 503,
    # so the container stays "starting" until every direction is in memory.
    assert _client().get('/ready').status_code == 503


def test_fonts_are_served_from_static():
    assert _client().get('/static/fonts/Faruma.ttf').status_code == 200
