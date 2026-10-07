"""/v1/models reports the served context window (context_length and vLLM's max_model_len) when the engine knows it."""

from contextlib import contextmanager
import http.client
import json
import threading
from types import SimpleNamespace

from tensorfold.cuda.http import make_handler, model_entry
from tensorfold.server.http import Server


@contextmanager
def serving(**extra):
    app = SimpleNamespace(served="fixture", served_name="fixture", model_ids=["fixture", "alias"], max_batch_size=1,
                          auth=None, **extra)
    server = Server(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def models(port):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("GET", "/v1/models")
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def test_every_served_name_reports_the_window():
    with serving(effective_context_window=262144) as port:
        status, body = models(port)
    assert status == 200
    assert [m["id"] for m in body["data"]] == ["fixture", "alias"]
    for entry in body["data"]:
        assert entry["context_length"] == entry["max_model_len"] == 262144
        assert entry["object"] == "model" and entry["owned_by"] == "tensorfold"


def test_an_unknown_window_omits_both_fields():
    with serving() as port:
        _, body = models(port)
    assert all("context_length" not in m and "max_model_len" not in m for m in body["data"])
    for window in (None, 0, True, "262144"):
        assert set(model_entry(SimpleNamespace(effective_context_window=window), "m")) == {"id", "object", "owned_by"}
