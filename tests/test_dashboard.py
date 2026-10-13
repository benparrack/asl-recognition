"""SignDesk dashboard: transcript editing, letter hold logic, settings
validation, and a server smoke test on a blank video."""

import json
import socket
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

import config
from dashboard_engine import LetterCommitter, Settings, Transcript, clean_settings


def test_transcript_letters_words_and_punctuation():
    t = Transcript()
    t.punct(".")                                # nothing to punctuate yet
    assert t.text == "" and not t._undo
    for ch in "HI":
        t.add_letter(ch)
    t.add_word("pencil", 1)
    assert t.text == "HI pencil "
    t.punct("?")
    assert t.text == "HI pencil? "
    t.newline()
    t.add_word("ok", 2)
    assert t.text == "HI pencil?\nok "
    t.delete_word()
    assert t.text == "HI pencil?\n"
    t.backspace()
    t.space()
    t.space()                                   # no double spaces
    assert t.text == "HI pencil? "


def test_transcript_swap_last_word_only_while_it_is_last():
    t = Transcript()
    t.add_word("pen", 7)
    assert t.replace_last_word("pencil", 7) and t.text == "pencil "
    assert not t.replace_last_word("hot", 6)    # another event's word
    t.add_letter("A")
    assert not t.replace_last_word("hot", 7)    # something typed after it
    assert t.text == "pencil A"


def test_transcript_undo_redo():
    t = Transcript()
    t.add_word("a")
    t.add_word("b")
    t.clear()
    t.undo()
    assert t.text == "a b "
    t.undo()
    assert t.text == "a "
    t.redo()
    assert t.text == "a b "
    t.add_letter("X")                           # new edit drops the redo stack
    t.redo()
    assert t.text == "a b X"


def test_letter_committer_fires_once_per_hold_and_suppresses():
    c = LetterCommitter(hold=0.5)
    assert c.update("A", 0.0) is None
    assert c.update("A", 0.4) is None and 0.7 < c.progress < 0.9
    assert c.update("A", 0.5) == "A"
    assert c.update("A", 2.0) is None           # held on: still only once
    assert c.update(None, 2.1) is None          # hand dropped
    c.update("A", 3.0)
    assert c.update("A", 3.6) == "A"            # same letter again after a drop
    c.suppress()                                # a J was just added...
    c.update("I", 4.0)
    assert c.update("I", 5.0) is None           # ...so its end handshape isn't


def test_clean_settings_coerces_clamps_and_ignores_junk():
    base = Settings()
    s = clean_settings({"letters_hold": "9", "letters_window": 2.7, "mirror": "no",
                        "letters_min_confidence": float("nan"), "evil": 1}, base)
    assert s.letters_hold == 3.0                 # "9" coerced, then clamped
    assert isinstance(s.letters_window, int)
    assert s.mirror is base.mirror               # non-bool ignored
    assert s.letters_min_confidence == base.letters_min_confidence
    assert not hasattr(s, "evil")


# ---------------------------------------------------------------- server
CK = config.CHECKPOINT_DIR
LETTERS, WORDS = CK / "mlp_pugeault_best.pt", CK / "words_transformer_full_aslc_r3_best.pt"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not (LETTERS.exists() and WORDS.exists()):
        pytest.skip("trained checkpoints not present")
    from http.server import ThreadingHTTPServer

    from dashboard import make_handler
    from dashboard_engine import Engine

    tmp = tmp_path_factory.mktemp("signdesk")
    video = tmp / "blank.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
    for _ in range(15):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = Engine(LETTERS, None, WORDS, 0, str(video), state_dir=tmp / "state")
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine, port))
    srv.daemon_threads = True
    engine.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield engine, f"http://127.0.0.1:{port}"
    srv.shutdown()
    engine.stop()
    srv.server_close()


def post(url, body, headers=None):
    req = urllib.request.Request(url + "/api/command", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def test_server_pages_state_and_streams(server):
    _, url = server
    with urllib.request.urlopen(url + "/", timeout=10) as r:
        assert b"SignDesk" in r.read() and "default-src 'self'" in r.headers["Content-Security-Policy"]
    for path in ("/app.js", "/app.css"):
        urllib.request.urlopen(url + path, timeout=10).close()
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/../dashboard.py", timeout=10)
    assert e.value.code == 404
    vocab = json.load(urllib.request.urlopen(url + "/api/vocab", timeout=10))
    assert len(vocab) == 250 and {"name", "label", "text", "two_handed"} <= set(vocab[0])

    with urllib.request.urlopen(url + "/stream.mjpg", timeout=10) as r:
        assert r.read(7) == b"--frame"
    with urllib.request.urlopen(url + "/events", timeout=10) as r:
        line = r.readline()
        while not line.startswith(b"data: "):
            line = r.readline()
        assert "transcript" in json.loads(line[6:])


def test_server_rejects_foreign_host_and_non_json(server):
    _, url = server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(url + "/api/state", headers={"Host": "evil.example:80"}), timeout=10)
    assert e.value.code == 403
    req = urllib.request.Request(url + "/api/command", b'{"action":"clear"}', {"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 415
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"action": "rm -rf"})
    assert e.value.code == 400


def test_server_commands_edit_transcript_and_practice_does_not_add(server):
    engine, url = server
    post(url, {"action": "clear"})
    st = post(url, {"action": "insert_word", "name": "thankyou"})
    assert st["transcript"]["text"] == "thank you "
    st = post(url, {"action": "punct", "char": "!"})
    assert st["transcript"]["text"] == "thank you! "
    assert post(url, {"action": "undo"})["transcript"]["text"] == "thank you "

    # A recognised word: added normally, scored-only in practice, swappable.
    ev = dict(outcome="accepted", reason="lull", duration=1.0, top=[("pen", 0.6), ("pencil", 0.3)])
    with engine.cond:
        engine._word_event(ev)
    assert engine.transcript.text == "thank you pen "
    st = post(url, {"action": "choose", "index": 1})
    assert st["transcript"]["text"] == "thank you pencil " and st["words"]["chosen"] == 1
    post(url, {"action": "practice", "on": True})
    with engine.cond:
        engine._word_event(ev)
    assert engine.transcript.text == "thank you pencil "
    assert engine.events[-1]["committed"] is False
    post(url, {"action": "practice", "on": False})

    st = post(url, {"action": "settings", "values": {"letters_hold": 1.2}})
    assert st["settings"]["letters_hold"] == 1.2
    assert (engine.state_dir / "settings.json").exists()
    assert post(url, {"action": "mode", "mode": "words"})["mode"] == "words"
