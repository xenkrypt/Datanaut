from datanaut.core.loader import load_file
from datanaut.ui.state import STEPS, ResultCache, WizardState, cache_key


def _loaded(csv: bytes = b"t,v\n2024-01-01,1\n2024-01-02,2\n"):
    return load_file(csv, "a.csv")


def test_steps_are_gated_by_their_prerequisites():
    s = WizardState()
    assert s.is_reachable("Upload") and not s.is_reachable("Profile")
    assert [s.status(x) for x in STEPS[:3]] == ["current", "locked", "locked"]
    assert s.go("Profile") is False and s.step == "Upload"
    s.set_loaded(_loaded(), b"x", "d1")
    assert s.go("Profile") and s.go("Map columns")
    assert s.go("Clean") is False  # needs the mapping
    s.mark_done("Map columns")
    assert s.go("Analyze") and s.go("Upload")  # going back is always allowed
    assert s.go("nonsense") is False


def test_loading_a_new_file_discards_dependent_work_with_a_message():
    s = WizardState()
    s.set_loaded(_loaded(), b"x", "d1")
    s.cache.get_or_compute(("k",), lambda: 1)
    s.completed |= {"Map columns", "Clean"}
    s.time_columns = ["t"]
    s.set_loaded(_loaded(), b"y", "d2")
    assert s.completed == {"Upload"} and s.time_columns == [] and len(s.cache) == 0
    assert len(s.log) == len(_loaded().log) and s.source_digest == "d2"


def test_invalidating_the_mapping_keeps_the_file_and_trims_the_log():
    s = WizardState()
    s.set_loaded(_loaded(), b"x", "d1")
    s.mark_done("Profile")
    s.log.append("profile", "p", read_only=True)
    s.base_log_length = len(s.log)
    s.log.append("build_time_index", "i")
    s.completed |= {"Map columns", "Clean"}
    s.step = "Analyze"
    s.invalidate_mapping("The time column choice changed.")
    assert s.loaded is not None and s.store is None
    assert s.completed == {"Upload", "Profile"}
    assert [e.operation for e in s.log][-1] == "profile" and len(s.log) == s.base_log_length
    assert s.step == "Map columns"
    assert [m.level for m in s.take_messages()] == ["warning"] and s.messages == []


def test_cache_computes_once_and_stays_bounded():
    c = ResultCache(size=2)
    calls = []

    def make(v):
        calls.append(v)
        return v

    assert c.get_or_compute(("a",), lambda: make(1)) == 1
    assert c.get_or_compute(("a",), lambda: make(99)) == 1 and calls == [1]
    c.get_or_compute(("b",), lambda: make(2))
    c.get_or_compute(("a",), lambda: make(1))  # refreshes a
    c.get_or_compute(("c",), lambda: make(3))  # evicts b, the least recently used
    assert ("a",) in c and ("c",) in c and ("b",) not in c and len(c) == 2


def test_cache_key_depends_on_fingerprint_name_and_parameters_only():
    assert cache_key("f", "x", a=1, b=2) == cache_key("f", "x", b=2, a=1)
    assert cache_key("f", "x", a=1) != cache_key("g", "x", a=1)
    assert cache_key("f", "x", a=1) != cache_key("f", "x", a=2)
    assert hash(cache_key("f", "x", cols=["a", "b"])) is not None


def test_two_sessions_never_share_state():
    a, b = WizardState(), WizardState()
    a.set_loaded(_loaded(), b"x", "d1")
    a.add_message("info", "only a")
    a.cache.get_or_compute(("k",), lambda: 1)
    assert b.loaded is None and b.messages == [] and len(b.cache) == 0
    assert a.log is not b.log and a.completed is not b.completed
