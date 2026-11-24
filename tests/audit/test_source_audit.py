"""Static audit of the source tree: no code execution, stray HTML, stray file writes or keys."""

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "datanaut"
APP = ROOT / "app.py"
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
SOURCES = sorted(SRC.rglob("*.py")) + [APP]
EVERY_PY = SOURCES + SCRIPTS


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def calls(path: Path):
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Call):
            yield node


def func_name(node: ast.Call) -> tuple[str | None, str | None]:
    """(object name, function name) of a call such as re.compile(...) or eval(...)."""
    f = node.func
    if isinstance(f, ast.Name):
        return None, f.id
    if isinstance(f, ast.Attribute):
        owner = f.value.id if isinstance(f.value, ast.Name) else None
        return owner, f.attr
    return None, None


# --- no code execution ---------------------------------------------------------------------


def test_there_is_no_eval_exec_or_string_query_in_the_source_tree():
    offences = []
    for path in EVERY_PY:
        for node in calls(path):
            owner, name = func_name(node)
            if name in {"eval", "exec", "__import__"}:
                offences.append(f"{rel(path)}:{node.lineno} {name}()")
            if name == "compile" and owner != "re":
                offences.append(f"{rel(path)}:{node.lineno} compile()")
            if name == "query" and isinstance(node.func, ast.Attribute):
                offences.append(f"{rel(path)}:{node.lineno} .query()")
    assert offences == []


def test_dangerous_modules_are_not_imported_by_the_app():
    banned = {"subprocess", "pickle", "marshal", "importlib", "ctypes", "tempfile", "shutil"}
    offences = []
    for path in SOURCES:
        for node in ast.walk(tree(path)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            offences += [f"{rel(path)}:{node.lineno} import {n}" for n in names if n in banned]
    assert offences == []


# --- HTML -----------------------------------------------------------------------------------


def test_unsafe_allow_html_is_used_only_for_the_stylesheet_and_the_escaped_dashboard():
    sites = []
    for path in SOURCES:
        for node in calls(path):
            if any(k.arg == "unsafe_allow_html" for k in node.keywords):
                sites.append((rel(path), node))
    assert [name for name, _ in sites] == ["app.py", "app.py"]  # nowhere else in the UI
    first, second = (n for _, n in sites)
    # the only things ever passed are our own stylesheet and the dashboard strip
    assert ast.unparse(first.args[0]) == "theme.style_tag()"
    assert ast.unparse(second.args[0]) == "board"
    assert ast.unparse(second.func).endswith("markdown")


def test_every_piece_of_text_in_the_dashboard_is_html_escaped():
    theme = (SRC / "ui" / "theme.py").read_text(encoding="utf-8")
    overview = (SRC / "ui" / "overview.py").read_text(encoding="utf-8")
    assert "import html" in theme and "html.escape" in theme
    for path in (SRC / "ui" / "theme.py",):
        for fn in tree(path).body:
            if isinstance(fn, ast.FunctionDef) and fn.name.endswith("_card"):
                body = ast.unparse(fn)
                assert "_esc(label)" in body and "_esc(hint)" in body, fn.name
    assert "<" not in overview.replace("<<", "")  # overview builds no markup of its own


# --- file writes -----------------------------------------------------------------------------

WRITERS = {"write_text", "write_bytes", "to_excel", "to_parquet", "to_pickle", "to_feather",
           "mkdir", "touch", "unlink", "rmdir", "dump", "savefig", "save"}  # fmt: skip
ALLOWED_WRITERS = {
    "src/datanaut/core/export.py",  # write_safe_csv: the export helper
    "src/datanaut/core/validation.py",  # RuleSet.save_json
    "src/datanaut/cli.py",  # command-line outputs to a path the user chose
}


def write_calls(path: Path) -> list[str]:
    found = []
    for node in calls(path):
        owner, name = func_name(node)
        where = f"{rel(path)}:{node.lineno}"
        if name == "open" and owner is None:
            mode = (
                node.args[1]
                if len(node.args) > 1
                else next((k.value for k in node.keywords if k.arg == "mode"), None)
            )
            if mode is None:
                continue
            if not (isinstance(mode, ast.Constant) and not set("wax+") & set(str(mode.value))):
                found.append(f"{where} open(mode)")
        elif name in WRITERS and isinstance(node.func, ast.Attribute):
            if name == "savefig" and node.args and ast.unparse(node.args[0]) == "buffer":
                continue  # a chart rendered into memory
            found.append(f"{where} .{name}()")
        elif name in {"to_csv", "to_json"} and isinstance(node.func, ast.Attribute):
            if node.args or any(k.arg == "path_or_buf" for k in node.keywords):
                found.append(f"{where} .{name}(path)")
        elif name == "output" and isinstance(node.func, ast.Attribute) and node.args:
            found.append(f"{where} .output(path)")
    return found


def test_files_are_only_written_by_the_export_helpers_and_the_command_line():
    offences = []
    for path in SOURCES:
        if rel(path) not in ALLOWED_WRITERS:
            offences += write_calls(path)
    assert offences == []


def test_the_allowed_writers_really_are_the_only_ones_that_write():
    writers = {rel(p) for p in SOURCES if write_calls(p)}
    assert writers == ALLOWED_WRITERS  # a stale entry in the allowlist would hide a gap


def test_the_ui_and_the_report_never_touch_the_filesystem_for_output():
    for path in [*sorted((SRC / "ui").rglob("*.py")), *sorted((SRC / "report").rglob("*.py")),
                 *sorted((SRC / "ai").rglob("*.py")), APP]:  # fmt: skip
        assert write_calls(path) == [], rel(path)


# --- keys and ignore rules -------------------------------------------------------------------

KEY_PATTERNS = {
    "google key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "groq key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "openai-style key": re.compile(r"sk-[A-Za-z0-9]{32,}"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "key assignment": re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*[\"'][A-Za-z0-9_\-]{24,}[\"']"
    ),
}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis",
             "htmlcov", "build", "dist", "node_modules"}  # fmt: skip
NEVER_OPEN = {".streamlit/secrets.toml", ".env"}  # a real secrets file is not read by tests
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".md",
    ".json",
    ".example",
    ".cfg",
    ".ini",
}


def repo_text_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        parts = set(path.relative_to(ROOT).parts)
        if not path.is_file() or parts & SKIP_DIRS or any(p.endswith(".egg-info") for p in parts):
            continue
        if rel(path) in NEVER_OPEN or path.suffix not in TEXT_SUFFIXES:
            continue
        out.append(path)
    return out


def test_no_api_key_is_written_into_any_file_of_the_project():
    files = repo_text_files()
    assert len(files) > 50
    hits = []
    for path in files:
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            for label, pattern in KEY_PATTERNS.items():
                if pattern.search(line):
                    hits.append(
                        f"{rel(path)}:{number} looks like a {label}"
                    )  # never the text itself
    assert hits == []


def test_the_key_patterns_do_catch_a_key_and_do_not_flag_placeholders():
    assert KEY_PATTERNS["google key"].search("AIza" + "A" * 35)
    assert KEY_PATTERNS["groq key"].search("gsk_" + "a1" * 15)
    assert KEY_PATTERNS["key assignment"].search('api_key = "' + "x" * 30 + '"')
    example = (ROOT / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8")
    assert not any(p.search(example) for p in KEY_PATTERNS.values())


def test_secrets_and_local_files_are_ignored_by_git():
    lines = {ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert {".streamlit/secrets.toml", ".env", ".claude/"} <= lines
    assert ".streamlit/secrets.toml.example" not in lines  # the template is meant to be committed


def test_the_project_dependencies_are_what_pyproject_says():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = {re.split(r"[<>=!~\[ ]", d)[0].lower() for d in project["dependencies"]}
    assert {"fpdf2", "matplotlib", "streamlit", "plotly"} <= runtime
    dev = {re.split(r"[<>=!~\[ ]", d)[0].lower() for d in project["optional-dependencies"]["dev"]}
    assert "pypdf" in dev and "pip-audit" not in runtime | dev
