"""The publish check finds what it claims to: committed bytecode, private paths and names, in the tree and in history."""

import subprocess
from pathlib import Path

import pytest

from phost.testing import hygiene
from phost.testing.hygiene import Finding, check, check_history, check_tree, identities, load_deny, problems

HOME = "/home/someone"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test Author", "-c", "user.email=test@example.org",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


def commit(repo, files: dict, message="c"):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        git(repo, "add", "-f", name)
    git(repo, "commit", "-q", "-m", message)


def kinds(findings):
    return sorted({f.kind for f in findings})


def test_a_clean_repository_has_no_problems(repo):
    commit(repo, {"a.py": "print('hi')\n", "README.md": "# x\n"})
    assert problems(check(repo, deny=[], home=HOME)) == []


def test_tracked_bytecode_is_a_problem_by_name_even_though_it_is_binary(repo):
    commit(repo, {"pkg/__pycache__/m.cpython-312.pyc": b"\x00\x01binary", "pkg/m.py": "x = 1\n"})
    found = check_tree(repo, home=HOME)
    assert kinds(found) == ["generated-file"] and "__pycache__" in found[0].where


def test_a_file_removed_later_is_still_found_in_history_but_not_in_the_tree(repo):
    commit(repo, {"keep.py": "x = 1\n", "m.pyc": b"\x00junk"})
    git(repo, "rm", "-q", "-f", "m.pyc")
    git(repo, "commit", "-q", "-m", "drop it")
    assert check_tree(repo, home=HOME) == []
    assert kinds(check_history(repo, home=HOME)) == ["generated-file"]


def test_the_home_path_is_found_in_content_in_the_tree_and_in_history_after_it_is_removed(repo):
    commit(repo, {"a.py": f"PATH = '{HOME}/project/data'\n"})
    assert kinds(check_tree(repo, home=HOME)) == ["private-path"]
    commit(repo, {"a.py": "PATH = 'data'\n"}, "scrub it")
    assert check_tree(repo, home=HOME) == []
    assert kinds(check_history(repo, home=HOME)) == ["private-path"]


def test_a_home_path_that_only_shares_a_prefix_is_not_a_match(repo):
    commit(repo, {"a.py": "P = '/home/someone_else/x'\n"})
    assert check_tree(repo, home=HOME) == []


def test_denied_terms_come_from_a_file_outside_the_repository(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("private") / "deny.txt"
    outside.write_text("# names that stay private\nsecret-project\n\n\\bacme\\b\n")
    commit(repo, {"a.py": "# talks to the Secret-Project service\n", "b.py": "# acmedia is fine\n", "c.py": "# ACME rules\n"})
    found = check_tree(repo, deny=load_deny(outside), home=HOME)
    assert sorted(f.where for f in found) == ["a.py", "c.py"] and kinds(found) == ["denied-term"]
    assert check_tree(repo, home=HOME) == []                      # without the list nothing is flagged


def test_no_deny_file_means_no_deny_patterns(monkeypatch, tmp_path):
    monkeypatch.delenv(hygiene.DENY_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))                     # an empty home: no ~/.config/phost/deny.txt
    assert load_deny() == []


def test_the_environment_variable_names_the_deny_file(monkeypatch, tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("zebra\n")
    monkeypatch.setenv(hygiene.DENY_ENV, str(path))
    assert [p.pattern for p in load_deny()] == ["zebra"]


def test_history_scan_ignores_removed_lines_and_reports_each_problem_once(repo):
    commit(repo, {"a.py": "# zebra\n"})
    commit(repo, {"a.py": "# zebra\n# more\n"})
    deny = [__import__("re").compile("zebra")]
    git(repo, "rm", "-q", "-f", "a.py")
    git(repo, "commit", "-q", "-m", "gone")
    found = [f for f in check_history(repo, deny=deny, home=HOME) if f.kind == "denied-term"]
    assert len(found) == 1


def test_identities_are_notices_not_problems(repo):
    commit(repo, {"a.py": "x = 1\n"})
    found = identities(repo)
    assert [f.detail for f in found] == ["Test Author <test@example.org>"] and not found[0].is_problem
    assert problems(check(repo, deny=[], home=HOME)) == []


def test_the_command_line_exits_nonzero_for_a_problem_and_zero_for_notices(repo, capsys):
    commit(repo, {"a.py": "x = 1\n"})
    assert hygiene.main([str(repo), "--deny", str(repo / "nothing.txt")]) == 0
    commit(repo, {"m.pyc": b"\x00"})
    assert hygiene.main([str(repo), "--no-history"]) == 1
    assert "generated-file" in capsys.readouterr().out


def test_history_findings_name_the_commit_not_the_hunk_header(repo):
    commit(repo, {"a.py": "# zebra\n"})
    found = check_history(repo, deny=[__import__("re").compile("zebra")], home=HOME)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert [f.where for f in found] == [f"{sha}:a.py"]
