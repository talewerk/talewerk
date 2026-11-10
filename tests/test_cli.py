import pytest

from talewerk import __version__
from talewerk.cli import main
from talewerk.doctor import check_llm, check_python, run_checks


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "doctor" in capsys.readouterr().out


def test_python_check_passes():
    assert check_python().ok


def test_unreachable_llm_is_a_warning_not_a_failure():
    c = check_llm("http://127.0.0.1:9", timeout=0.5)
    assert not c.ok
    assert not c.required


def test_run_checks_returns_all():
    names = [c.name for c in run_checks("http://127.0.0.1:9")]
    assert "FFmpeg" in names and "Local LLM server" in names
