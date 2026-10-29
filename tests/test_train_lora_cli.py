"""CPU-only argument checks; loading the training CLI must not import GPU libraries."""
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def training_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("training_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["unsloth", "standard", "off"])
def test_accepts_explicit_checkpointing_mode(training_cli, monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--adapter-dir", "synthetic-output",
                                    "--gradient-checkpointing", mode])
    assert training_cli.parse_args().gradient_checkpointing == mode


def test_checkpointing_defaults_to_historical_unsloth_mode(training_cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--adapter-dir", "synthetic-output"])
    assert training_cli.parse_args().gradient_checkpointing == "unsloth"


def test_rejects_unrecognized_checkpointing_mode(training_cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_lora.py", "--adapter-dir", "synthetic-output",
                                    "--gradient-checkpointing", "unknown"])
    with pytest.raises(SystemExit):
        training_cli.parse_args()
