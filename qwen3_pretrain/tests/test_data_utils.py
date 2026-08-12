import pytest

from data_utils import format_example, pack_sequences


class StubTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def test_format_example_without_input():
    ex = {"instruction": "指令", "input": "", "output": "回答"}
    assert format_example(ex) == "指令\n\n回答"


def test_format_example_with_input():
    ex = {"instruction": "指令", "input": "输入", "output": "回答"}
    assert format_example(ex) == "指令\n\n输入\n\n回答"


def test_pack_sequences_single_window():
    tok = StubTokenizer()
    texts = ["ab", "cd"]
    packed = pack_sequences(tok, texts, seq_len=4, eos_token_id=99)
    assert packed == [[97, 98, 99, 99]]


def test_pack_sequences_multiple_windows_and_tail_drop():
    tok = StubTokenizer()
    texts = ["abcdefgh", "ij"]
    packed = pack_sequences(tok, texts, seq_len=4, eos_token_id=99)
    assert packed == [[97, 98, 99, 100], [101, 102, 103, 104]]
