import random

from research.robustness.attacks import break_grammar, clean_generation, negate


def test_negate_rules():
    assert negate("A stack is LIFO.")["text"] == "A stack is FIFO."          # antonym before aux
    assert negate("A stack is a data structure.")["text"] == "A stack is not a data structure."
    assert negate("The bulb does not light.")["text"] == "The bulb does light."
    assert negate("It won't work")["text"] == "It will work"
    assert negate("Voltage increases with resistance")["text"] == "Voltage decreases with resistance"
    assert negate("Lists can only be read, it is slow")["text"] == "Lists can only be read, it is not slow"
    assert negate("hello world") is None


def test_break_grammar_is_deterministic_and_keeps_content():
    t = "The stack is a LIFO structure. Push adds to the top."
    a, b = break_grammar(t, random.Random(3)), break_grammar(t, random.Random(3))
    assert a == b
    for w in ["stack", "lifo", "structure", "push", "adds", "top"]:
        assert w in a.lower()


def test_clean_generation():
    assert clean_generation('Paraphrase: "A queue is FIFO."\n\nNote: blah') == "A queue is FIFO."
