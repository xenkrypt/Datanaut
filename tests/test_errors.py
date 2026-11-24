import pytest

from datanaut.errors import (
    DatanautError,
    FileTooLargeError,
    ParseError,
    UnsupportedFormatError,
)


@pytest.mark.parametrize("cls", [FileTooLargeError, UnsupportedFormatError, ParseError])
def test_subclasses_have_friendly_default_and_base(cls):
    err = cls()
    assert isinstance(err, DatanautError)
    assert str(err) == cls.default_message
    assert len(str(err)) > 20


def test_custom_message():
    assert str(ParseError("bad row 3")) == "bad row 3"
