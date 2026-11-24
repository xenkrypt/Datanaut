"""Exceptions raised by datanaut, each carrying a message safe to show to users."""

from __future__ import annotations


class DatanautError(Exception):
    """Base class for all datanaut errors."""

    default_message = "Something went wrong while processing your data."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class FileTooLargeError(DatanautError):
    """The input file exceeds the allowed size."""

    default_message = "This file is larger than the allowed size limit. Try a smaller file."


class UnsupportedFormatError(DatanautError):
    """The input is not a CSV or Excel (xlsx) file."""

    default_message = (
        "This file format is not supported. Please upload a CSV or Excel (.xlsx) file."
    )


class ParseError(DatanautError):
    """The file could not be read or parsed."""

    default_message = (
        "The file could not be read. It may be empty, corrupted or in an unexpected layout."
    )
