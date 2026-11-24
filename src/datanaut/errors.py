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


class CleaningError(DatanautError):
    """A cleaning operation was given invalid parameters or data it cannot work on."""

    default_message = "This cleaning step could not be applied to the data."


class RuleSetError(DatanautError):
    """A validation rule or rule file is malformed or cannot be used."""

    default_message = "The validation rules are not valid."


class VersioningError(DatanautError):
    """A dataset version or recipe could not be found, rebuilt or read."""

    default_message = "The dataset version could not be rebuilt."


class RecipeMismatchError(VersioningError):
    """Replaying a recipe did not reproduce the fingerprints it was made with."""

    default_message = "Replaying the recipe produced different data than it was made with."


class AnalysisError(DatanautError):
    """An analysis could not be run on this data (too short, wrong columns, no time index)."""

    default_message = "This analysis could not be run on the data."


class WeatherError(AnalysisError):
    """Weather data could not be requested, read or lined up with the series."""

    default_message = "The weather data could not be used."
