"""
Enums for the models module.

This module contains all the enumeration classes used by the Tortoise ORM models.
"""

from enum import Enum


class SessionTypeEnum(str, Enum):
    """Enumeration for session types."""
    CHATBOT = "CHATBOT"

    @classmethod
    def choices(cls):
        """Return choices in Django-style format for compatibility."""
        return [(item.value, item.value) for item in cls]


class UserTypeConstants(str, Enum):
    """Enumeration for user types in chat interactions."""
    HUMAN = "Human"
    AI = "A.I"

    @classmethod
    def choices(cls):
        """Return choices in Django-style format for compatibility."""
        return [(item.value, item.value) for item in cls]