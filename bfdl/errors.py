#!/usr/bin/env python3

from typing import Optional
from .atom import Atom
from .tokens import CLOSE_PARENS

class BFDLError(Exception):
    pass

class ParserError(BFDLError):
    pass

class TypeUnificationError(BFDLError):
    location1: Atom
    location2: Atom
    type1: str
    type2: str

    def __init__(self, location1: Atom, location2: Atom, type1: str, type2: str, message:Optional[str]=None):
        super().__init__(message or f"cannot unify types {type1} and {type2}")
        self.location1 = location1
        self.location2 = location2
        self.type1 = type1
        self.type2 = type2

class UndeclaredTypeError(BFDLError):
    name: str
    token: Atom

    def __init__(self, name: str, token: Atom, message:Optional[str]=None):
        super().__init__(message or f"type {name} is undeclared")
        self.name  = name
        self.token = token

class UnexpectedEndOfFileError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:Optional[str]=None):
        super().__init__(message or "unexpected end of file")
        self.token = token

class IllegalTokenError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:Optional[str]=None):
        super().__init__(message or f"unexpected {token.token.name} token: {token.value}")

class IllegalStringPrefixError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:Optional[str]=None):
        super().__init__(message or f"illegal string prefix: {token.value}")
        self.token = token

class IllegalImportError(ParserError):
    name: str
    fileid: int
    location: Atom

    def __init__(self, name: str, fileid: int, location: Atom, message:Optional[str]=None):
        super().__init__(message or f"type {name} was not found in imported module")
        self.name     = name
        self.fileid   = fileid
        self.location = location

class IllegalReferenceError(ParserError):
    name: str
    location: Atom

    def __init__(self, name: str, location: Atom, message:Optional[str]=None):
        super().__init__(message or f"field {name} was not found in this context")
        self.name     = name
        self.location = location

class FieldAccessError(ParserError):
    name: str
    location: Atom

    def __init__(self, name: str, location: Atom, message:Optional[str]=None):
        super().__init__(message or f"tried to access field {name} in something that is not a struct")
        self.name     = name
        self.location = location

class ItemAccessError(ParserError):
    name: str

    def __init__(self, location: Atom, message:Optional[str]=None):
        super().__init__(message or f"tried to an array element in something that is not an array")
        self.location = location

class NameConflictError(ParserError):
    name: str
    new_location: Atom
    old_location: Atom

    def __init__(self, name: str, new_location: Atom, old_location: Atom, message: Optional[str]=None):
        super().__init__(message or f'{name} redeclared')
        self.new_location = new_location
        self.old_location = old_location

class TypeNameConflictError(NameConflictError):
    pass

class AttributeRedeclaredError(NameConflictError):
    pass

class FieldRedeclaredError(NameConflictError):
    pass

class FieldRedefinedError(NameConflictError):
    def __init__(self, name: str, new_location: Atom, old_location: Atom, message: Optional[str]=None):
        super().__init__(name, new_location, old_location, message or f'{name} redefined')

class UnbalancedParanthesesError(ParserError):
    open_token:  Atom
    close_token: Atom

    def __init__(self, open_token: Atom, close_token: Atom, message: Optional[str]=None):
        super().__init__(message or f'expected {CLOSE_PARENS[open_token].value}, but got {close_token.value}')
        self.open_token  = open_token
        self.close_token = close_token
