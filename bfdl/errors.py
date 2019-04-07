#!/usr/bin/env python3

from typing import Optional
from .cursor import Span
from .atom import Atom
from .tokens import CLOSE_PARENS, TOK

class BFDLError(Exception):
    pass

class ParserError(BFDLError):
    pass

class TypeUnificationError(BFDLError):
    location1: Span
    location2: Span
    type1: str
    type2: str

    def __init__(self, location1: Span, location2: Span, type1: str, type2: str, message: Optional[str] = None):
        super().__init__(message or f"cannot unify types {type1} and {type2}")
        self.location1 = location1
        self.location2 = location2
        self.type1 = type1
        self.type2 = type2

class UndeclaredTypeError(BFDLError):
    name: str
    location: Span

    def __init__(self, name: str, location: Span, message: Optional[str] = None):
        super().__init__(message or f"type {name} is undeclared")
        self.name     = name
        self.location = location

class UnexpectedEndOfFileError(ParserError):
    location: Span

    def __init__(self, location: Span, message: Optional[str] = None):
        super().__init__(message or "unexpected end of file")
        self.location = location

class IllegalTokenError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message: Optional[str] = None):
        super().__init__(message or f"unexpected {token.token.name} token: {token.value}")
        self.token = token

class IllegalImportError(ParserError):
    name: str
    fileid: int
    location: Span

    def __init__(self, name: str, fileid: int, location: Span, message: Optional[str] = None):
        super().__init__(message or f"type {name} was not found in imported module")
        self.name     = name
        self.fileid   = fileid
        self.location = location

class IllegalReferenceError(ParserError):
    name: str
    location: Span

    def __init__(self, name: str, location: Span, message: Optional[str] = None):
        super().__init__(message or f"field {name} was not found in this context")
        self.name     = name
        self.location = location

class FieldAccessError(ParserError):
    name: str
    location: Span

    def __init__(self, name: str, location: Span, message: Optional[str] = None):
        super().__init__(message or f"tried to access field {name} in something that is not a struct")
        self.name     = name
        self.location = location

class ItemAccessError(ParserError):
    name: str

    def __init__(self, location: Span, message: Optional[str] = None):
        super().__init__(message or f"tried to an array element in something that is not an array")
        self.location = location

class NameConflictError(ParserError):
    name: str
    new_location: Span
    old_location: Span

    def __init__(self, name: str, new_location: Span, old_location: Span, message: Optional[str] = None):
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
    def __init__(self, name: str, new_location: Span, old_location: Span, message: Optional[str] = None):
        super().__init__(name, new_location, old_location, message or f'{name} redefined')

class UnbalancedParanthesesError(ParserError):
    open_at:   Span
    close_at:  Span
    open_tok:  TOK
    close_tok: TOK

    def __init__(self, open_tok: TOK, close_tok: TOK, open_at: Span, close_at: Span, message: Optional[str] = None):
        super().__init__(message or f'expected {CLOSE_PARENS[open_tok].value}, but got {close_tok.value}')
        self.open_at   = open_at
        self.close_at  = close_at
        self.open_tok  = open_tok
        self.close_tok = close_tok

class BFDLTypeError(BFDLError):
    location: Span
    source: str
    target: str

    def __init__(self, source: str, target: str, location: Span, message: Optional[str] = None):
        super().__init__(message or f"{source} is not assignable to {target}")
        self.source = source
        self.target = target
