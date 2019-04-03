#!/usr/bin/env python3

from typing import Optional
from .atom import Atom
from .tokens import CLOSE_PARENS

class BFDLError(Exception):
    pass

class ParserError(BFDLError):
    pass

class UnexpectedEndOfFileError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:str=None):
        super().__init__(message or "unexpected end of file")
        self.token = token

class IllegalTokenError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:str=None):
        super().__init__(message or f"unexpected {token.token.name} token: {token.value}")

class IllegalStringPrefixError(ParserError):
    token: Atom

    def __init__(self, token: Atom, message:str=None):
        super().__init__(message or f"illegal string prefix: {token.value}")
        self.token = token

class IllegalImportError(ParserError):
    name: str
    module: str
    token: Atom

    def __init__(self, name: str, module: str, token: Atom, message:str=None):
        super().__init__(message or f"type {name} was not found in module {repr(module)}")
        self.name   = name
        self.module = module
        self.token  = token

class NameConflictError(ParserError):
    name: str
    location1: Atom
    location2: Atom

    def __init__(self, name: str, location1: Atom, location2: Atom, message: Optional[str]=None):
        super().__init__(message or f'{name} redeclared')
        self.location1 = location1
        self.location2 = location2

class TypeNameConflictError(NameConflictError):
    pass

class AttributeRedeclaredError(NameConflictError):
    pass

class UnbalancedParanthesesError(ParserError):
    open_token:  Atom
    close_token: Atom

    def __init__(self, open_token: Atom, close_token: Atom, message: Optional[str]=None):
        super().__init__(message or f'expected {CLOSE_PARENS[open_token].value}, but got {close_token.value}')
        self.open_token  = open_token
        self.close_token = close_token
