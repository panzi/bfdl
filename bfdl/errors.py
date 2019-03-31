#!/usr/bin/env python3

from .atom import Atom

class BFDLError(Exception):
    pass

class IllegalTokenError(BFDLError):
    atom: Atom

    def __init__(self, atom):
        super().__init__("illegal token")
        self.atom = atom

class ParserError(BFDLError):
    start: Atom
    end: Atom

    def __init__(self, start, end, message):
        super().__init__(message)
        self.start = start
        self.end   = end

class UnexpectedTokenError(ParserError):
    def __init__(self, atom, message=None):
        super().__init__(atom, atom,
            "unexpected token: %s %r" % (atom.token.name, atom.value)
                if message is None else message)