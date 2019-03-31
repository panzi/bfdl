#!/usr/bin/env python3

from .tokens import TOK

class Atom:
    start_lineno: int
    start_column: int
    end_lineno:   int
    end_column:   int
    token:        TOK
    value:        str

    def __init__(self, start_lineno, start_column, end_lineno, end_column, token, value):
        self.start_lineno = start_lineno
        self.start_column = start_column
        self.end_lineno   = end_lineno
        self.end_column   = end_column
        self.token        = token
        self.value        = value