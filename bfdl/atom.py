#!/usr/bin/env python3

import re
from .tokens import TOK

class Atom:
    fileid:       int
    start_lineno: int
    start_column: int
    end_lineno:   int
    end_column:   int
    token:        TOK
    value:        str

    def __init__(self, fileid: int, start_lineno: int, start_column: int,
                 end_lineno: int, end_column: int, token: TOK, value: str):
        self.fileid       = fileid
        self.start_lineno = start_lineno
        self.start_column = start_column
        self.end_lineno   = end_lineno
        self.end_column   = end_column
        self.token        = token
        self.value        = value
