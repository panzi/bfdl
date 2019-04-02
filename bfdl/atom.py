#!/usr/bin/env python3

from .tokens import TOK

class Atom:
    fileid:       int
    start_lineno: int
    start_column: int
    end_lineno:   int
    end_column:   int
    token:        TOK
    value:        str

    def __init__(self, fileid:int, start_lineno:int, start_column:int,
                 end_lineno:int, end_column:int, token: TOK, value: str):
        self.fileid       = fileid
        self.start_lineno = start_lineno
        self.start_column = start_column
        self.end_lineno   = end_lineno
        self.end_column   = end_column
        self.token        = token
        self.value        = value

    def parse_value(self):
        # TODO
        if self.token == TOK.STR:
            raise NotImplementedError
        elif self.token == TOK.INT:
            val = self.value.lower()

            if val.startswith("0x"):
                return int(self.value, 16)

            if val.startswith("0o"):
                return int(self.value, 8)

            if val.startswith("0b"):
                return int(self.value, 2)

            return int(self.value, 10)
        elif self.token == TOK.FLOAT:
            return float(self.value)

        elif self.token == TOK.BYTES:
            raise NotImplementedError

        elif self.token == TOK.BYTE:
            raise NotImplementedError
        else:
            raise TypeError(f"cannot parse value of {self.token.name} token")
