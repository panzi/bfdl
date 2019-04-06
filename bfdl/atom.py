#!/usr/bin/env python3

import re
from .tokens import TOK

HEX = r'\\x[0-9a-fA-F]{2}'
ESC = r'\\["ntrvfb\\]'
UNI = r'\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{6}'

R_STR = re.compile(
    rf'^(\b[^a-zA-Z]+)?"((?:[^"\n\\]|{HEX}|{ESC}|{UNI})*)"',
    re.M | re.U)

R_STR_ELEM = re.compile(rf'({ESC})|({HEX}|{UNI})', re.M | re.U)

R_BYTES = re.compile(rf'\bb"((?:[^"\n\\]|{HEX}|{ESC})*)"', re.M | re.U)

R_BYTES_ELEM = re.compile(rf'({HEX})|({ESC})|(.)', re.M | re.U)

R_INT = re.compile(
    r'^(?:([-+]?[0-9]+)|([-+]?0x[0-9a-fA-F]+)|([-+]?0o[0-7]+)|([-+]?0b[0-1]+))(?:([ui])(8|16|32|64)\b)?$',
    re.M | re.U)

R_FLOAT = re.compile(
    r'([-+]?[0-9]+(?:\.[0-9]+|[eE][-+]?[0-9]+))(?:f(32|64)\b)?',
    re.M | re.U)

ESC_CHAR_MAP = {
    '\\n': '\n',
    '\\t': '\t',
    '\\r': '\r',
    '\\v': '\v',
    '\\f': '\f',
    '\\b': '\b',
    '\\"': '"',
    "\\'": "'",
    '\\\\': '\\',
}

ESC_BYTE_MAP = dict((esc, ord(char)) for esc, char in ESC_CHAR_MAP.items())

def _replace_str_elem(match):
    val = match.group(1)
    if val: # ESC
        return ESC_CHAR_MAP[val]

    # HEX / UNI
    val = match.group(2)
    return chr(int(val[2:], 16))

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
        if self.token == TOK.STR:
            match = R_STR.match(self.value)
            if match.group(1):
                # circular import workaround
                from .errors import IllegalStringPrefixError
                raise IllegalStringPrefixError(self)
            return R_STR_ELEM.sub(_replace_str_elem, match.group(2))

        elif self.token == TOK.INT:
            match = R_INT.match(self.value)
            signed_char = match.group(5)
            str_bits    = match.group(6)
            signed      = signed_char == 'i'
            bits        = int(str_bits) if str_bits is not None else None

            str_val = match.group(0)
            if str_val is not None:
                value = int(str_val, 10)

            else:
                str_val = match.group(1)
                if str_val is not None:
                    value = int(str_val, 16)
                else:
                    str_val = match.group(2)
                    if str_val is not None:
                        value = int(str_val, 8)
                    else:
                        value = int(str_val, 2)

            if value < 0 and not signed:
                # circular import workaround
                from .errors import IllegalTokenError
                raise IllegalTokenError(self, "unsigned integer cannot be negative")

            return value, signed, bits

        elif self.token == TOK.FLOAT:
            match    = R_FLOAT.match(self.value)
            val      = match.group(1)
            str_bits = match.group(2)
            bits     = int(str_bits) if str_bits is not None else None

            return float(val), bits

        elif self.token == TOK.BYTES:
            match = R_BYTES.match(self.value)
            body = match.group(1)
            buf = bytearray()
            index = 0
            end = len(body)
            while index < end:
                match = R_BYTES_ELEM.match(body, index)
                val = match.group(1) # HEX
                if val:
                    buf.append(int(val[2:], 16))
                else:
                    val = match.group(2) # ESC
                    if val:
                        buf.append(ESC_BYTE_MAP[val])
                    else:
                        val = match.group(3) # CHAR
                        buf.append(ord(val))
                index = match.end()

            return bytes(buf)

        elif self.token == TOK.BYTE:
            val = self.value[1:-1]
            if val.startswith('\\x'):
                return int(val[2:], 16)

            if val.startswith('\\'):
                return ESC_BYTE_MAP[val]

            return ord(val)

        elif self.token == TOK.NULL:
            return None

        else:
            raise TypeError(f"cannot parse value of {self.token.name} token")
