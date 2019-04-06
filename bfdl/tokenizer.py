#!/usr/bin/env python3

import re
from collections import OrderedDict
from .errors import IllegalTokenError
from .atom import Atom
from .tokens import TOK, IGNORABLE

REG_EXPS = OrderedDict([
    (TOK.COMMENT,   r'//[^\n]*\n|/\*(?:[^\*]|\n|\*(?!\/))*\*/'),
    (TOK.HASH,      r'#'),
    (TOK.BR_OPEN,   r'\['),
    (TOK.BR_CLOSE,  r'\]'),
    (TOK.PAR_OPEN,  r'\('),
    (TOK.PAR_CLOSE, r'\)'),
    (TOK.CUR_OPEN,  r'\{'),
    (TOK.CUR_CLOSE, r'\}'),
    (TOK.INT,       r'[-+]?(?:[0-9]+|0x[0-9a-fA-F]+|0o[0-7]+|0b[0-1]+)(?:[ui](?:8|16|32|64)\b)?'),
    (TOK.FLOAT,     r'[-+]?[0-9]+(?:\.[0-9]+|[eE][-+]?[0-9]+)(?:f32\b|f64\b)?'),
    (TOK.EQ,        r'=='),
    (TOK.NE,        r'!='),
    (TOK.BANG,      r'!'),
    (TOK.LSHIFT,    r'<<'),
    (TOK.RSHIFT,    r'>>'),
    (TOK.LE,        r'<='),
    (TOK.GE,        r'>='),
    (TOK.LT,        r'<'),
    (TOK.GT,        r'>'),
    (TOK.ADD,       r'\+'),
    (TOK.SUB,       r'-'),
    (TOK.MUL,       r'\*'),
    (TOK.DIV,       r'/'),
    (TOK.MOD,       r'%'),
    (TOK.OR,        r'\|\|'),
    (TOK.AND,       r'&&'),
    (TOK.BOR,       r'\|'),
    (TOK.BAND,      r'&'),
    (TOK.XOR,       r'\^'),
    (TOK.BNOT,      r'~'),
    (TOK.ASSIGN,    r'='),
    (TOK.BYTES,     r'\bb"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]))*"'),
    (TOK.BOOL,      r'\btrue\b|\bfalse\b'),
    (TOK.STRUCT,    r'\bstruct\b'),
    (TOK.IF,        r'\bif\b'),
    (TOK.ENUM,      r'\benum\b'),
    (TOK.IMPORT,    r'\bimport\b'),
    (TOK.FROM,      r'\bfrom\b'),
    (TOK.AS,        r'\bas\b'),
    (TOK.RAISE,     r'\braise\b'),
    (TOK.TYPE,      r'\btype\b'),
    (TOK.UNION,     r'\bunion\b'),
    (TOK.NULL,      r'\bnull\b'),
    (TOK.ID,        r'\b[_a-zA-Z][_a-zA-Z0-9]*\b'),
    (TOK.DOT,       r'\.'),
    (TOK.SEMICOL,   r';'),
    (TOK.COMMA,     r','),
    (TOK.QUEST,     r'\?'),
    (TOK.COLON,     r':'),
    (TOK.STR,       r'(?:\b[^a-zA-Z]+)?"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{6}))*"'),
    (TOK.BYTE,      r"'(?:[^'\n\\]|\\(?:x[0-9a-fA-F]{2}|['ntrvfb\\]))'"),
    (TOK.SPACE,     r'[\s\n]+'),
])

if __name__ == '__main__':
    import sys

    # self check
    for tok, reg in REG_EXPS.items():
        try:
            re.compile(reg, re.M | re.U)
        except re.error as err:
            print('error in', tok, 'regex:', err, file=sys.stderr)
            sys.exit(1)

TOKENS = list(REG_EXPS.keys())

TOKENIZER = re.compile('^' +
    r'|'.join(r'(?P<%s>%s)' % (tok.name, regex) for tok, regex in REG_EXPS.items()),
    re.M | re.U
)

def tokenize(source: str, fileid:int=-1):
    index  = 0
    lineno = 1
    column = 1
    eof    = len(source)
    while index < eof:
        match = TOKENIZER.match(source, index)
        if not match:
            raise IllegalTokenError(Atom(
                fileid, lineno, column, lineno, column + 1,
                TOK.ILLEGAL, source[index:index + 1]))

        end_index = match.end()

        token = TOK[match.lastgroup]
        value = source[index:end_index]

        start_lineno = lineno
        start_column = column

        newlines = value.count('\n')
        if newlines > 0:
            lineno += newlines
            column = len(value) - value.rindex('\n')
        else:
            column += len(value)

        end_lineno = lineno
        end_column = column

        if not token in IGNORABLE:
            yield Atom(fileid, start_lineno, start_column,
                       end_lineno, end_column, token, value)

        index = end_index

def print_atoms(atoms):
    for atom in atoms:
        print('%4d,%3d ... %4d,%3d: %-10s %r' % (
            atom.start_lineno, atom.start_column,
            atom.end_lineno, atom.end_column,
            atom.token.name, atom.value))

if __name__ == '__main__':
    import sys

    FILENAMES = sys.argv[1:]
    if FILENAMES:
        for filename in FILENAMES:
            with open(filename, "r") as fp:
                print(filename + ':')
                print_atoms(tokenize(fp.read()))
                print()
    else:
        print_atoms(tokenize(sys.stdin.read()))
