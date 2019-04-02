#!/usr/bin/env python3

from enum import Enum

class TOK(Enum):
    COMMENT    = '//'
    HASH       = '#'
    BR_OPEN    = '['
    BR_CLOSE   = ']'
    PAR_OPEN   = '('
    PAR_CLOSE  = ')'
    CUR_OPEN   = '{'
    CUR_CLOSE  = '}'
    EQ         = '=='
    NE         = '!='
    BANG       = '!'
    LSHIFT     = '<<'
    RSHIFT     = '>>'
    LE         = '<='
    GE         = '>='
    LT         = '<'
    GT         = '>'
    ADD        = '+'
    SUB        = '-'
    MUL        = '*'
    DIV        = '/'
    MOD        = '%'
    OR         = '||'
    AND        = '&&'
    BOR        = '|'
    BAND       = '&'
    XOR        = '^'
    BNOT       = '~'
    ASSIGN     = '='
    ID         = 'id'
    INT        = '0'
    FLOAT      = '0.0'
    BOOL       = 'bool'
    STRUCT     = 'struct'
    IF         = 'if'
    ENUM       = 'enum'  # reserved for future use
    TYPE       = 'type'  # reserved for future use
    UNION      = 'union' # reserved for future use
    NULL       = 'null'  # reserved for future use
    IMPORT     = 'import'
    FROM       = 'from'
    AS         = 'as'
    RAISE      = 'raise'
    DOT        = '.'
    SEMICOL    = ';'
    COMMA      = ','
    QUEST      = '?'
    COLON      = ':'
    STR        = '""'
    BYTE       = "''"
    BYTES      = 'b""'
    SPACE      = ' '
    ILLEGAL    = '<illegal>'
    EOF        = '<eof>'

IGNORABLE = (TOK.COMMENT, TOK.SPACE)

OPEN_PARENS = {
    TOK.PAR_CLOSE: TOK.PAR_OPEN,
    TOK.BR_CLOSE:  TOK.BR_OPEN,
    TOK.CUR_CLOSE: TOK.CUR_OPEN,
}

CLOSE_PARENS = dict((val, key) for key, val in OPEN_PARENS.items())
