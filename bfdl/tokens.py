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
    IF         = 'if'     # reserved for future use
    ENUM       = 'enum'   # reserved for future use
    IMPORT     = 'import' # reserved for future use
    FROM       = 'from'   # reserved for future use
    RAISE      = 'raise'  # reserved for future use
    DOT        = '.'
    SEMICOL    = ';'
    COMMA      = ','
    QUEST      = '?'
    COLON      = ':'
    STR        = '""'
    BYTE       = "''"
    BYTES      = 'b""'
    SPACE      = ' '
    ILLEGAL    = ''

IGNORABLE = (TOK.COMMENT, TOK.SPACE)
