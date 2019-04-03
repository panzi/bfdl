#!/usr/bin/env python3

from contextlib import contextmanager
from typing import List, Optional, Iterator, Dict, Union
from os.path import abspath

from .tokens import TOK
from .tokenizer import tokenize
from .atom import Atom
from .errors import (
    UnexpectedEndOfFileError, ParserError, IllegalTokenError, AttributeRedeclaredError,
    UnbalancedParanthesesError, TypeNameConflictError, IllegalImportError
)

class Primitive:
    pytype: type
    size: int
    name: str

    def __init__(self, pytype: type, size: int, name: str):
        self.pytype = pytype
        self.size   = size
        self.name   = name

UINT8  = Primitive(int,   1, "uint8")
INT8   = Primitive(int,   1, "int8")
BYTE   = Primitive(int,   1, "byte")
UINT16 = Primitive(int,   2, "uint16")
INT16  = Primitive(int,   2, "int16")
UINT32 = Primitive(int,   4, "uint32")
INT32  = Primitive(int,   4, "int32")
UINT64 = Primitive(int,   8, "uint64")
INT64  = Primitive(int,   8, "int64")
BOOL   = Primitive(bool,  1, "bool")
FLOAT  = Primitive(float, 4, "float")
DOUBLE = Primitive(float, 8, "double")

class ASTNode:
    location: Atom

    def __init__(self, location: Atom):
        self.location = location

class TypeRef(ASTNode):
    name: str

class TypeDef(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

class Field(ASTNode):
    name: str
    type: str

class StructDef(TypeDef):
    fields: List[Field]

class Value(ASTNode):
    pass

class Identifier(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

class Attribute(ASTNode):
    name: str
    body: Union[Value | TypeRef | Identifier | None]

    def __init__(self, name: str, body: Union[Value | TypeRef | Identifier | None], location: Atom):
        super().__init__(location)
        self.name = name
        self.body = body

class Attributes:
    parent: Optional["Attributes"]
    defined_attrs: Dict[str, Attribute]

    def __init__(self,
                 defined_attrs: Dict[str, Attribute]=None,
                 parent: Optional["Attributes"]=None):
        self.defined_attrs = defined_attrs or {}
        self.parent = parent

    def __contains__(self, name: str):
        return name in self.defined_attrs or (self.parent is not None and name in self.parent)

    def __getitem__(self, name: str):
        if self.parent is None or name in self.defined_attrs:
            return self.defined_attrs[name]
        return self.parent[name]

    def declare(self, attr: Attribute):
        if attr.name in self.defined_attrs:
            other = self.defined_attrs[attr.name]
            raise AttributeRedeclaredError(attr.name, other.location, attr.location)
        self.defined_attrs[attr.name] = attr

class Module:
    fileid:     int
    filename:   str
    source:     str
    attributes: Attributes
    types: Dict[str, TypeDef]

    def __init__(self, fileid: int, filename: str, source: str, attributes: Optional[Attributes]=None):
        self.fileid     = fileid
        self.filename   = filename
        self.source     = source
        self.attributes = attributes or Attributes()
        self.types      = {}

class State:
    tokens: Iterator[Atom]
    current_token: Optional[Atom]
    module: Module

    def __init__(self, tokens: Iterator[Atom], module: Module):
        self.tokens = tokens
        self.module = module
        self.current_token = None

class Parser:
    module_map: Dict[str, int]
    modules:    List[Module]
    stack:      List[State]

    def __init__(self):
        self.module_map = {}
        self.modules    = []
        self.stack      = []

    def parse_file(self, filename: str) -> Module:
        filename = abspath(filename)
        with open(filename, "r") as stream:
            source = stream.read()
        return self.parse_source(source, filename)

    def parse_source(self, source: str, filename: str) -> Module:
        if filename in self.module_map:
            module = self.modules[self.module_map[filename]]
            if module.source != source:
                raise ParserError("re-parsing file with different source: %s" % filename)
            return module

        fileid = len(self.modules)
        tokens = tokenize(source, fileid)
        module = Module(fileid, filename, source)
        state  = State(tokens, module)

        self.module_map[filename] = fileid
        self.modules.append(module)
        self.stack.append(state)

        self._parse_module()

        self.stack.pop()

        return module

    def _next_token(self) -> Atom:
        state = self.stack[-1]
        token = state.current_token
        if token is None:
            try:
                token = next(state.tokens)
            except StopIteration:
                raise UnexpectedEndOfFileError(self._make_eof_token())
            else:
                return token

        state.token = None
        return token

    def _peek_token(self) -> Optional[Atom]:
        state = self.stack[-1]
        token = state.current_token
        if token is None:
            try:
                token = next(state.tokens)
            except StopIteration:
                return None
            else:
                state.token = token
                return token

        return token

    def _has_next(self, tok: Optional[Atom]=None, val: Optional[str]=None) -> bool:
        token = self._next_token()

        if tok is not None and token.token != tok:
            return False

        if val is not None and token.value != val:
            return False

        return True

    def _make_eof_token(self) -> Atom:
        state = self.stack[-1]
        source = state.source
        lineno = source.count("\n") + 1
        column = len(source) - source.rindex("\n")
        return Atom(state.module.fileid, lineno, column, lineno, column, TOK.EOF, '')

    def _expect(self, tok: Optional[Atom]=None, val: Optional[str]=None) -> Atom:
        token = self._next_token()

        if tok is not None and token.token != tok:
            raise IllegalTokenError(token)

        if val is not None and token.value != val:
            raise IllegalTokenError(token)

        return token

    def _parse_module(self):
        while self._has_next(TOK.BANG):
            self._parse_file_attribute()

        while self._has_next(TOK.IMPORT):
            self._parse_import()

        while self._has_next():
            self._parse_struct()

    def _parse_file_attribute(self):
        self._expect(TOK.BANG)
        attr = self._parse_attribute()
        state = self.stack[-1]
        state.module.attributes.declare(attr)

    def _parse_attribute(self):
        self._expect(TOK.HASH)
        with self._expect_paren(TOK.CUR_OPEN):
            name_tok = self._expect(TOK.ID)

            if self._has_next(TOK.ASSIGN):
                self._next_token()

                if self._has_next(TOK.ID):
                    body_tok = self._next_token()
                    value = Identifier(body_tok.value, body_tok)
                elif self._has_next(TOK.TYPE):
                    value = self._parse_type()
                else:
                    value = self._parse_value()
            else:
                value = None

        attr = Attribute(name_tok.value, value, name_tok)
        return attr

    @contextmanager
    def _expect_paren(self, paren: TOK):
        open_tok = self._expect(paren)
        yield open_tok
        self._expect_close(open_tok)

    def _expect_close(self, open_tok: Atom):
        close_tok = self._next_token()
        if close_tok.tok != open_tok.token:
            raise UnbalancedParanthesesError(open_tok, close_tok)
        return close_tok

    def _parse_import(self):
        state = self.stack[-1]
        self._expect(TOK.IMPORT)
        atom = self._peek_token()

        if atom.tok == TOK.STR:
            # import all types
            module_filename = atom.parse_value()
            module = self.parse_file(module_filename)
            for name, typedef in module.types:
                if name in state.module.types:
                    raise TypeNameConflictError(name, typedef.location, state.module.types[name].location)
                state.module.types[name] = typedef
        else:
            # import only explicitely listed types
            import_map = {}
            with self._expect_paren(TOK.CUR_OPEN):
                while not self._has_next(TOK.CUR_CLOSE):
                    name_tok = self._expect(TOK.ID)

                    if self._has_next(TOK.AS):
                        self._next_token()
                        mapped_name_tok = self._expect(TOK.ID)
                        name = mapped_name_tok.value

                        if name in state.module.types:
                            raise TypeNameConflictError(name, mapped_name_tok, state.module.types[name].location)

                        if name in import_map:
                            raise TypeNameConflictError(name, mapped_name_tok, import_map[name][1])

                        import_map[name] = (name_tok.value, mapped_name_tok)
                    else:
                        name = name_tok.value

                        if name in state.module.types:
                            raise TypeNameConflictError(name, name_tok, state.module.types[name].location)

                        if name in import_map:
                            raise TypeNameConflictError(name, name_tok, import_map[name][1])

                        import_map[name_tok.value] = (name_tok.value, name_tok)

                    if self._has_next(TOK.SEMICOL):
                        self._next_token()
                    else:
                        break

            self._expect(TOK.FROM)
            module_filename = self._expect(TOK.STR).parse_value()
            module = self.parse_file(module_filename)

            for name, (mapped_name, token) in import_map.items():
                if name not in module.types:
                    raise IllegalImportError(name, module_filename, token)
                state.module.types[mapped_name] = module.types[name]

        self._expect(TOK.SEMICOL)

    def _parse_struct(self):
        self._expect(TOK.STRUCT)
        name_tok = self._expect(TOK.ID)
        with self._expect_paren(TOK.CUR_OPEN):
            # TODO
            pass

        raise NotImplementedError

    def _parse_type(self):
        raise NotImplementedError

    def _parse_value(self):
        raise NotImplementedError

def parse_file(filename: str):
    return Parser().parse_file(filename)

def parse_string(source: str, filename: str):
    return Parser().parse_source(source, filename)
