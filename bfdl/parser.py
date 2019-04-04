#!/usr/bin/env python3

from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Optional, Iterator, Dict, Union
from os.path import abspath

from .tokens import TOK
from .tokenizer import tokenize
from .atom import Atom
from .errors import (
    UnexpectedEndOfFileError, ParserError, IllegalTokenError, AttributeRedeclaredError,
    UnbalancedParanthesesError, TypeNameConflictError, IllegalImportError, FieldRedeclaredError,
)

class ASTNode:
    location: Atom

    def __init__(self, location: Atom):
        self.location = location

class TypeRef(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

class Expr(ASTNode):
    pass

class ArrayTypeRef(TypeRef):
    item_ref: TypeRef
    size:     Optional[Expr]

    def __init__(self, location: Atom, item_ref: TypeRef, size: Optional[Expr]):
        super().__init__(item_ref.name + '[]', location)
        self.item_ref = item_ref
        self.size     = size

class TypeDef(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

class Primitive(TypeDef):
    pytype: type
    size: int
    name: str

    def __init__(self, pytype: type, size: int, name: str, location: Optional[Atom] = None):
        super().__init__(name, location or Atom(0, 0, 0, 0, 0, TOK.TYPE, name))
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

PRIMITIVES = dict((tp.name, tp) for tp in [
    UINT8, INT8, BYTE, UINT16, INT16, UINT32, INT32,
    UINT64, INT64, BOOL, FLOAT, DOUBLE,
])

class Value(ASTNode):
    pass

class FieldDef(ASTNode):
    name:       str
    type_ref:   TypeRef
    optional:   bool
    default:    Optional[Value]
    attributes: "Attributes"

    def __init__(self, location: Atom, name: str, type_ref: TypeRef,
                 default: Optional[Value], optional: bool, attributes: "Attributes"):
        super().__init__(location)
        self.name       = name
        self.type_ref   = type_ref
        self.optional   = optional
        self.default    = default
        self.attributes = attributes

    @property
    def fixed(self):
        return self.attributes.get('fixed', False)

class Section(ASTNode):
    pass

class UnconditionalSection(Section):
    start_field_index: int
    field_count: int

    def __init__(self, location: Atom, start_field_index: int, field_count: int):
        super().__init__(location)
        self.start_field_index = start_field_index
        self.field_count = field_count

class ConditionalSection(Section):
    condition: Expr
    sections: List[Section]

    def __init__(self, location: Atom, condition: Expr, sections: List[Section]):
        super().__init__(location)
        self.condition = condition
        self.sections  = sections

class StructDef(TypeDef):
    fields:   List[FieldDef]
    size:     Optional[int]
    sections: List[Section]

    def __init__(self, name: str, location: Atom, fields: List[FieldDef]=None,
                 size: Optional[int]=None, sections: List[Section]=None):
        super().__init__(name, location)
        self.fields   = fields if fields is not None else []
        self.size     = size
        self.sections = sections if sections is not None else []

class Identifier(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

class Attribute(ASTNode):
    name: str
    value: Union[Value | TypeRef | Identifier | None]

    def __init__(self, name: str, value: Union[Value | TypeRef | Identifier | None], location: Atom):
        super().__init__(location)
        self.name  = name
        self.value = value

class Attributes:
    parent: Optional["Attributes"]
    defined_attrs: Dict[str, Attribute]

    def __init__(self,
                 defined_attrs: Dict[str, Attribute]=None,
                 parent: Optional["Attributes"]=None):
        self.defined_attrs = defined_attrs or {}
        self.parent = parent

    def __contains__(self, name: str) -> bool:
        return name in self.defined_attrs or (self.parent is not None and name in self.parent)

    def __getitem__(self, name: str) -> Attribute:
        if self.parent is None or name in self.defined_attrs:
            return self.defined_attrs[name]
        return self.parent[name]

    def get(self, name: str, default=None):
        if name in self.defined_attrs:
            return self.defined_attrs[name]

        if self.parent is not None and name in self.parent:
            return self.parent[name]

        return default

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
    types:          Dict[str, TypeDef] # imported and declared
    declared_types: Dict[str, TypeDef]

    def __init__(self, fileid: int, filename: str, source: str, attributes: Optional[Attributes]=None):
        self.fileid     = fileid
        self.filename   = filename
        self.source     = source
        self.attributes = attributes or Attributes()
        self.types      = {}

    def declare(self, name: str, typedef: TypeDef):
        if name in self.types:
            raise TypeNameConflictError(name, typedef.location, self.types[name].location)
        self.declared_types[name] = self.types[name] = typedef

    def import_type(self, name: str, typedef: TypeDef):
        if name in self.types:
            raise TypeNameConflictError(name, typedef.location, self.types[name].location)
        self.types[name] = typedef

    def declare_all(self, types: Dict[str, TypeDef]):
        for name, typedef in types.items():
            self.declare(name, typedef)

    def import_all(self, types: Dict[str, TypeDef]):
        for name, typedef in types.items():
            self.import_type(name, typedef)

class State:
    tokens: Iterator[Atom]
    current_token: Optional[Atom]
    module: Module

    def __init__(self, tokens: Iterator[Atom], module: Module):
        self.tokens = tokens
        self.module = module
        self.current_token = None

PRELUDE = Module(0, '<prelude>', '')
PRELUDE.declare_all(PRIMITIVES)

class Parser:
    module_map: Dict[str, int]
    modules:    List[Module]
    stack:      List[State]

    def __init__(self):
        self.module_map = {0: PRELUDE}
        self.modules    = [PRELUDE]
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
            self._parse_struct_def()

        # TODO: typecheck phase and calculate struct sizes

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
                    value = self._parse_type_ref()
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
            for name, typedef in module.types.items():
                state.module.import_type(name, typedef)
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

                        import_map[name] = name_tok.value
                    else:
                        name = name_tok.value

                        if name in state.module.types:
                            raise TypeNameConflictError(name, name_tok, state.module.types[name].location)

                        if name in import_map:
                            raise TypeNameConflictError(name, name_tok, import_map[name][1])

                        import_map[name_tok.value] = name_tok.value

                    if self._has_next(TOK.SEMICOL):
                        self._next_token()
                    else:
                        break

            self._expect(TOK.FROM)
            module_filename = self._expect(TOK.STR).parse_value()
            module = self.parse_file(module_filename)

            for name, mapped_name in import_map.items():
                state.module.import_type(mapped_name, module.types[name])

        self._expect(TOK.SEMICOL)

    def _parse_struct_def(self):
        state = self.stack[-1]

        attrs = Attributes(parent=state.module.attributes)
        while self._has_next(TOK.HASH):
            attr = self._parse_attribute()
            attrs.declare(attr)

        self._expect(TOK.STRUCT)
        name_tok = self._expect(TOK.ID)

        name = name_tok.value
        if name in state.module.types:
            raise TypeNameConflictError(name, name_tok, state.module.types[name].location)

        fields   = OrderedDict()
        stack    = []
        section  = None
        sections = []

        with self._expect_paren(TOK.CUR_OPEN):
            while True:
                if self._has_next(TOK.IF):
                    section = None
                    sections.append(self._parse_conditional_section(fields, stack))
                elif self._has_next(TOK.CUR_CLOSE):
                    break
                else:
                    field = self._parse_field_def()
                    if section is None:
                        section = UnconditionalSection(field.location, len(fields), 0)
                        sections.append(section)

                    if field.name in fields:
                        raise FieldRedeclaredError(field.name, field.location, fields[field.name].location)

                    fields[field.name] = field
                    section.field_count += 1

        # TODO: resolve size after typecheck phase
        struct_def = StructDef(name, name_tok, list(fields.values()), None, sections)
        state.module.declare(name, struct_def)

    def _parse_conditional_section(self, fields: Dict[str, FieldDef], stack: List[Expr]):
        location = self._expect(TOK.IF)
        sections = []
        section  = None

        with self._expect_paren(TOK.PAR_OPEN):
            condition = self._parse_expr()

        stack.append(condition)
        with self._expect_paren(TOK.CUR_OPEN):
            while True:
                if self._has_next(TOK.IF):
                    section = None
                    sections.append(self._parse_conditional_section(fields, stack))
                elif self._has_next(TOK.CUR_CLOSE):
                    break
                else:
                    field = self._parse_field_def()
                    if section is None:
                        section = UnconditionalSection(field.location, len(fields), 0)
                        sections.append(section)

                    if field.name in fields:
                        raise FieldRedeclaredError(field.name, field.location, fields[field.name].location)

                    fields[field.name] = field
                    section.field_count += 1
        stack.pop()

        return ConditionalSection(location, condition, sections)

    def _parse_field_def(self):
        attrs = Attributes()
        while self._has_next(TOK.HASH):
            attr = self._parse_attribute()
            attrs.declare(attr)

        type_ref = self._parse_type_ref()
        optional = False
        if self._has_next(TOK.QUEST):
            self._next_token()
            optional = True
        name_tok = self._expect(TOK.ID)

        if self._has_next(TOK.ASSIGN):
            self._next_token()
            value = self._parse_value()
        else:
            value = None

        self._expect(TOK.SEMICOL)

        return FieldDef(name_tok, name_tok.name, type_ref, value, optional, attrs)

    def _parse_type_ref(self):
        name_tok = self._expect(TOK.ID)
        type_ref = TypeRef(name_tok.value, name_tok)

        while self._has_next(TOK.BR_OPEN):
            with self._expect_paren(TOK.BR_OPEN):
                if not self._has_next(TOK.BR_CLOSE):
                    size = self._parse_expr()
                else:
                    size = None
            type_ref = ArrayTypeRef(name_tok, type_ref, size)

        return type_ref

    def _parse_value(self):
        raise NotImplementedError

    def _parse_expr(self):
        raise NotImplementedError

def parse_file(filename: str):
    return Parser().parse_file(filename)

def parse_string(source: str, filename: str):
    return Parser().parse_source(source, filename)
