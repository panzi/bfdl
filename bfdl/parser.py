#!/usr/bin/env python3

from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Optional, Iterator, Dict, Union, Set, Tuple
from os.path import abspath, join as join_path
from enum import Enum

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

class Import(ASTNode):
    fielid: int
    import_map: Optional[Dict[str, Tuple[TOK, TOK]]]

    def __init__(self, location: Atom, fileid: int, import_map: Optional[Dict[str, (str, Atom)]]=None):
        super().__init__(location)
        self.fielid     = fileid
        self.import_map = import_map

class ModuleState:
    LOADING  = 0
    LOADED   = 1
    FINISHED = 2

class Module:
    fileid:     int
    filename:   str
    source:     str
    state:      ModuleState
    attributes: Attributes
    imports:    List[Import]
    unfinished_refs: Set[int] # fileids of modules that import this module
    types:           Dict[str, TypeDef] # prelude, imported and declared
    declared_types:  Dict[str, TypeDef]

    def __init__(self, fileid: int, filename: str, source: str, attributes: Optional[Attributes]=None):
        self.fileid          = fileid
        self.filename        = filename
        self.source          = source
        self.state           = ModuleState.LOADING
        self.attributes      = attributes or Attributes()
        self.imports         = []
        self.unfinished_refs = set()
        self.types           = dict(PRIMITIVES)

    def declare(self, name: str, typedef: TypeDef):
        if name in self.types:
            raise TypeNameConflictError(name, typedef.location, self.types[name].location)
        self.declared_types[name] = self.types[name] = typedef

PRELUDE = Module(0, '<prelude>', '')
PRELUDE.state = ModuleState.FINISHED

class Parser:
    _root_path:      str
    _module_map:     Dict[str, int]
    _modules:        List[Module]
    _module_queue:   List[int]
    _tokens:         Iterator[Atom]
    _current_token:  Optional[Atom]
    _current_module: Module

    def __init__(self, root_path='.'):
        self._root_path      = abspath(root_path)
        self._module_map     = {0: PRELUDE}
        self._modules        = [PRELUDE]
        self._module_queue   = []
        self._tokens         = iter(())
        self._current_token  = None
        self._current_module = PRELUDE

    def parse_file(self, filename: str) -> Module:
        filename = join_path(self._root_path, filename)
        with open(filename, "r") as stream:
            source = stream.read()
        return self.parse_string(source, filename)

    def parse_string(self, source: str, filename: str) -> Module:
        fileid = self._queue_module(filename, source)
        module = self._modules[fileid]

        finish_queue = []
        while self._module_queue:
            other_fileid = self._module_queue[0]
            del self._module_queue[0]
            other_module = self._modules[other_fileid]

            self._tokens         = tokenize(other_module.source, other_fileid)
            self._current_token  = None
            self._current_module = other_module
            self._parse_module()
            self._current_module = PRELUDE

            other_module.state = ModuleState.LOADED

            finish_queue.append(other_module)

        finish_queue.append(module)

        index = 0
        while index < len(finish_queue):
            other_module = finish_queue[index]
            self._try_finish_module(other_module)

            for fileid in other_module.unfinished_refs:
                ref_module = self._modules[fileid]
                if ref_module.state is not ModuleState.FINISHED:
                    finish_queue.append(ref_module)
            index += 1

        return module

    def _try_finish_module(self, module: Module) -> bool:
        if module.state is ModuleState.FINISHED:
            return True

        # check if all dependencies have finished
        for imp in module.imports:
            imp_module = self._modules[imp.fielid]
            if imp_module.state is ModuleState.LOADING:
                return False

        # resolve imports
        for imp in module.imports:
            imp_module = self._modules[imp.fielid]
            if imp.import_map is not None:
                for mapped_name, (name_tok, mapped_name_tok) in imp.import_map.items():
                    if mapped_name in module.declared_types:
                        typedef = module.declared_types[mapped_name]
                        raise TypeNameConflictError(mapped_name, typedef.location, mapped_name_tok)

                    if name_tok.value not in imp_module.declared_types:
                        raise IllegalImportError(name_tok.value, imp.fileid, name_tok)

                    module.types[mapped_name] = imp_module.declared_types[name_tok.value]
            else:
                for name, imp_typedef in imp_module.declared_types.items():
                    if name in module.declared_types:
                        this_typedef = module.declared_types[name]
                        raise TypeNameConflictError(name, this_typedef.location, imp_typedef.location)

                    module.types[name] = imp_typedef

            imp_module.unfinished_refs.remove(module.fileid)

        module.state = ModuleState.FINISHED
        return True

    def _next_token(self) -> Atom:
        token = self._current_token
        if token is None:
            try:
                token = next(self._tokens)
            except StopIteration:
                raise UnexpectedEndOfFileError(self._make_eof_token())
            else:
                return token

        self._current_token = None
        return token

    def _peek_token(self) -> Optional[Atom]:
        token = self._current_token
        if token is None:
            try:
                token = next(self._tokens)
            except StopIteration:
                return None
            else:
                self._current_token = token
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
        module = self._current_module
        source = module.source
        lineno = source.count("\n") + 1
        column = len(source) - source.rindex("\n")
        return Atom(module.fileid, lineno, column, lineno, column, TOK.EOF, '')

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
        self._current_module.attributes.declare(attr)

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

    def _expect_close(self, open_tok: Atom) -> Atom:
        close_tok = self._next_token()
        if close_tok.tok != open_tok.token:
            raise UnbalancedParanthesesError(open_tok, close_tok)
        return close_tok

    def _queue_module(self, filename: str, source: Optional[str]=None) -> int:
        if self._current_module is not PRELUDE and (filename.startswith('./') or filename.startswith('../')):
            filename = join_path(self._current_module.filename, filename)
        else:
            filename = join_path(self._root_path, filename)

        if filename in self._module_map:
            fileid = self._module_map[filename]
            module = self._modules[fileid]
            if source is not None:
                if module.source != source:
                    raise ParserError("re-parsing file with different source: %s" % filename)
        else:
            if source is None:
                with open(filename, "r") as stream:
                    source = stream.read()
            fileid = len(self._modules)
            module = Module(fileid, filename, source)
            self._module_map[filename] = fileid
            self._modules.append(module)
            self._module_queue.append(fileid)

        if self._current_module is not PRELUDE:
            module.unfinished_refs.add(self._current_module.fileid)

        return fileid

    def _parse_import(self):
        self._expect(TOK.IMPORT)
        atom = self._peek_token()

        if atom.tok == TOK.STR:
            # import all types
            module_filename = atom.parse_value()
            fileid = self._queue_module(module_filename)
            self._current_module.imports.append(Import(atom, fileid))
        else:
            # import only explicitely listed types
            import_map:Dict[str, Tuple[TOK, TOK]] = OrderedDict()
            with self._expect_paren(TOK.CUR_OPEN):
                while not self._has_next(TOK.CUR_CLOSE):
                    name_tok = self._expect(TOK.ID)

                    if self._has_next(TOK.AS):
                        self._next_token()
                        mapped_name_tok = self._expect(TOK.ID)
                        name = mapped_name_tok.value

                        if name in import_map:
                            raise TypeNameConflictError(name, mapped_name_tok, import_map[name].location)

                        import_map[name] = (name_tok, mapped_name_tok)
                    else:
                        name = name_tok.value

                        if name in import_map:
                            raise TypeNameConflictError(name, name_tok, import_map[name].location)

                        import_map[name_tok.value] = (name_tok, name_tok)

                    if self._has_next(TOK.SEMICOL):
                        self._next_token()
                    else:
                        break

            self._expect(TOK.FROM)

            atom = self._expect(TOK.STR)
            module_filename = atom.parse_value()
            fileid = self._queue_module(module_filename)
            self._current_module.imports.append(Import(atom, fileid, import_map))

        self._expect(TOK.SEMICOL)

    def _parse_struct_def(self):
        attrs = Attributes(parent=self._current_module.attributes)
        while self._has_next(TOK.HASH):
            attr = self._parse_attribute()
            attrs.declare(attr)

        self._expect(TOK.STRUCT)
        name_tok = self._expect(TOK.ID)

        name = name_tok.value
        if name in self._current_module.declared_types:
            raise TypeNameConflictError(name, name_tok, self._current_module.types[name].location)

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
        self._current_module.declare(name, struct_def)

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

def parse_file(filename: str, root_path:str='.'):
    return Parser(root_path).parse_file(abspath(filename))

def parse_string(source: str, filename: str, root_path:str='.'):
    return Parser(root_path).parse_string(source, filename)
