#!/usr/bin/env python3

from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Optional, Iterator, Dict, Union, Set, Tuple, Any
from os.path import abspath, join as join_path, dirname
from enum import Enum

from .tokens import TOK
from .tokenizer import tokenize
from .atom import Atom
from .errors import (
    UnexpectedEndOfFileError, ParserError, IllegalTokenError, AttributeRedeclaredError,
    UnbalancedParanthesesError, TypeNameConflictError, IllegalImportError, FieldRedeclaredError,
    FieldRedefinedError, TypeUnificationError, UndeclaredTypeError, IllegalReferenceError,
    FieldAccessError, ItemAccessError,
)

class ASTNode:
    location: Atom

    def __init__(self, location: Atom):
        self.location = location

class TypeDef(ASTNode):
    name: str
    size: Optional[int]

    def __init__(self, name: str, size: Optional[int], location: Atom):
        super().__init__(location)
        self.name = name
        self.size = size

    def __hash__(self) -> int:
        return hash((self.location.fileid, self.name))

    def __eq__(self, other: TypeDef) -> bool:
        return (
            self.location.fileid == other.location.fileid and
            self.name == other.name
        )

class TypeRef(ASTNode):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

    def resolve(self, parser: "Parser", module: "Module") -> TypeDef:
        if self.name not in module.types:
            raise UndeclaredTypeError(self.name, self.location)
        return module.types[self.name]

def unify_types(lhs: TypeDef, rhs: TypeDef) -> Optional[TypeDef]:
    if lhs is rhs:
        return lhs

    if lhs is NEVER:
        return rhs

    if rhs is NEVER:
        return lhs

    lhs_nullable = isinstance(lhs, NullableType)
    rhs_nullable = isinstance(rhs, NullableType)

    if lhs_nullable and rhs_nullable:
        typedef = unify_types(lhs.typedef, rhs.typedef)
        return NullableType(typedef) if typedef is not None else None

    if lhs_nullable:
        typedef = unify_types(lhs.typedef, rhs)
        return NullableType(typedef) if typedef is not None else None

    if rhs_nullable:
        typedef = unify_types(lhs, rhs.typedef)
        return NullableType(typedef) if typedef is not None else None

    if not isinstance(lhs, PrimitiveType) or not isinstance(rhs, PrimitiveType):
        return None

    if lhs.pytype is int and rhs.pytype is int:
        if lhs.size > rhs.size:
            return lhs
        elif rhs.size > lhs.size:
            return rhs
        elif rhs is BYTE:
            return rhs
        else:
            return lhs

    if lhs.pytype is float and rhs.pytype is float:
        return rhs if rhs.size > lhs.size else lhs

    if lhs.pytype is float and rhs.pytype is int:
        return lhs

    if lhs.pytype is int and rhs.pytype is float:
        return rhs

    return None

class Expr(ASTNode):
    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        raise NotImplementedError

    def type_check(self, typedef: TypeDef):
        raise NotImplementedError

    def fold(self) -> "Expr":
        raise NotImplementedError

class ConditionalExpr(Expr):
    condition:  Expr
    true_expr:  Expr
    false_expr: Expr

    def __init__(self, condition: Expr, true_expr: Expr, false_expr: Expr, location: Atom):
        super().__init__(location)
        self.condition  = condition
        self.true_expr  = true_expr
        self.false_expr = false_expr

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        true_type  = self.true_expr.get_type_def(parser, module, context)
        false_type = self.false_expr.get_type_def(parser, module, context)
        typedef = unify_types(true_type, false_type)
        if typedef is None:
            raise TypeUnificationError(self.true_expr.location, self.false_expr.location, true_type, false_type)
        return typedef

COMPARE_OPS = frozenset((
    TOK.EQ, TOK.NE, TOK.LT, TOK.GT, TOK.LE, TOK.GE,
))

BIT_OPS = frozenset((
    TOK.BAND, TOK.BOR, TOK.XOR, TOK.LSHIFT, TOK.RSHIFT,
))

NUM_OPS = frozenset((
    TOK.ADD, TOK.SUB, TOK.MUL, TOK.DIV, TOK.MOD,
))

BOOL_OPS = frozenset((
    TOK.AND, TOK.OR,
))

BOOL_RES_OPS = COMPARE_OPS | BOOL_OPS

class BinaryExpr(Expr):
    lhs: Expr
    rhs: Expr
    op:  TOK

    def __init__(self, lhs: Expr, rhs: Expr, op: TOK, location: Atom):
        super().__init__(location)
        self.lhs = lhs
        self.rhs = rhs
        self.op  = op

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        if self.op in BOOL_RES_OPS:
            return BOOL

        lhs_type = self.lhs.get_type_def(parser, module, context)
        rhs_type = self.rhs.get_type_def(parser, module, context)
        typedef = unify_types(lhs_type, rhs_type)
        if typedef is None:
            raise TypeUnificationError(self.lhs.location, self.rhs.location, lhs_type, rhs_type)
        return typedef

class UnaryExpr(Expr):
    op:   TOK
    expr: Expr

    def __init__(self, op: TOK, expr: Expr, location: Atom):
        super().__init__(location)
        self.op   = op
        self.expr = expr

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return self.expr.get_type_def(parser, module, context)

class PostfixExpr(Expr):
    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        raise NotImplementedError

class Identifier(Expr):
    name: str

    def __init__(self, name: str, location: Atom):
        super().__init__(location)
        self.name = name

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        if context is None or self.name not in context.fields:
            raise IllegalReferenceError(self.name, self.location)

        field = context.fields[self.name]
        typedef = field.type_ref.resolve(parser, module)
        if field.optional:
            typedef = NullableType(typedef, field.type_ref.location)
        return typedef

class FieldAccessExpr(PostfixExpr):
    expr:  Expr
    field: Identifier

    def __init__(self, expr: Expr, field: Identifier, location: Atom):
        super().__init__(location)
        self.expr  = expr
        self.field = field

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        struct_def = self.expr.get_type_def(parser, module, context)

        nullable = isinstance(struct_def, NullableType)
        if nullable:
            struct_def = struct_def.typedef

        if not isinstance(struct_def, StructDef):
            raise FieldAccessError(self.field.name, self.field.location)

        if self.field.name not in struct_def.fields:
            raise IllegalReferenceError(self.field.name, self.field.location)

        typedef = struct_def.fields[self.field.name]

        if nullable and not isinstance(typedef, NullableType):
            typedef = NullableType(typedef, self.field.location)

        return typedef

class ArrayItemAccessExpr(PostfixExpr):
    array: Expr
    item:  Expr

    def __init__(self, array: Expr, item: Expr, location: Atom):
        super().__init__(location)
        self.array = array
        self.item  = item

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        array_def = self.array.get_type_def(parser, module, context)

        nullable = isinstance(array_def, NullableType)
        if nullable:
            array_def = array_def.typedef

        if not isinstance(array_def, StructDef):
            raise ItemAccessError(self.item.location)

        typedef = array_def.items

        if nullable and not isinstance(typedef, NullableType):
            typedef = NullableType(typedef, self.item.location)

        return typedef

class ArrayTypeRef(TypeRef):
    item_ref: TypeRef
    count:    Optional[Expr]

    def __init__(self, item_ref: TypeRef, count: Optional[Expr], location: Atom):
        if count is None or not isinstance(count, Integer):
            count_val = None
        else:
            count_val = count.value
        super().__init__(
            f"{item_ref.name}[]" if count_val is None else f"{item_ref.name}[{count_val}]",
            location)
        self.item_ref = item_ref
        self.count    = count

    def resolve(self, parser: "Parser", module: "Module") -> TypeDef:
        items = self.item_ref.resolve(parser, module)
        if self.count is None or not isinstance(self.count, Integer):
            count = None
        else:
            count = self.count.value
        return parser._get_array_type(items, count, self.location)

class PrimitiveType(TypeDef):
    pytype: type

    def __init__(self, pytype: type, size: int, name: str, location: Optional[Atom] = None):
        super().__init__(name, size, location or Atom(0, 0, 0, 0, 0, TOK.TYPE, name))
        self.pytype = pytype

UINT8  = PrimitiveType(int,   1, "uint8")
INT8   = PrimitiveType(int,   1, "int8")
BYTE   = PrimitiveType(int,   1, "byte")
UINT16 = PrimitiveType(int,   2, "uint16")
INT16  = PrimitiveType(int,   2, "int16")
UINT32 = PrimitiveType(int,   4, "uint32")
INT32  = PrimitiveType(int,   4, "int32")
UINT64 = PrimitiveType(int,   8, "uint64")
INT64  = PrimitiveType(int,   8, "int64")
BOOL   = PrimitiveType(bool,  1, "bool")
FLOAT  = PrimitiveType(float, 4, "float")
DOUBLE = PrimitiveType(float, 8, "double")

PRIMITIVES = dict((tp.name, tp) for tp in [
    UINT8, INT8, BYTE, UINT16, INT16, UINT32, INT32,
    UINT64, INT64, BOOL, FLOAT, DOUBLE,
])

SIGNED_INTS   = {8: INT8,  16: INT16,  32: INT32,  64: INT64}
UNSIGNED_INTS = {8: UINT8, 16: UINT16, 32: UINT32, 64: UINT64}

FLOATS = {32: FLOAT, 64: DOUBLE}

class NeverType(TypeDef):
    def __init__(self, name: str, location: Optional[Atom] = None):
        super().__init__(name, 0, location or Atom(0, 0, 0, 0, 0, TOK.TYPE, name))

NEVER = NeverType('never')

class NullableType(TypeDef):
    typedef: TypeDef

    def __init__(self, typedef: TypeDef, location: Optional[Atom] = None):
        name = typedef.name + '?'
        super().__init__(name, typedef.size, location or Atom(0, 0, 0, 0, 0, TOK.TYPE, name))
        self.typedef = typedef

NULL = NullableType(NEVER)

class StringType(TypeDef):
    pass

STRING = StringType("string", None, Atom(0, 0, 0, 0, 0, TOK.TYPE, "string"))

class Value(Expr):
    value: Any

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        raise NotImplementedError

class AtomicValue(Value):
    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        raise NotImplementedError

class PrimitiveValue(AtomicValue):
    value: Union[int, float, bool]
    typedef: Optional[TypeDef]

    def __init__(self, typedef: Optional[TypeDef], location: Atom):
        super().__init__(location)
        self.typedef = typedef

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return self.typedef

class Integer(PrimitiveValue):
    value: int

    def __init__(self, value: int, typedef: Optional[TypeDef], location: Atom):
        super().__init__(typedef, location)
        self.value = value

class Float(PrimitiveValue):
    value: float

    def __init__(self, value: float, typedef: Optional[TypeDef], location: Atom):
        super().__init__(typedef, location)
        self.value = value

class Bool(PrimitiveValue):
    value: bool

    def __init__(self, value: bool, location: Atom):
        super().__init__(BOOL, location)
        self.value = value

class String(AtomicValue):
    value: str

    def __init__(self, value: str, location: Atom):
        super().__init__(location)
        self.value = value

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return STRING

class Null(AtomicValue):
    value: None

    def __init__(self, location: Atom):
        super().__init__(location)
        self.value = None

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return NULL

class ArrayLiteral(Value):
    value:    Union[List[Any], bytes]
    type_ref: ArrayTypeRef

    def __init__(self, value: Union[List[Any], bytes], type_ref: ArrayTypeRef, location: Atom):
        super().__init__(location)
        self.value    = value
        self.type_ref = type_ref

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return self.type_ref.resolve(parser, module)

class StructLiteral(Value):
    value: Dict[str, Tuple[Identifier, Value]]
    type_ref: TypeRef

    def __init__(self, value: Dict[str, Tuple[Identifier, Value]], type_ref: TypeRef, location: Atom):
        super().__init__(location)
        self.value    = value
        self.type_ref = type_ref

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return self.type_ref.resolve(parser, module)

class RaiseExpr(Expr):
    message: String

    def __init__(self, message: String, location: Atom):
        super().__init__(location)
        self.message = message

    def get_type_def(self, parser: "Parser", module: "Module", context: Optional[StructDef]) -> TypeDef:
        return NEVER

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
    fields:   Dict[str, FieldDef] # OrderedDict
    sections: List[Section]

    def __init__(self, name: str, location: Atom, fields: Dict[str, FieldDef]=None,
                 size: Optional[int]=None, sections: List[Section]=None):
        super().__init__(name, size, location)
        self.fields   = fields if fields is not None else OrderedDict()
        self.sections = sections if sections is not None else []

def _array_type_info(items: TypeDef, count: Optional[int]) -> Tuple[str, Optional[int]]:
    item_size = items.size
    if item_size is not None and count is not None:
        size = item_size * count
    else:
        size = None
    name = f'{items.name}[{count}]' if count is not None else f'{items.name}[]'
    return (name, size)

class ArrayTypeDef(TypeDef):
    items: TypeDef
    count: Optional[int]

    def __init__(self, items: TypeDef, count: Optional[int], location: Atom):
        name, size = _array_type_info(items, count)
        super().__init__(name, size, location)
        self.items = items
        self.count = count

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

PRELUDE_TYPES = PRIMITIVES | frozenset((STRING,))

class ModuleState(Enum):
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
        self.types           = dict(PRELUDE_TYPES)

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
    _array_types:    Dict[Tuple[int, str], ArrayTypeDef]

    def __init__(self, root_path: str = '.'):
        self._root_path      = abspath(root_path)
        self._module_map     = {0: PRELUDE}
        self._modules        = [PRELUDE]
        self._module_queue   = []
        self._tokens         = iter(())
        self._current_token  = None
        self._current_module = PRELUDE

    def _get_array_type(self, items: TypeDef, count: Optional[int], location: Atom) -> ArrayTypeDef:
        name, _ = _array_type_info(items, count)
        key = (location.fileid, name)
        if key in self._array_types:
            return self._array_types[key]

        typedef = ArrayTypeDef(items, count, location)
        self._array_types[key] = typedef
        return typedef

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
            filename = join_path(dirname(self._current_module.filename), filename)
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
        struct_def = StructDef(name, name_tok, fields, None, sections)
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
            type_ref = ArrayTypeRef(type_ref, size, name_tok)

        return type_ref

    def _parse_value(self):
        if self._has_next(TOK.INT):
            token = self._next_token()
            value, signed, bits = token.parse_value()
            if bits is None:
                typedef = None
            elif signed:
                typedef = SIGNED_INTS[bits]
            else:
                typedef = UNSIGNED_INTS[bits]

            return Integer(value, typedef, token)

        elif self._has_next(TOK.BYTE):
            token = self._next_token()
            return Integer(token.parse_value(), BYTE, token)

        elif self._has_next(TOK.FLOAT):
            token = self._next_token()
            value, bits = token.parse_value()
            if bits is None:
                typedef = None
            else:
                typedef = FLOATS[bits]

            return Float(value, typedef, token)

        elif self._has_next(TOK.BOOL):
            token = self._next_token()
            return Bool(token.parse_value(), token)

        elif self._has_next(TOK.STR):
            token = self._next_token()
            return String(token.parse_value(), token)

        elif self._has_next(TOK.BYTES):
            token = self._next_token()
            value = token.parse_value()
            return ArrayLiteral(value, ArrayTypeRef(TypeRef(BYTE.name, token), len(value), token), token)

        elif self._has_next(TOK.NULL):
            token = self._next_token()
            return Null(token)

        type_ref = self._parse_type_ref()

        with self._expect_paren(TOK.BR_OPEN):
            if isinstance(type_ref, ArrayTypeRef):
                items = []
                # parse array literal
                while not self._has_next(TOK.CUR_CLOSE):
                    value = self._parse_value()
                    items.append(value)

                    if self._has_next(TOK.SEMICOL):
                        self._next_token()
                    else:
                        break

                value = ArrayLiteral(items, type_ref, type_ref.location)
            else:
                items: Dict[str, Tuple[Identifier, Value]] = OrderedDict()
                # parse struct literal
                while not self._has_next(TOK.CUR_CLOSE):
                    ident_tok = self._expect(TOK.ID)
                    key = Identifier(ident_tok.value, ident_tok)
                    if key.name in items:
                        raise FieldRedefinedError(key, ident_tok, items[key][0])

                    self._expect(TOK.COLON)
                    value = self._parse_value()
                    items[key.name] = (key, value)

                    if self._has_next(TOK.SEMICOL):
                        self._next_token()
                    else:
                        break

                value = StructLiteral(items, type_ref, type_ref.location)

        return value

    def _parse_expr(self):
        return self._parse_cond_expr()

    def _parse_cond_expr(self):
        expr = self._parse_or_expr()

        while self._has_next(TOK.QUEST):
            self._next_token()
            true_expr = self._parse_cond_expr()
            self._expect(TOK.COLON)
            false_expr = self._parse_cond_expr()
            expr = ConditionalExpr(expr, true_expr, false_expr, expr.location)

        return expr

    def _parse_or_expr(self):
        expr = self._parse_and_expr()
        while self._has_next(TOK.OR):
            self._next_token()
            rhs = self._parse_and_expr()
            expr = BinaryExpr(expr, rhs, TOK.OR, expr.location)
        return expr

    def _parse_and_expr(self):
        expr = self._parse_bit_or_expr()
        while self._has_next(TOK.AND):
            self._next_token()
            rhs = self._parse_bit_or_expr()
            expr = BinaryExpr(expr, rhs, TOK.AND, expr.location)
        return expr

    def _parse_bit_or_expr(self):
        expr = self._parse_xor_expr()
        while self._has_next(TOK.BOR):
            self._next_token()
            rhs = self._parse_xor_expr()
            expr = BinaryExpr(expr, rhs, TOK.BOR, expr.location)
        return expr

    def _parse_xor_expr(self):
        expr = self._parse_bit_and_expr()
        while self._has_next(TOK.XOR):
            self._next_token()
            rhs = self._parse_bit_and_expr()
            expr = BinaryExpr(expr, rhs, TOK.XOR, expr.location)
        return expr

    def _parse_bit_and_expr(self):
        expr = self._parse_eq_expr()
        while self._has_next(TOK.BAND):
            self._next_token()
            rhs = self._parse_eq_expr()
            expr = BinaryExpr(expr, rhs, TOK.BAND, expr.location)
        return expr

    def _parse_eq_expr(self):
        expr = self._parse_rel_expr()
        while self._has_next(TOK.EQ) or self._has_next(TOK.NE):
            operator = self._next_token()
            rhs = self._parse_rel_expr()
            expr = BinaryExpr(expr, rhs, operator.token, expr.location)
        return expr

    def _parse_rel_expr(self):
        expr = self._parse_shift_expr()
        while self._has_next(TOK.LT) or self._has_next(TOK.GT) or self._has_next(TOK.LE) or self._has_next(TOK.GE):
            operator = self._next_token()
            rhs = self._parse_shift_expr()
            expr = BinaryExpr(expr, rhs, operator.token, expr.location)
        return expr

    def _parse_shift_expr(self):
        expr = self._parse_add_expr()
        while self._has_next(TOK.LSHIFT) or self._has_next(TOK.RSHIFT):
            operator = self._next_token()
            rhs = self._parse_add_expr()
            expr = BinaryExpr(expr, rhs, operator.token, expr.location)
        return expr

    def _parse_add_expr(self):
        expr = self._parse_mul_expr()
        while self._has_next(TOK.ADD) or self._has_next(TOK.SUB):
            operator = self._next_token()
            rhs = self._parse_mul_expr()
            expr = BinaryExpr(expr, rhs, operator.token, expr.location)
        return expr

    def _parse_mul_expr(self):
        expr = self._parse_unary_expr()
        while self._has_next(TOK.MUL) or self._has_next(TOK.DIV) or self._has_next(TOK.MOD):
            operator = self._next_token()
            rhs = self._parse_unary_expr()
            expr = BinaryExpr(expr, rhs, operator.token, expr.location)
        return expr

    def _parse_unary_expr(self):
        if self._has_next(TOK.SUB) or self._has_next(TOK.BANG) or self._has_next(TOK.BNOT):
            operator = self._next_token()
            expr = self._parse_unary_expr()
            return UnaryExpr(operator.token, expr, operator)

        if self._has_next(TOK.RAISE):
            token = self._next_token()
            msg_tok = self._expect(TOK.STR)
            return RaiseExpr(String(msg_tok.value, msg_tok), token)

        return self._parse_postfix_expr()

    def _parse_postfix_expr(self):
        expr = self._parse_primary_expr()
        while self._has_next(TOK.DOT) or self._has_next(TOK.BR_OPEN):
            token = self._next_token()
            if token.token == TOK.DOT:
                ident_tok = self._expect(TOK.ID)
                expr = FieldAccessExpr(expr, Identifier(ident_tok.value, ident_tok), expr.location)
            else:
                item = self._parse_expr()
                self._expect_close(token)
                expr = ArrayItemAccessExpr(expr, item, expr.location)
        return expr

    def _parse_primary_expr(self):
        if self._has_next(TOK.ID):
            token = self._next_token()
            return Identifier(token.value, token)

        if self._has_next(TOK.PAR_OPEN):
            token = self._next_token()
            expr = self._parse_expr()
            self._expect_close(token)
            return expr

        return self._parse_value()

def parse_file(filename: str, root_path:str='.'):
    return Parser(root_path).parse_file(abspath(filename))

def parse_string(source: str, filename: str='-', root_path:str='.'):
    return Parser(root_path).parse_string(source, filename)
