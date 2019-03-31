#!/usr/bin/env python3

from .tokenizer import tokenize

class TypeMeta(type):
    size: int

class Primitive(metaclass=TypeMeta):
    pass

class Bool(Primitive):
    size = 1

class Integer(Primitive):
    pass

class FloatingPoint(Primitive):
    pass

class UInt8(Integer):
    size = 1

class Byte(Integer):
    size = 1

class Int8(Integer):
    size = 1

class UInt16(Integer):
    size = 2

class Int16(Integer):
    size = 2

class UInt32(Integer):
    size = 4

class Int32(Integer):
    size = 4

class UInt64(Integer):
    size = 8

class Int64(Integer):
    size = 8

class Float(FloatingPoint):
    size = 4

class Double(FloatingPoint):
    size = 8

class ASTNode:
    pass

class Annotation(ASTNode):
    pass

class Struct(ASTNode):
    pass

def parse(source):
    tokens = tokenize(source)
    # TODO
    raise NotImplementedError
