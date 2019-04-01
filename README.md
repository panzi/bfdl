Binary Format Description Language
==================================

In this project I'm coming up with a binary format description language, just
for fun. Haven't even googled if there is such a thing already before I
started this. I did now, there is a BFD and it uses XML. I'd rather stop being
a software developer than to use that.

The goal of this for fun project is to have a BFDL compiler that can generate
Python and C code for reading and writing binary files and for printing
information about files in a nice textual form.

**NOTE:** This is work in progress. I've only wrote a BFDL tokenizer so far.
And I haven't made my mind up about the license. Will probably be some sort of
open source license.

Backus-Naur Form
----------------

```BNF

BDFLFile ::= FileAnnotation*
            Struct*

Struct ::= Annotation*
           "struct" Identifier "{"
               ( Field | ConditionalSection ) *
           "}"

Field ::= Annotation* Type Identifier [ "=" Value ] ";"

ConditionalSection ::= "if" "(" Expression ")" "{"
                           ( Field | ConditionalSection ) *
                       "}"

Type ::= PrimitiveType | ArrayType | TypeName

TypeName ::= Identifier

ArrayType ::= Type "[" "]" |
              Type "[" Integer "]" |
              Type "[" Expression "]"

PrimitiveType ::= "byte" | "uint8" | "int8" | "uint16" | "int16"
                | "uint32" | "int32" | "uint64" | "int64"
                | "float" | "double" | "bool"

FileAnnotation ::= "!#" AnnotationBody

Annotation ::= "#" AnnotationBody

AnnotationBody ::= "[" Identifier ["=" (Value | Identifier | Type)] "]"

Value ::= Integer | Boolean | Float | String | Byte | ByteArray | Array

Array ::= "{" Value ("," Value)* [","] "}"

Expression ::= ConditionalExpr

ConditionalExpr ::= OrExpr | OrExpr "?" Expression ":" ConditionalExpr

OrExpr ::= AndExpr | OrExpr "||" AndExpr

AndExpr ::= BitOrExpr | AndExpr "&&" BitOrExpr

BitOrExpr ::= XOrExpr | BitOrExpr "|" XOrExpr

XOrExpr ::= BitAndExpr | XOrExpr "^" BitAndExpr

BitAndExpr ::= EqExpr | BitAndExpr "&" EqExpr

EqExpr ::= RelationalExpr
         | EqExpr "==" RelationalExpr
         | EqExpr "!=" RelationalExpr

RelationalExpr ::= ShiftExpr
                 | RelationalExpr "<" ShiftExpr
                 | RelationalExpr ">" ShiftExpr
                 | RelationalExpr "<=" ShiftExpr
                 | RelationalExpr ">=" ShiftExpr

ShiftExpr ::= AddExpr
            | ShiftExpr "<<" AddExpr
            | ShiftExpr ">>" AddExpr

AddExpr ::= MulExpr
          | AddExpr "+" MulExpr
          | AddExpr "-" MulExpr

MulExpr ::= UnaryExpr
          | MulExpr "*" UnaryExpr
          | MulExpr "/" UnaryExpr
          | MulExpr "%" UnaryExpr

UnaryExpr ::= PostfixExpr | UnaryOp UnaryExpr

PostfixExpr ::= PrimaryExpr
              | PostfixExpr "." Identifier
              | PostfixExpr "[" Expression "]"

PrimaryExpr ::= Identifier | Value | "(" Expression ")"

UnaryOp ::= "-" | "!" | "~"

Identifier ::= r'\b[_a-zA-Z][_a-zA-Z0-9]*\b'

Integer ::= r'[-+]?(?:[0-9]+|0x[0-9a-fA-F]+|0o[0-7]+|0b[0-1]+)'

Float ::= r'[-+]?[0-9]+(?:\.[0-9]+|[eE][-+]?[0-9]+)'

String ::= r'"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{6}))*"'

Byte ::= r"'(?:[^'\n\\]|\\(?:x[0-9a-fA-F]{2}|['ntrvfb\\]))'"

ByteArray ::= r'\bb"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]))*"'

Ignorable ::= Comment | Space

Comment ::= r'//[^\n]*\n|/\*(?:[^\*]|\n|\*(?!\/))*\*/'

Space ::= r'\s+'
```