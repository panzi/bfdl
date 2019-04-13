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

BDFLFile ::= FileAttribute*
             Import*
             StructDef*

Import ::= "import" String ";"
         | "import" "{" ImportRef ("," ImportRef)* [","] "}" "from" String ";"

ImportRef ::= Identifier ["as" Identifier]

StructDef ::= Attribute*
           "struct" Identifier "{"
               ( FieldDef | ConditionalSection ) *
           "}"

FieldDef ::= Attribute* Type Identifier [ "=" Value ] ";"

ConditionalSection ::= "if" "(" Expression ")" "{"
                           ( FieldDef | ConditionalSection ) *
                       "}"

Type ::= PrimitiveType | ArrayType | PointerType | TypeName

TypeName ::= Identifier

ArrayType ::= Type "[" "]"
            | Type "[" IntegerType "]"
            | Type "[" Integer "]"
            | Type "[" Expression "]"

PointerType ::= Type "*" [ "<" IntegerType ">" ]

IntegerType ::= "byte" | "uint8" | "int8" | "uint16" | "int16"
              | "uint32" | "int32" | "uint64" | "int64"
              | "uint128" | "int128"

FloatType ::= "float" | "double"

PrimitiveType ::= IntegerType | FloatType | "bool"

# Not sure if I want to keep the "#" for attributes or
# if the brackets are enough.
FileAttribute ::= "!" "#" AttributeBody

Attribute ::= "#" AttributeBody

AttributeBody ::= "[" Identifier ["=" (Value | Identifier | Type)] "]"

Value ::= AtomicValue | ArrayLiteral | StructLiteral

AtomicValue ::= PrimitiveValue | String | ByteArray | Null

PrimitiveValue ::= Integer | Boolean | Float | Byte

ArrayLiteral ::= "[" Value ("," Value)* [","] "]"

StructLiteral ::= "{" FieldAssignment ("," FieldAssignment) [","] "}"

FieldAssignment ::= Identifier "=" Value

Expression ::= ConditionalExpr

ConditionalExpr ::= OrExpr | OrExpr "?" ConditionalExpr ":" ConditionalExpr

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

UnaryExpr ::= UnaryOp UnaryExpr | "raise" String | PostfixExpr

PostfixExpr ::= PrimaryExpr
              | PostfixExpr "." Identifier
              | PostfixExpr "[" Expression "]"
              | PostfixExpr "as" Type # TODO

PrimaryExpr ::= Identifier | Value | "(" Expression ")"

UnaryOp ::= "-" | "!" | "~"

Identifier ::= r'\b[_a-zA-Z][_a-zA-Z0-9]*\b'

Integer ::= r'[-+]?(?:[0-9]+|0x[0-9a-fA-F]+|0o[0-7]+|0b[0-1]+)'

Float ::= r'[-+]?[0-9]+(?:\.[0-9]+|[eE][-+]?[0-9]+)'

String ::= r'"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{6}))*"'

Byte ::= r"'(?:[^'\n\\]|\\(?:x[0-9a-fA-F]{2}|['ntrvfb\\]))'"

ByteArray ::= r'\bb"(?:[^"\n\\]|\\(?:x[0-9a-fA-F]{2}|["ntrvfb\\]))*"'

Null ::= "null"

Ignorable ::= Comment | Space

Comment ::= r'//[^\n]*\n|/\*(?:[^\*]|\n|\*(?!\/))*\*/'

Space ::= r'\s+'
```

Attributes
----------

| Target              | Field Type            | Name              | Type/Values                | Default     |
| ------------------- | --------------------- | ----------------- | -------------------------- | ----------- |
| field, struct, file | (any)                 | endian            | `little` or `big`          | `little`    |
| field, struct, file | (any)                 | alignment         | integer                    | `4`         |
| field, struct, file | array                 | pack_array        | boolean                    | `true`      |
| field, struct, file | dynamic array, string | size_type         | integer type               | `uint32`    |
| field, struct, file | string                | encoding          | string                     | `"UTF-8"`   |
| field, struct, file | bool                  | bool_size         | integer                    | `1`         |
| field, struct, file | bool                  | true_value        | integer                    | `1`         |
| field, struct, file | bool                  | false_value       | integer                    | `0`         |
| struct, file        | N/A                   | dynamically_sized | `inclusive` or `exclusive` | `inclusive` |
| field               | (any)                 | fixed             | N/A                        | N/A         |