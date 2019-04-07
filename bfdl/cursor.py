#!/usr/bin/env python3

class Span:
    fielid:      int
    start_index: int
    end_index:   int

    def __init__(self, fileid: int, start_index: int, end_index: int):
        self.fileid      = fileid
        self.start_index = start_index
        self.end_index   = end_index

    def clone(self) -> "Span":
        return Span(self.fileid, self.start_index, self.end_index)

class Cursor:
    fileid: int
    index:  int

    def __init__(self, fileid: int, index: int):
        self.fileid = fileid
        self.index  = index

    def clone(self) -> "Cursor":
        return Cursor(self.fileid, self.index)

    def to_span(self) -> Span:
        return Span(self.fileid, self.index, self.index + 1)

def make_span(cursor1: Cursor, cursor2: Cursor) -> Span:
    if cursor1.fileid != cursor2.fileid:
        raise ValueError(f"cursor1.fileid ({cursor1.fileid}) != cursor2.fileid ({cursor2.fileid})")

    return Span(cursor1.fileid, cursor1.index, cursor2.index)

def join_spans(first: Span, *spans: Span) -> Span:
    span = first.clone()
    for other in spans:
        if other.fileid != span.fileid:
            raise ValueError(f"first.fileid ({first.fileid}) != other.fileid ({other.fileid})")

        if other.start_index < span.start_index:
            span.start_index = span.start_index

        if other.end_index > span.end_index:
            span.end_index = span.end_index

    return span
