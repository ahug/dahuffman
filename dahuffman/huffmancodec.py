import collections
import itertools
import sys
from heapq import heappush, heappop, heapify

from .compat import to_byte, from_byte, concat_bytes


class _EndOfFileSymbol(object):
    """
    Internal class for "end of file" symbol to be able
    to detect the end of the encoded bit stream,
    which does not necessarily align with byte boundaries.
    """

    def __repr__(self):
        return '_EOF'

    # Because _EOF will be compared with normal symbols (strings, bytes),
    # we have to provide a minimal set of comparison methods.
    # We'll make _EOF smaller than the rest (meaning lowest frequency)
    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __eq__(self, other):
        return other.__class__ == self.__class__

    def __hash__(self):
        return hash(self.__class__)


# Singleton-like "end of file" symbol
_EOF = _EndOfFileSymbol()


# TODO store/load code table from file
# TODO Directly encode to and decode from file

def _guess_concat(data):
    """
    Guess concat function from given data
    """
    return {
        type(u''): u''.join,
        type(b''): concat_bytes,
    }.get(type(data), list)


class PrefixCodec(object):
    """
    Prefix code codec, using given code table.
    """

    def __init__(self, code_table, concat=list, check=True):
        """
        Initialize codec with given code table.

        :param code_table: mapping of symbol to code tuple (bitsize, value)
        :param concat: function to concatenate symbols
        :param check: whether to check the code table
        """
        # Code table is dictionary mapping symbol to (bitsize, value)
        self._table = code_table
        self._concat = concat
        if check:
            assert isinstance(self._table, dict) and all(
                isinstance(b, int) and b >= 1 and isinstance(v, int) and v >= 0
                for (b, v) in self._table.itervalues()
            )
            # TODO check if code table is actually a prefix code

    def get_expected_code_length(self, frequencies):
        """
        Computes the expected code-length of the encoding.
        :return: L = E_X[l(X)]  where P(x_i) = probs[i]
        """
        assert set(frequencies.keys()) == set(self._table.keys())  # assert that the vocabulary is the same

        normalization = 0
        expected_length = 0
        for symbol, (length, _) in self._table.items():
            freq = frequencies[symbol]
            expected_length += freq * length
            normalization += freq

        return expected_length / normalization

    def get_average_discrepancy(self, frequencies):
        """
        Computes the average discrepancy between the node probabilities and the frequencies. (assuming {0,1}* ~ Unif({0,1}^*))
        """
        assert set(frequencies.keys()) == set(self._table.keys())

        probabilities = {}
        normalization = 0
        for symbol, (length, _) in self._table.items():
            normalization += frequencies[symbol]
            probabilities[symbol] = 2**(-length)

        discrepancy = 0
        for symbol, (length, _) in self._table.items():
            discrepancy += abs(probabilities[symbol] - frequencies[symbol]/normalization)

        return discrepancy / len(self._table)

    def get_code_table(self):
        """
        Get code table
        :return: dictionary mapping symbol to code tuple (bitsize, value)
        """
        return self._table

    def print_code_table(self, out=sys.stdout):
        """
        Print code table overview
        """
        out.write(u'bits  code       (value)  symbol\n')
        for symbol, (bitsize, value) in sorted(self._table.items()):
            out.write(u'{b:4d}  {c:10} ({v:5d})  {s!r}\n'.format(
                b=bitsize, v=value, s=symbol, c=bin(value)[2:].rjust(bitsize, '0')
            ))


class HuffmanCodec(PrefixCodec):
    """
    Huffman coder, with code table built from given symbol frequencies or raw data,
    providing encoding and decoding methods.
    """

    @classmethod
    def from_frequencies(cls, frequencies, concat=None):
        """
        Build Huffman code table from given symbol frequencies
        :param frequencies: symbol to frequency mapping
        :param concat: function to concatenate symbols
        """
        concat = concat or _guess_concat(next(iter(frequencies)))

        # Heap consists of tuples: (frequency, [list of tuples: (symbol, (bitsize, value))])
        heap = [(f, [(s, (0, 0))]) for s, f in frequencies.items()]
        # Add EOF symbol.
        # TODO: argument to set frequency of EOF?
        # heap.append((1, [(_EOF, (0, 0))]))

        # Use heapq approach to build the encodings of the huffman tree leaves.
        heapify(heap)
        while len(heap) > 1:
            # Pop the 2 smallest items from heap
            a = heappop(heap)
            b = heappop(heap)
            # Merge nodes (update codes for each symbol appropriately)
            merged = (
                a[0] + b[0],
                [(s, (n + 1, v)) for (s, (n, v)) in a[1]]
                + [(s, (n + 1, (1 << n) + v)) for (s, (n, v)) in b[1]]
            )
            heappush(heap, merged)

        # Code table is dictionary mapping symbol to (bitsize, value)
        table = dict(heappop(heap)[1])

        return cls(table, concat=concat, check=False)
