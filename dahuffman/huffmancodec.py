import collections
import itertools
import sys
import json
from collections import defaultdict
from heapq import heappush, heappop, heapify

from .compat import to_byte, from_byte, concat_bytes


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

    def __init__(self, code_2_symbol, probs_dict):
        self.code_2_symbol = code_2_symbol

        normalization = sum(v for k, v in probs_dict.items())
        self.probs_dict = {k: v / normalization for k, v in probs_dict.items()}

        # also compute the actual probability of each symbol being chosen (assumes that the bits {0,1}* are equally likely and i.i.d.)
        self.symbol_probs = defaultdict(float)
        for code, symbol in self.code_2_symbol.items():
            self.symbol_probs[symbol] += 2**(-len(code))

    def expected_code_length(self):
        """
        Computes the expected code-length of the encoding.
        :return: L = E_X[l(X)]  where P(x_i) = probs[i]
        """
        assert set(self.probs_dict.keys()) == set(self.code_2_symbol.values())  # assert that the vocabulary is the same

        expected_length = sum(2**(-len(code)) * len(code) for code, _ in self.code_2_symbol.items())
        return expected_length

    def discrepancy(self):
        """
        Computes the average discrepancy between the node probabilities and the probabilities. (assuming {0,1}* ~ Unif({0,1}^*))
        """
        assert set(self.probs_dict.keys()) == set(self.code_2_symbol.values())

        discrepancy = 0
        for symbol, prob in self.symbol_probs.items():
            discrepancy += abs(self.symbol_probs[symbol] - self.probs_dict[symbol])

        return discrepancy

    def expected_discrepancy(self):
        assert set(self.probs_dict.keys()) == set(self.code_2_symbol.values())

        exp_discr = 0
        for symbol, prob in self.symbol_probs.items():
            exp_discr += self.symbol_probs[symbol] * abs(self.symbol_probs[symbol] - self.probs_dict[symbol])

        return exp_discr

    def num_leaves(self):
        return len(self.code_2_symbol)

    def __repr__(self):
        return "Codes: " + repr(self.code_2_symbol)


class HuffmanCodec(PrefixCodec):
    """
    Huffman coder, with code table built from given symbol probabilities or raw data,
    providing encoding and decoding methods.
    """

    @classmethod
    def from_probabilities(cls, probs_dict, concat=None):
        """
        Build Huffman code table from given symbol probabilities
        :param probabilities: symbol to probability mapping
        :param concat: function to concatenate symbols
        """
        concat = concat or _guess_concat(next(iter(probs_dict)))

        # Heap consists of tuples: (frequency, [list of tuples: (symbol, (bitsize, value))])
        heap = [(f, [(s, (0, 0))]) for s, f in probs_dict.items()]
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

        code_2_symbol = {}
        for symbol, (bitsize, value) in table.items():
            code = bin(value)[2:].rjust(bitsize, '0')
            code_2_symbol[code] = symbol

        return PrefixCodec(code_2_symbol, probs_dict)


class BinaryApproximationTree(PrefixCodec):

    @classmethod
    def from_probabilities(cls, probs_dict, max_depth=10):
        # assert abs(1 - sum(probs_dict.values())) < 1e-6

        symbols, probs = zip(*probs_dict.items())

        free_nodes = ["0", "1"]
        symbol_2_nodes = defaultdict(list)

        bin_exps = [cls._binary_expansion(p, max_depth=max_depth+1) for p in probs]

        for i in range(max_depth):
            for j, bin_repr in enumerate(bin_exps):
                if bin_repr[i] == 1:
                    symbol_2_nodes[j].append(free_nodes.pop())

            # all remaining free nodes are expanded
            free_nodes = [node+bit for node in free_nodes for bit in ["0", "1"]]

        # assert len(free_nodes) == 0

        code_2_symbol = {}
        for symbol_ix, codes in symbol_2_nodes.items():
            for code in codes:
                code_2_symbol[code] = symbols[symbol_ix]  # we only considered symbol indices until here

        # make sure all symbols have actually been encoded, i.e. each symbol has at least one associated code
        assert set(code_2_symbol.values()) == set(probs_dict.keys()) and "not all symbols have been encoded"

        return PrefixCodec(code_2_symbol, probs_dict), free_nodes

    @classmethod
    def _binary_expansion(cls, x, max_depth):
        assert x >= 0 and x <= 1

        bits = []
        approx = 0
        for i in range(1, max_depth):
            add = 2**(-i)
            if approx + add <= x:
                bits.append(1)
                approx += add
            else:
                bits.append(0)

        return bits  # , approx, abs(approx - x)
