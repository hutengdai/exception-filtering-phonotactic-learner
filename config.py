# -*- coding: utf-8 -*-

from pynini import SymbolTable

epsilon = 'ϵ'  # <eps>
bos = '<s>'  # '>' | <s>
eos = '<e>'  # '<' | </s>
λ = ''  # Empty string (de la Higuera, p. 48)
unk = '⊥'  # Unknown / empty set (de la Higuera, p. 376)
sigma = ['a', 'b']  # Ordinary symbols
special_syms = []  # Special symbols
syms = []  # All symbols in symtable
symtable = None  # SymbolTable

verbosity = 0


def init(config):
    """ Set globals with dictionary or module """
    global epsilon, bos, eos
    global sigma, special_syms
    global syms, symtable
    #if not isinstance(config, dict):
    #    print(config)
    #    config = vars(config)
    if 'epsilon' in config:
        epsilon = config['epsilon']
    if 'bos' in config:
        bos = config['bos']
    if 'eos' in config:
        eos = config['eos']
    if 'sigma' in config:
        sigma = config['sigma']
    if 'special_syms' in config:
        special_syms = config['special_syms']
    symtable = SymbolTable()
    symtable.add_symbol(epsilon)
    symtable.add_symbol(bos)
    symtable.add_symbol(eos)
    for sym in special_syms:
        symtable.add_symbol(sym)
    for sym in sigma:
        symtable.add_symbol(sym)
    syms = [sym for (sym_id, sym) in symtable]
    #print(syms)