#!/usr/bin/env python3
import __future__
import math
import operator as op
import re
import sys
from functools import reduce
import logging
import readline # lets input() use history and shortcuts like C-a, C-e, C-u

# General notes about the code:
# Functions executed by interpreters usually take `interpreter` as argument.
#   Then instead of running a function like tokenize() directly, we run interpreter.get_my("tokenize").
#   This allows us to replace those functions for a given interpreter without having to rewrite the entire
#   evaluation function to reference the new definitions of the sub-functions.
#   For the same reason functions that do not need `interpreter` as argument take it as argument anyway.
# Also, functions are generally decomposed more than in normal coding style. This allows client code to 
#   change behavior more easily by redefining those subfunctions.
# See usage by running ./demiurge.py -h
# This file uses doctests. To run them: ./demiurge.py -t


# Some functions used to tokenize and parse Scheme.
########################################################################

# Compile a regex representing tokens.
token_regex = re.compile(r"""\(|\)          # Parentheses are tokens

                             |[^"()\s]+     # Ordinary tokens like "if" or "3.5" are any number >1
                                            # of non-space non-paren non-quote chars.

                             |"             # String tokens are one quote,
                                 [^"]*      # then any number of non-quote characters,
                                 "          # then a closing quote. (Double-quotes can't appear inside strings.)
                                 """, re.VERBOSE)

def tokenize(s, interpreter):
    """Given string of Scheme code, split into tokens, up to the first occurrence of ";FINISH-BEFORE" (if any).
    Returns list of tokens as strings, and text after retokenize.
    Tokens are (, ), literals, symbols, and strings (each string is one token).
    Tests:
    >>> T = Tower()
    >>> tokenize('(* x 2.5)', T.top_interpreter)
    (['(', '*', 'x', '2.5', ')'], False)
    >>> tokenize('(concat "this is a string" "another string")', T.top_interpreter)
    (['(', 'concat', '"this is a string"', '"another string"', ')'], False)
    >>> # chr(10) is newline, needed so doctest doesn't panic
    >>> tokenize(chr(10).join(("(+", "; comment", "1", "; comment", "2)")), T.top_interpreter)
    (['(', '+', '1', '2', ')'], False)
    >>> tokenize('alpha' + chr(10) + ';FINISH-BEFORE' + chr(10) + 'beta', T.top_interpreter)
    (['alpha'], 'beta')
    """
    # Split at the first ;FINISH-BEFORE directive, if any
    if '\n;FINISH-BEFORE\n' in s:
        s, remainder = s.split('\n;FINISH-BEFORE\n', 1)
    else:
        remainder = False
    # Remove comments
    s = interpreter.get_my("remove_comments")(s, interpreter)
    # Then apply the regex above to tokenize
    return re.findall(token_regex, s), remainder

def remove_comments(s, interpreter):
    """Removes comments (anything with a ';', until newline) from a string.
    Assumes no ; inside strings."""
    retval = re.sub(";[^\n]*\n", '\n', s)
    return retval

def parse_tokens(tokens, interpreter):
    """Given a list of tokens, return a nested list with separations as the parentheses indicate.
    Note that there can be more than one expression, so eg "(a) (b)" => [['a'], ['b']], and "(a b)" => [['a', 'b']].
    Tests:
    >>> parse_tokens(['(', '+', 'x', '3', ')'], None)
    [['+', 'x', '3']]
    >>> parse_tokens("( + 3 ( - 5 3 ) )".split(), None)
    [['+', '3', ['-', '5', '3']]]
    >>> parse_tokens("()", None)
    [[]]
    >>> parse_tokens("((())", None)
    Traceback (most recent call last):
        ...
    SyntaxError: Unmatched '('
    >>> parse_tokens("(()))", None)
    Traceback (most recent call last):
        ...
    SyntaxError: Unmatched ')'
    >>> parse_tokens("3", None)
    ['3']
    >>> parse_tokens("(ab)(cd)", None)
    [['a', 'b'], ['c', 'd']]
    """
    result = []
    stack = [result] # stack of lists to append to
    for token in tokens:
        if token == '(':
            # Add a new list to the last list in the stack, and add that new list to the stack.
            new_list = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif token == ')':
            # Go back one layer in the stack.
            stack = stack[:-1]
            if not stack: raise SyntaxError("Unmatched ')'")
        else:
            # Add token to the last list in the stack.
            stack[-1].append(token)
    # Stack should be empty now
    if stack != [result]:
        raise SyntaxError("Unmatched '('")
    # Result has one extra layer of listing, to make code simpler.
    #if len(result) != 1: raise SyntaxError("All code must be in one wrapping paren-pair.")
    #return result[0]
    return result

def indent(s):
    """Indents a multi-line string, by adding a tab at the start of every line."""
    return '\t' + s.replace('\n', '\n\t')


# Define the Context class, used to store variable bindings in a cactus stack.
##############################################################################

class Context(dict):
    """Stores a context, which is a mapping from symbols (strings) to Scheme expressions.
    Contexts can have 'parent' contexts, so they can be nested."""
    def __init__(self, base_dict=None, parent=None, name=None):
        self.parent = parent
        if name is None and self.parent is not None:
            name = self.parent.name + "'"
        self.name = name
        if base_dict is not None:
            self.update(base_dict)

    def get(self, symbol):
        """Try to get a symbol from a Context or its parents etc.
        >>> Context({'x': 3}).get('x')
        3
        >>> Context({'x': 3}).get('y')
        Traceback (most recent call last):
            ...
        SyntaxError: Unknown symbol 'y'
        """
        logging.debug("trying to get %s from context named %s" % (symbol, self.name))
        if not isinstance(symbol, str):
            raise Exception("Tried to look up a non-string symbol: %r" % symbol)
        if symbol in self:
            return self[symbol]
        elif self.parent is not None:
            return self.parent.get(symbol)
        else:
            raise SyntaxError("Unknown symbol %r" % symbol)

    def __str__(self):
        result = "Context named %s with mappings:" % self.name
        for key, val in self.items():
            result += "\n    %r : %r" % (key, val)
        if self.parent is not None:
            result += "\nand with parent context as follows:\n"
            result += str(self.parent)
        return result

    def copy(self):
        """Returns a copy of this context."""
        return Context(base_dict = self.base_dict, parent = self.parent, name = self.name + "-copy")


class CalledWithContext(object):
    """Wrapper class for special functions which need to be given the context object to run.
    For instance, a get-current-context function can be written as:
        CalledWithContext(lambda args, context: context)
    """
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, args, context):
        return self.fn(args, context)


# Code for evaluating expressions, from most to least general
########################################################################

def base_eval(code, interpreter):
    """Used by interpreter `interpreter` to evaluate client code, unless redefined."""
    assert type(code) == str
    val = None
    while code:
        logging.debug("running base_eval on code:\n%s" % indent(code))
        tokens, code = interpreter.get_my("tokenize")(code, interpreter)
        logging.debug("I%d: Tokens: %r" % (interpreter.depth, tokens))
        asts = interpreter.get_my("parse_tokens")(tokens, interpreter)
        logging.debug("I%d: Parsed: %r" % (interpreter.depth, asts))
        for ast in asts:
            val = interpreter.get_my("ast_eval")(ast, interpreter.globals, interpreter)
            logging.debug("I%d: Value: %s" % (interpreter.depth, val))
    return val

def ast_eval(exp, context, interpreter):
    """Evaluate an expression, either a literal/symbol (string) or an s-expression of the form outputted by parse_tokens.
    `exp` is the expression, `context` is an object of type Context to use when evaluating the expression.
    Examples/tests:
    >>> T = Tower()
    >>> c = Context({'+': op.add})
    >>> ast_eval('2.', c, T.top_interpreter)
    2.0
    >>> ast_eval(['+', '2', '3'], c, T.top_interpreter)
    5
    >>> ast_eval(['+', ['+', '1', '2'], '3'], c, T.top_interpreter)
    6
    >>> ast_eval(['set', 'x', '2'], c, T.top_interpreter)
    2
    >>> ast_eval('x', c, T.top_interpreter)
    2
    >>> ast_eval(['+', 'y', '1'], c, T.top_interpreter)
    Traceback (most recent call last):
        ...
    ValueError: not a literal: y
    """
    if isinstance(exp, str):
        return interpreter.get_my("eval_atomic")(exp, context, interpreter)
    elif isinstance(exp, list):
        return interpreter.get_my("eval_s_expression")(exp, context, interpreter)
    else:
        raise SyntaxError("Trying to ast_eval() an expression of type %r" % type(exp))

def eval_s_expression(exp, context, interpreter):
    """Evaluate an expression which is a list.
    To evaluate an s-expression, we evaluate all terms, which should be a function and its args,
    then apply the function.
    There are exceptions to this:
    - Empty lists evaluate to themselves
    - Lambda-expressions have a list of arg names, which can't be evaluated normally.
    - Joots statements are more simply written here than in the context.
    - If-statements could have side effects in their if-then and else terms, so we don't evaluate them.
    - Set / set-global statements have an unevaluatable symbol as first argument.
    """
    if not exp:
        return []
    elif exp[0] == "if":
        return interpreter.get_my("eval_if")(exp, context, interpreter)
    elif exp[0] == "lambda":
        return SchemeFunction(exp, context, interpreter)
    elif exp[0] == "floatinglambda":
        return SchemeFunction(exp, None, None)
    elif exp[0] == "joots":
        return interpreter.get_my("eval_joots")(exp, context, interpreter)
    elif exp[0] in ("set", "set_global", "set_universal"):
        return interpreter.get_my("eval_set_variant")(exp, context, interpreter)
    else:
        return interpreter.get_my("eval_application")(exp, context, interpreter)

def eval_application(exp, context, interpreter):
    """Evaluates an application of a non-keyword function to some arguments."""
    terms_evalled = [interpreter.get_my("ast_eval")(term, context, interpreter) for term in exp]
    func = terms_evalled[0]
    args = terms_evalled[1:]
    if isinstance(func, CalledWithContext):
        return func(args, context)
    elif isinstance(func, SchemeFunction) and func.floating:
        return func(context, interpreter, *args)
    else:
        return func(*args)

def eval_joots(exp, context, interpreter):
    """Evaluates a joots ("jump out of the system") expression, which is run by the interpreter below this one."""
    if len(exp) != 2:
        raise SyntaxError("Joots terms have only  1 argument. Consider using (joots (begin ...)).")
    below = interpreter.get_below()
    return below.get_my("ast_eval")(exp[1], below.globals, below)

def eval_set(exp, new_val, context, interpreter):
    """Set a local variable to the new value."""
    context[exp[1]] = new_val
    return new_val

def eval_set_global(exp, new_val, context, interpreter):
    """Set a global (to the interpreter) variable to the new value."""
    interpreter.globals[exp[1]] = new_val
    return new_val

def eval_set_universal(exp, new_val, context, interpreter):
    """Set a universal (to the tower) variable to the new value."""
    interpreter.tower.universals[exp[1]] = new_val
    return new_val

def eval_set_variant(exp, context, interpreter):
    """Evaluate a set/set_global/set_universal expression."""
    if len(exp) != 3:
        raise SyntaxError("Keyword set needs 2 args, but this doesn't: %r" % exp)
    if not isinstance(exp[1], str):
        raise SyntaxError("Trying to set a non-symbol: %r" % exp[1])
    new_val = interpreter.get_my("ast_eval")(exp[2], context, interpreter)
    if exp[0] == "set":
        return interpreter.get_my("eval_set")(exp, new_val, context, interpreter)
    elif exp[0] == "set_global":
        return interpreter.get_my("eval_set_global")(exp, new_val, context, interpreter)
    elif exp[0] == "set_universal":
        return interpreter.get_my("eval_set_universal")(exp, new_val, context, interpreter)

def eval_if(exp, context, interpreter):
    """Evaluate an if-expression."""
    if len(exp) != 4:
        raise SyntaxError("If-statements must have exactly 4 terms (if boolean then else); error at: %r" % exp)
    _, boolval, thendo, elsedo = exp
    if interpreter.get_my("ast_eval")(boolval, context, interpreter):
        val = interpreter.get_my("ast_eval")(thendo, context, interpreter)
    else:
        val = interpreter.get_my("ast_eval")(elsedo, context, interpreter)
    return val

def eval_symbol(exp, context, interpreter):
    """Evaluate a symbol by grabbing definition from the context.
    (Very short but useful for redefinitions."""
    return context.get(exp)

def eval_atomic(exp, context, interpreter):
    """Evaluate a token which is just a string, and thus must be either a symbol or a literal."""
    # Try to evaluate as a symbol by fetching from context
    try:
        return interpreter.get_my("eval_symbol")(exp, context, interpreter)
    # Else try to evaluate as a literal
    except SyntaxError:
        return interpreter.get_my("eval_literal")(exp, context, interpreter)

def eval_literal(exp, context, interpreter):
    """Evaluate a literal, which is either a string or a number or a #t/#f boolean."""
    # String literals evaluate to the string, with enclosing "" stripped, and some substitutions for string escaping
    if exp[0] == exp[-1] == '"':
        s = exp[1:-1]
        s = s.replace(r"\'", '"')
        return s
    # #t and #f are boolean literals
    if exp == "#t": return True
    if exp == "#f": return False
    # try interpreting it as a numeric literal
    as_num = interpreter.get_my("try_numeric")(exp, context, interpreter)
    if as_num is not None:
        return as_num
    # otherwise it's not a literal
    raise ValueError("not a literal: %s" % exp)

def try_numeric(s, context, interpreter):
    """Try to convert a string to an int/float; return None if unsuccessful"""
    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError:
            return None


# Tools for building lambda-functions in Scheme, and calling them
########################################################################

class SchemeFunction(object):
    """Objects are Scheme functions defined by lambda expressions.
    There are two valid forms:
    - (lambda (x y...) (body expression))
    - (lambda x (body expression)), where x gets assigned the argument list

    Additionally, you can set context and interpreter to None to create a "floating lambda"
    which does not capture its current context.

    >>> T = Tower()
    >>> T.run("(lambda (x) (+ x x))")(3)
    6
    >>> T.run("(lambda x (index 0 x))")(1,2,3)
    1
    >>> T.run("((lambda (x) (+ x x)) 99)")
    198
    """
    def __init__(self, lambda_exp, context, interpreter):
        if len(lambda_exp) != 3:
            raise SyntaxError("Lambda expression must have 3 terms (lambda args body), but this one doesn't: %r" % lambda_exp)

        # Store arguments
        self.body = lambda_exp[2]
        self.lambda_exp = lambda_exp # For reporting function when there are errors
        self.defined_in_context = context
        self.interpreter = interpreter

        # Figure out whether it's floating or not; this is relevant when it's called
        self.floating = (context is None and interpreter is None)
        assert (lambda_exp[0] == "lambda" and not self.floating) \
                or (lambda_exp[0] == "floatinglambda" and self.floating)

        # Figure out which of the forms described above it's in
        if isinstance(lambda_exp[1], list):
            self.one_list_arg = False
            self.arg_names = lambda_exp[1]
        elif isinstance(lambda_exp[1], str):
            self.one_list_arg = True
            self.arg_name = lambda_exp[1]

    def __call__(self, *args):
        # If it's floating, context and interpreter are given as arguments, else use stored values
        if self.floating: 
            assert len(args)
            context = args[0]
            interpreter = args[1]
            args = args[2:]
        else:
            context = self.defined_in_context
            interpreter = self.interpreter

        # Bind one or many arguments
        if self.one_list_arg:
            new_context = Context({self.arg_name: args}, parent = context)
        else:
            if len(args) != len(self.arg_names):
                raise SyntaxError("Expected %d args but got %d, in call to Scheme lambda-function %r" \
                        % (len(self.arg_names), len(args), self.lambda_exp))

            # Build a context with the variable names bound
            new_context = Context(parent = context)
            for arg_name, arg_val in zip(self.arg_names, args):
                new_context[arg_name] = arg_val

        # Add the function itself to the context, to allow recursion in anonymous functions
        new_context['recurse'] = self

        # Evaluate the function body in the new context
        return interpreter.get_my("ast_eval")(self.body, new_context, interpreter)

    def __str__(self):
        return "SchemeFunction generated by code: %r" % self.lambda_exp


# The reflective tower!
########################################################################

class Tower(object):
    """Represents the entire stack of interpreters.
    Can run() base code.
    Tests:
    >>> T = Tower()
    >>> T.run('(set x 123)')
    123
    >>> T.run('x')
    123
    >>> T.run('DEPTH')
    0
    >>> T.run('(joots DEPTH)')
    1
    >>> T.run('(joots (joots DEPTH))')
    2
    >>> T.run('(joots (set eval_literal (lambda _ 123)))')
    <__main__.SchemeFunction object at ...>
    >>> T.run('1')
    123
    """

    def __init__(self):
        """Creates a new Tower."""
        # Create universal context
        self.universals = self.make_universal_context()
        # Create top interpreter
        self.top_interpreter = Interpreter(None, self, 1)

    def run(self, code):
        """Run some code through the entire tower.
        Passes code to top interpreter, which passes it downward until the bottom of the tower."""
        logging.debug("Tower running code:\n%s" % indent(code))
        return self.top_interpreter.run(code)

    def make_universal_context(self):
        """Creates the Tower's universal context.
        This stores all functions that are common to all layers, including various useful functions
        >>> T = Tower()
        >>> T.run("(+ 1 2 3)")
        6
        >>> T.run("(+ (` 1 2) (` 3 4))")
        [1, 2, 3, 4]
        >>> T.run("(/ 1 (+ 1 1))")
        0.5
        >>> T.run("(begin (set x 2) x)")
        2
        >>> T.run('(tokenize "(+ 1)" INTERPRETER)')
        (['(', '+', '1', ')'], False)
        >>> T.run('(cons 1 (cons 2 ()))')
        [1, 2]
        >>> T.run('(len (` 1 2 3))')
        3
        """
        def embed_ipython(args, context): # allow embedding IPython session in the middle of Scheme code
            import IPython
            IPython.embed()

        # First make Context object, then later add items, so items can refer to the context itself.
        result = Context(name = "UNIVERSAL")
        cdict = {
                # Allow access to this tower
                "TOWER": self,

                # Basic operators; + and * can have arbitrarily many arguments
                "-": op.sub, "/": (lambda x,y: float(x) / y),
                "+": lambda *args: reduce(op.add, args), "*": lambda *args: reduce(op.mul, args),
                ">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le, "=": op.eq,
                "is": lambda x,y: (x is y),
                "and": lambda x,y: (x and y), "or": lambda x,y: (x or y),
                "pi": math.pi,
                "id": lambda x: x,

                # For operating on lists and dictionaries
                "`": lambda *args: list(args), # backtick creates a list, (` 1 2) => [1, 2]
                "index": lambda idx, l: l[idx],
                "set_index": lambda ls, idx, val: exec('ls[idx] = val'),
                "car": lambda l: l[0],
                "cdr": lambda l: l[1:],
                "cons": lambda x, l: [x] + l,

                # Convenience functions
                "infinite_loop": lambda f: exec("while True: f"),
                    # Could implement in Scheme, but causes stack to explode
                    # "(lambda (f) ((lambda () (begin (f) (recurse)))))"
                "len": len,
                "map": lambda fn, li: list(map(fn, li)),
                "dict": dict,
                "in": lambda x, l: (x in l),

                # Begin function returns its final argument; ast_eval() already evaluates them all
                "begin": lambda *args: args[-1],

                # I/O
                "print": print,
                "input": input,
                "to_num": try_numeric,
                "ipython": CalledWithContext(embed_ipython),
                    # embedded IPython session, for inspecting `context`
                    # For instance: run (begin (set x 1) (ipython)) and you'll have context.x == 1

                # Working with contexts
                "get_current_context": CalledWithContext(lambda args, context: context),
                "get_universal_context": lambda: result,
                "new_context": Context,
                "context_with_parent": lambda parent: Context(parent=parent),
                "copy_context": lambda context: context.copy(),
                "update_context": lambda c1, c2: c1.update(c2),

                # Utils used by interpreters
                "eval": base_eval,
                "tokenize": tokenize,
                "remove_comments": remove_comments,
                "parse_tokens": parse_tokens,
                "ast_eval": ast_eval,
                "eval_atomic": eval_atomic,
                "eval_literal": eval_literal,
                "eval_symbol": eval_symbol,
                "try_numeric": try_numeric,
                "eval_set": eval_set,
                "eval_set_global": eval_set_global,
                "eval_set_universal": eval_set_universal,
                "eval_set_variant": eval_set_variant,
                "eval_s_expression": eval_s_expression,
                "eval_application": eval_application,
                "eval_joots": eval_joots,
                "eval_if": eval_if,

                # Miscellaneous useful features
                "regex_findall": re.findall,
                "py_eval": exec,
                "assert_eq": lambda a, b: exec("assert a == b, 'assertion failed: %s != %s' % (a, b)"),
                "assert": lambda x: exec("assert x, 'assertion failed: %s' % x"),
                }
        result.update(cdict)
        return result

    def repl(self):
        """Runs a read-eval-print loop using this Tower."""
        while True:
            try:
                exp = input("> ")
                if exp.startswith("%load "):
                    code = open(exp[len("%load "):].strip(), 'r').read()
                    val = self.run(code)
                else:
                    val = self.run(exp)
                print(val)
            except EOFError:
                break
            except Exception as e:
                print("EXCEPTION:", e)
                input("(press enter)")

class Interpreter(object):
    """Represents one interpreter layer, with everything necessary to interpret code at the layer above.
    Note that client code is not considered an interpreter layer."""
    def __init__(self, above, tower, depth):
        """Creates interpreter layer.
        `above` is interpreter this one interprets, or None if this is the top.
        `depth` is number of layers above including end code, so Tower's top interpreter should have depth 1."""
        logging.debug("CREATING INTERPRETER AT DEPTH %d" % depth)
        self.above = above
        self.tower = tower
        self.depth = depth

        # Create global context, used for running layer/code above.
        self.globals = self.make_context()

        # Interpreter below this is currently None, can be created later.
        self.below = None

    def make_context(self):
        """Construct the context used to evaluate the layer above this interpreter.
        The universal context is a parent, so client code can use all of those functions too."""
        result = Context(parent = self.tower.universals, name = "Level%d" % self.depth)
        cdict = {
                "INTERPRETER": self,
                "DEPTH": self.depth - 1,
                "GLOBALS": result,
                }
        result.update(cdict)
        return result

    def get_my(self, symbol):
        """Fetches the definition of a symbol in this interpreter's code.
        If there's an interpreter below, that means we fetch from self.below.globals.
            (Which will often just fall back on universals anyway.)
        If this is the bottom interpreter, fetch it from universals."""
        if self.below is not None:
            return self.below.get_aboves(symbol)
        else:
            # Otherwise we haven't redefined the symbol (or we'd have a lower interpreter),
            # so we fall back on the universal definition.
            return self.tower.universals.get(symbol)

    def get_aboves(self, symbol):
        """Fetches definition of symbol in self.globals, for use in interpreting the layer above this one."""
        return self.globals.get(symbol)

    def run(self, code):
        """Given code, runs it as a client, referring to this interpreter's stored globals.
        Returns the value the code evaluates to.
        Either runs own functions directly, or (if there's another layer below),
            constructs an expression for that layer to run.
        Input `code` is a string eg "(+ 1 2)"."""
        logging.debug("I%d: calling run on code:\n%s" % (self.depth, indent(code)))
        assert type(code) == str
        evalfn = self.get_my("eval")
        return evalfn(code, self)

    def get_below(self):
        """Returns interpreter beneath this one, creating one if none exists yet."""
        if self.below is None:
            self.below = Interpreter(self, self.tower, self.depth + 1)
        return self.below

    def __repr__(self):
        return "<Interpreter at depth %d>" % self.depth


# Main functions; either run one line of code, or run a file, or REPL
########################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Interpreter for Scheme-like language, with reflective tower",
            epilog="Or run without any arguments for REPL.")
    parser.add_argument("-v", "--verbose", help="Output information about execution or testing", action="store_true")
    parser.add_argument("-t", "--testing", help="Run the program's doctests", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", help="Execute a file (can also %%load inside REPL).")
    group.add_argument("-x", "--exec", help="Execute a single string of code")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.testing:
        import doctest
        doctest.testmod(verbose=args.verbose, optionflags=doctest.ELLIPSIS)
    else:
        T = Tower()
        if args.file:
            code = open(args.file, "r").read()
            print(T.run(code))
        elif args.exec:
            print(T.run(args.exec))
        else:
            T.repl()
