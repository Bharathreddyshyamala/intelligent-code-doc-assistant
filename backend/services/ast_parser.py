# backend/services/ast_parser.py
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

class ASTParser:
    def __init__(self):
        # Initialize the Python language grammar
        self.py_language = Language(tspython.language(), "python")
        self.parser = Parser()
        self.parser.set_language(self.py_language)

    def parse_code(self, raw_code: str):
        """
        Takes a raw string of Python code and returns the root node of the AST.
        """
        tree = self.parser.parse(bytes(raw_code, "utf8"))
        return tree.root_node