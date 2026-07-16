# backend/services/chunker.py
from services.ast_parser import ASTParser

class CodeChunker:
    def __init__(self):
        self.parser = ASTParser()

    def chunk_file(self, raw_code: str, file_name: str) -> list[dict]:
        """
        Uses the AST to extract functions and classes into searchable chunks.
        """
        root_node = self.parser.parse_code(raw_code)
        chunks = []
        
        # Traverse top-level nodes
        for node in root_node.children:
            if node.type in ['function_definition', 'class_definition']:
                # Extract the exact code block
                chunk_text = raw_code[node.start_byte:node.end_byte]
                
                # Try to get the name of the function/class
                name_node = node.child_by_field_name('name')
                entity_name = raw_code[name_node.start_byte:name_node.end_byte] if name_node else "unnamed"
                
                chunks.append({
                    "file_name": file_name,
                    "entity_type": node.type,
                    "entity_name": entity_name,
                    "code_chunk": chunk_text,
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0]
                })
                
        return chunks