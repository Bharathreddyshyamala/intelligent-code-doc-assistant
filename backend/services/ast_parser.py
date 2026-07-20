import ast
import hashlib
import io
import json
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from services.file_scanner import (
    get_ast_path,
    get_source_directory,
)


IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".next",
    "dist",
    "build",
}


class DirectCallableVisitor(ast.NodeVisitor):
    """
    Collect details from a function or method body.

    Nested functions, classes, and lambdas are skipped so their calls and
    control flow are not incorrectly assigned to the parent callable.
    """

    def __init__(self) -> None:
        self.calls: List[ast.Call] = []
        self.returns: List[ast.Return] = []
        self.raises: List[ast.Raise] = []
        self.conditions: List[ast.If] = []
        self.loops: List[Union[ast.For, ast.AsyncFor, ast.While]] = []
        self.try_blocks: List[ast.Try] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raises.append(node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.conditions.append(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loops.append(node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops.append(node)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loops.append(node)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.try_blocks.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class ASTParser:
    """
    Parse Python files inside a project's source directory and save a
    chunking-ready representation to ast.json.
    """

    SCHEMA_VERSION = "2.0"

    def parse_project(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Parse a complete Python project.

        This method intentionally keeps the same interface expected by the
        existing FastAPI route:

            ast_parser.parse_project(project_id)
        """

        source_directory = get_source_directory(project_id)

        if not source_directory.exists():
            raise FileNotFoundError(
                f"Source directory not found: {source_directory}"
            )

        if not source_directory.is_dir():
            raise ValueError(
                f"Source path is not a directory: {source_directory}"
            )

        python_files = sorted(
            file_path
            for file_path in source_directory.rglob("*.py")
            if not self._is_ignored_file(
                file_path=file_path,
                source_directory=source_directory,
            )
        )

        project_result: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "project_id": project_id,
            "language": "python",
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "total_python_file_count": len(python_files),
            "file_count": 0,
            "error_count": 0,
            "files": [],
            "dependencies": [],
        }

        parsed_file_count = 0
        parsing_error_count = 0

        for file_path in python_files:
            relative_path = file_path.relative_to(
                source_directory
            )

            try:
                parsed_file = self.parse_file(
                    project_id=project_id,
                    file_path=file_path,
                    root=source_directory,
                )

                project_result["files"].append(
                    parsed_file
                )
                parsed_file_count += 1

            except SyntaxError as error:
                parsing_error_count += 1

                project_result["files"].append(
                    {
                        "id": self._generate_hash(
                            f"{project_id}:{relative_path}"
                        ),
                        "path": str(relative_path),
                        "module_name": self._get_module_name(
                            relative_path
                        ),
                        "status": "error",
                        "error": {
                            "type": "SyntaxError",
                            "message": error.msg,
                            "line": error.lineno,
                            "column": error.offset,
                            "source_line": (
                                error.text.strip()
                                if error.text
                                else None
                            ),
                        },
                    }
                )

            except Exception as error:
                parsing_error_count += 1

                project_result["files"].append(
                    {
                        "id": self._generate_hash(
                            f"{project_id}:{relative_path}"
                        ),
                        "path": str(relative_path),
                        "module_name": self._get_module_name(
                            relative_path
                        ),
                        "status": "error",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )

        project_result["file_count"] = parsed_file_count
        project_result["error_count"] = parsing_error_count
        project_result["dependencies"] = (
            self._build_dependencies(
                project_result["files"]
            )
        )

        self.save_ast(
            project_id=project_id,
            data=project_result,
        )

        return project_result

    def parse_file(
        self,
        project_id: str,
        file_path: Path,
        root: Path,
    ) -> Dict[str, Any]:
        """
        Parse one Python source file.
        """

        relative_path = file_path.relative_to(root)
        relative_path_string = str(relative_path)
        module_name = self._get_module_name(
            relative_path
        )

        source_code = file_path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source_code,
            filename=relative_path_string,
        )

        return {
            "id": self._generate_hash(
                f"{project_id}:{relative_path_string}"
            ),
            "path": relative_path_string,
            "module_name": module_name,
            "status": "parsed",
            "content_hash": self._generate_hash(
                source_code
            ),
            "source_line_count": len(
                source_code.splitlines()
            ),
            "module_docstring": ast.get_docstring(tree),
            "imports": self.extract_imports(tree),
            "module_variables": (
                self.extract_module_variables(tree)
            ),
            "classes": self.extract_classes(
                project_id=project_id,
                tree=tree,
                module_name=module_name,
                file_path=relative_path_string,
                source_code=source_code,
            ),
            "functions": self.extract_functions(
                project_id=project_id,
                tree=tree,
                module_name=module_name,
                file_path=relative_path_string,
                source_code=source_code,
            ),
            "comments": self.extract_comments(
                source_code
            ),
        }

    def extract_imports(
        self,
        tree: ast.AST,
    ) -> List[str]:
        """
        Extract imports using the same string format as File 2 so existing
        downstream code remains compatible.
        """

        imports: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                relative_prefix = "." * node.level
                module_name = node.module or ""

                for alias in node.names:
                    import_path = (
                        f"{relative_prefix}{module_name}"
                    )

                    if import_path:
                        import_path = (
                            f"{import_path}.{alias.name}"
                        )
                    else:
                        import_path = alias.name

                    imports.append(import_path)

        return sorted(set(imports))

    def extract_module_variables(
        self,
        tree: ast.Module,
    ) -> List[Dict[str, Any]]:
        """
        Extract top-level module variables and constants.
        """

        variables: List[Dict[str, Any]] = []

        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = self.get_name(node.value)

                for target in node.targets:
                    for variable_name in (
                        self._extract_target_names(target)
                    ):
                        variables.append(
                            {
                                "name": variable_name,
                                "value": value,
                                "annotation": None,
                                "line": node.lineno,
                            }
                        )

            elif isinstance(node, ast.AnnAssign):
                for variable_name in (
                    self._extract_target_names(
                        node.target
                    )
                ):
                    variables.append(
                        {
                            "name": variable_name,
                            "value": self.get_name(
                                node.value
                            ),
                            "annotation": self.get_name(
                                node.annotation
                            ),
                            "line": node.lineno,
                        }
                    )

        return variables

    def extract_classes(
        self,
        project_id: str,
        tree: ast.Module,
        module_name: str,
        file_path: str,
        source_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract top-level classes, attributes, methods, and source code.
        """

        classes: List[Dict[str, Any]] = []

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            qualified_name = (
                f"{module_name}.{node.name}"
                if module_name
                else node.name
            )

            class_code = self._extract_source_code(
                source_code=source_code,
                node=node,
            )

            methods = [
                self.extract_callable(
                    project_id=project_id,
                    node=method,
                    qualified_name=(
                        f"{qualified_name}.{method.name}"
                    ),
                    parent_qualified_name=qualified_name,
                    file_path=file_path,
                    source_code=source_code,
                    callable_kind=self._get_method_kind(
                        method
                    ),
                )
                for method in node.body
                if isinstance(
                    method,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
            ]

            classes.append(
                {
                    "id": self._generate_hash(
                        f"{project_id}:{file_path}:"
                        f"{qualified_name}"
                    ),
                    "content_hash": self._generate_hash(
                        class_code
                    ),
                    "node_type": "ClassDef",
                    "name": node.name,
                    "qualified_name": qualified_name,
                    "docstring": ast.get_docstring(node),
                    "bases": [
                        self.get_name(base)
                        for base in node.bases
                    ],
                    "decorators": [
                        self.get_name(decorator)
                        for decorator in node.decorator_list
                    ],
                    "start_line": node.lineno,
                    "end_line": getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),
                    "start_column": node.col_offset,
                    "end_column": getattr(
                        node,
                        "end_col_offset",
                        node.col_offset,
                    ),
                    "code": class_code,
                    "class_attributes": (
                        self._extract_class_attributes(
                            node
                        )
                    ),
                    "instance_attributes": (
                        self._extract_instance_attributes(
                            node
                        )
                    ),
                    "methods": methods,
                }
            )

        return classes

    def extract_functions(
        self,
        project_id: str,
        tree: ast.Module,
        module_name: str,
        file_path: str,
        source_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract top-level functions.
        """

        functions: List[Dict[str, Any]] = []

        for node in tree.body:
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            qualified_name = (
                f"{module_name}.{node.name}"
                if module_name
                else node.name
            )

            functions.append(
                self.extract_callable(
                    project_id=project_id,
                    node=node,
                    qualified_name=qualified_name,
                    parent_qualified_name=module_name,
                    file_path=file_path,
                    source_code=source_code,
                    callable_kind="function",
                )
            )

        return functions

    def extract_callable(
        self,
        project_id: str,
        node: Union[
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ],
        qualified_name: str,
        parent_qualified_name: str,
        file_path: str,
        source_code: str,
        callable_kind: str,
    ) -> Dict[str, Any]:
        """
        Extract a function or method in a form ready for semantic chunking.
        """

        function_code = self._extract_source_code(
            source_code=source_code,
            node=node,
        )

        visitor = DirectCallableVisitor()

        for statement in node.body:
            visitor.visit(statement)

        calls = [
            {
                "name": self.get_name(call.func),
                "arguments": [
                    self.get_name(argument)
                    for argument in call.args
                ],
                "keyword_arguments": {
                    keyword.arg or "**": self.get_name(
                        keyword.value
                    )
                    for keyword in call.keywords
                },
                "line": call.lineno,
                "column": call.col_offset,
            }
            for call in visitor.calls
        ]

        returns = [
            {
                "expression": self.get_name(
                    return_node.value
                ),
                "line": return_node.lineno,
            }
            for return_node in visitor.returns
        ]

        raises = [
            {
                "exception": self._extract_raise_type(
                    raise_node
                ),
                "message": self._extract_raise_message(
                    raise_node
                ),
                "line": raise_node.lineno,
            }
            for raise_node in visitor.raises
        ]

        conditions = [
            {
                "type": "if",
                "expression": self.get_name(
                    condition.test
                ),
                "line": condition.lineno,
            }
            for condition in visitor.conditions
        ]

        loops = [
            self._extract_loop(loop)
            for loop in visitor.loops
        ]

        handled_exceptions = (
            self._extract_handled_exceptions(
                visitor.try_blocks
            )
        )

        return {
            "id": self._generate_hash(
                f"{project_id}:{file_path}:"
                f"{qualified_name}"
            ),
            "content_hash": self._generate_hash(
                function_code
            ),
            "node_type": (
                "AsyncFunctionDef"
                if isinstance(
                    node,
                    ast.AsyncFunctionDef,
                )
                else "FunctionDef"
            ),
            "callable_kind": callable_kind,
            "name": node.name,
            "qualified_name": qualified_name,
            "parent_qualified_name": (
                parent_qualified_name
            ),
            "is_async": isinstance(
                node,
                ast.AsyncFunctionDef,
            ),
            "is_generator": any(
                isinstance(
                    child,
                    (
                        ast.Yield,
                        ast.YieldFrom,
                    ),
                )
                for child in ast.walk(node)
            ),
            "parameters": self._extract_parameters(node),
            "return_annotation": self.get_name(
                node.returns
            ),
            "docstring": ast.get_docstring(node),
            "decorators": [
                self.get_name(decorator)
                for decorator in node.decorator_list
            ],
            "start_line": node.lineno,
            "end_line": getattr(
                node,
                "end_lineno",
                node.lineno,
            ),
            "start_column": node.col_offset,
            "end_column": getattr(
                node,
                "end_col_offset",
                node.col_offset,
            ),
            "code": function_code,
            "calls": calls,
            "returns": returns,
            "raises": raises,
            "handled_exceptions": handled_exceptions,
            "conditions": conditions,
            "loops": loops,
            "control_flow": {
                "if_count": len(conditions),
                "loop_count": len(loops),
                "try_count": len(
                    visitor.try_blocks
                ),
                "return_count": len(returns),
                "raise_count": len(raises),
            },
        }

    def extract_comments(
        self,
        source_code: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract standalone and inline comments with Python tokenize.
        """

        comments: List[Dict[str, Any]] = []

        try:
            tokens = tokenize.generate_tokens(
                io.StringIO(source_code).readline
            )

            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comments.append(
                        {
                            "line": token.start[0],
                            "column": token.start[1],
                            "content": (
                                token.string
                                .lstrip("#")
                                .strip()
                            ),
                        }
                    )

        except tokenize.TokenError:
            return []

        return comments

    def get_name(
        self,
        node: Optional[ast.AST],
    ) -> Optional[str]:
        """
        Convert an AST node into a readable source-like string.
        """

        if node is None:
            return None

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            left_side = self.get_name(node.value)

            if left_side:
                return f"{left_side}.{node.attr}"

            return node.attr

        if isinstance(node, ast.Constant):
            return repr(node.value)

        try:
            return ast.unparse(node).strip()
        except Exception:
            return type(node).__name__

    def save_ast(
        self,
        project_id: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Save the complete parser result to ast.json.
        """

        output_path = get_ast_path(project_id)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _extract_parameters(
        self,
        node: Union[
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ],
    ) -> Dict[str, Any]:
        """
        Extract positional-only, regular, keyword-only, *args, **kwargs,
        annotations, and default values.
        """

        arguments = node.args

        positional_arguments = (
            list(arguments.posonlyargs)
            + list(arguments.args)
        )
        defaults = list(arguments.defaults)

        default_start_index = (
            len(positional_arguments) - len(defaults)
        )

        positional_parameters = []

        for index, argument in enumerate(
            positional_arguments
        ):
            default_node = None

            if index >= default_start_index:
                default_node = defaults[
                    index - default_start_index
                ]

            positional_parameters.append(
                {
                    "name": argument.arg,
                    "annotation": self.get_name(
                        argument.annotation
                    ),
                    "default": self.get_name(
                        default_node
                    ),
                    "required": default_node is None,
                    "kind": (
                        "positional_only"
                        if index
                        < len(arguments.posonlyargs)
                        else "regular"
                    ),
                }
            )

        keyword_only_parameters = []

        for argument, default_node in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
        ):
            keyword_only_parameters.append(
                {
                    "name": argument.arg,
                    "annotation": self.get_name(
                        argument.annotation
                    ),
                    "default": self.get_name(
                        default_node
                    ),
                    "required": default_node is None,
                    "kind": "keyword_only",
                }
            )

        return {
            "positional": positional_parameters,
            "vararg": (
                {
                    "name": arguments.vararg.arg,
                    "annotation": self.get_name(
                        arguments.vararg.annotation
                    ),
                }
                if arguments.vararg
                else None
            ),
            "keyword_only": keyword_only_parameters,
            "kwarg": (
                {
                    "name": arguments.kwarg.arg,
                    "annotation": self.get_name(
                        arguments.kwarg.annotation
                    ),
                }
                if arguments.kwarg
                else None
            ),
        }

    def _extract_class_attributes(
        self,
        class_node: ast.ClassDef,
    ) -> List[Dict[str, Any]]:
        """
        Extract class-level Assign and AnnAssign values.
        """

        attributes: List[Dict[str, Any]] = []

        for node in class_node.body:
            if isinstance(node, ast.Assign):
                value = self.get_name(node.value)

                for target in node.targets:
                    for attribute_name in (
                        self._extract_target_names(target)
                    ):
                        attributes.append(
                            {
                                "name": attribute_name,
                                "value": value,
                                "annotation": None,
                                "line": node.lineno,
                            }
                        )

            elif isinstance(node, ast.AnnAssign):
                for attribute_name in (
                    self._extract_target_names(
                        node.target
                    )
                ):
                    attributes.append(
                        {
                            "name": attribute_name,
                            "value": self.get_name(
                                node.value
                            ),
                            "annotation": self.get_name(
                                node.annotation
                            ),
                            "line": node.lineno,
                        }
                    )

        return attributes

    def _extract_instance_attributes(
        self,
        class_node: ast.ClassDef,
    ) -> List[Dict[str, Any]]:
        """
        Extract assignments such as self.name = value from class methods.
        """

        attributes: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, int]] = set()

        for method in class_node.body:
            if not isinstance(
                method,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            for child in ast.walk(method):
                value = None
                annotation = None

                if isinstance(child, ast.Assign):
                    targets = child.targets
                    value = self.get_name(child.value)

                elif isinstance(
                    child,
                    ast.AnnAssign,
                ):
                    targets = [child.target]
                    value = self.get_name(child.value)
                    annotation = self.get_name(
                        child.annotation
                    )

                else:
                    continue

                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(
                            target.value,
                            ast.Name,
                        )
                        and target.value.id == "self"
                    ):
                        line_number = getattr(
                            child,
                            "lineno",
                            method.lineno,
                        )
                        key = (
                            target.attr,
                            line_number,
                        )

                        if key in seen:
                            continue

                        seen.add(key)

                        attributes.append(
                            {
                                "name": target.attr,
                                "value": value,
                                "annotation": annotation,
                                "defined_in": method.name,
                                "line": line_number,
                            }
                        )

        return attributes

    def _extract_target_names(
        self,
        target: ast.AST,
    ) -> List[str]:
        """
        Extract names from simple and destructuring assignment targets.
        """

        if isinstance(target, ast.Name):
            return [target.id]

        if isinstance(
            target,
            (
                ast.Tuple,
                ast.List,
            ),
        ):
            names: List[str] = []

            for element in target.elts:
                names.extend(
                    self._extract_target_names(element)
                )

            return names

        return []

    def _extract_source_code(
        self,
        source_code: str,
        node: ast.AST,
    ) -> str:
        """
        Extract the exact source segment, falling back to line slicing.
        """

        source_segment = ast.get_source_segment(
            source_code,
            node,
        )

        if source_segment is not None:
            return source_segment

        start_line = getattr(node, "lineno", 1)
        end_line = getattr(
            node,
            "end_lineno",
            start_line,
        )

        lines = source_code.splitlines()

        return "\n".join(
            lines[start_line - 1:end_line]
        )

    def _extract_raise_type(
        self,
        raise_node: ast.Raise,
    ) -> Optional[str]:
        if raise_node.exc is None:
            return None

        if isinstance(raise_node.exc, ast.Call):
            return self.get_name(
                raise_node.exc.func
            )

        return self.get_name(raise_node.exc)

    def _extract_raise_message(
        self,
        raise_node: ast.Raise,
    ) -> Optional[str]:
        if (
            isinstance(raise_node.exc, ast.Call)
            and raise_node.exc.args
        ):
            return self.get_name(
                raise_node.exc.args[0]
            )

        return None

    def _extract_loop(
        self,
        loop: Union[
            ast.For,
            ast.AsyncFor,
            ast.While,
        ],
    ) -> Dict[str, Any]:
        if isinstance(loop, ast.While):
            return {
                "type": "while",
                "condition": self.get_name(
                    loop.test
                ),
                "line": loop.lineno,
            }

        return {
            "type": (
                "async_for"
                if isinstance(loop, ast.AsyncFor)
                else "for"
            ),
            "target": self.get_name(loop.target),
            "iterable": self.get_name(loop.iter),
            "line": loop.lineno,
        }

    def _extract_handled_exceptions(
        self,
        try_blocks: List[ast.Try],
    ) -> List[Dict[str, Any]]:
        handled: List[Dict[str, Any]] = []

        for try_block in try_blocks:
            for handler in try_block.handlers:
                handled.append(
                    {
                        "exception": self.get_name(
                            handler.type
                        ),
                        "alias": handler.name,
                        "line": handler.lineno,
                    }
                )

        return handled

    def _get_method_kind(
        self,
        node: Union[
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ],
    ) -> str:
        decorator_names = {
            self.get_name(decorator)
            for decorator in node.decorator_list
        }

        if "staticmethod" in decorator_names:
            return "static_method"

        if "classmethod" in decorator_names:
            return "class_method"

        if "property" in decorator_names:
            return "property"

        if any(
            decorator_name
            and decorator_name.endswith(".setter")
            for decorator_name in decorator_names
        ):
            return "property_setter"

        return "method"

    def _get_module_name(
        self,
        relative_path: Path,
    ) -> str:
        """
        Convert a relative path to a dotted Python module name.
        """

        parts = list(
            relative_path.with_suffix("").parts
        )

        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts)

    def _is_ignored_file(
        self,
        file_path: Path,
        source_directory: Path,
    ) -> bool:
        relative_parts = file_path.relative_to(
            source_directory
        ).parts

        return any(
            part in IGNORED_DIRECTORIES
            for part in relative_parts
        )

    def _generate_hash(
        self,
        text: str,
    ) -> str:
        """
        Use SHA-256 for deterministic file, symbol, and content IDs.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def _build_dependencies(
        self,
        files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Resolve imports that point to modules inside the parsed project.
        """

        parsed_files = [
            file_data
            for file_data in files
            if file_data.get("status") == "parsed"
        ]

        known_modules = {
            file_data["module_name"]: file_data["path"]
            for file_data in parsed_files
            if file_data.get("module_name")
        }

        module_names = sorted(
            known_modules,
            key=len,
            reverse=True,
        )

        dependencies: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, Tuple[str, ...]]] = set()

        for file_data in parsed_files:
            for imported_path in file_data["imports"]:
                normalized_import = (
                    imported_path.lstrip(".")
                )
                matched_module = None

                for module_name in module_names:
                    if (
                        normalized_import == module_name
                        or normalized_import.startswith(
                            f"{module_name}."
                        )
                    ):
                        matched_module = module_name
                        break

                if matched_module is None:
                    continue

                target_path = known_modules[
                    matched_module
                ]

                if target_path == file_data["path"]:
                    continue

                remainder = normalized_import[
                    len(matched_module):
                ].lstrip(".")

                symbols = (
                    remainder.split(".")
                    if remainder
                    else []
                )

                dependency_key = (
                    file_data["path"],
                    target_path,
                    tuple(symbols),
                )

                if dependency_key in seen:
                    continue

                seen.add(dependency_key)

                dependencies.append(
                    {
                        "source": file_data["path"],
                        "target": target_path,
                        "symbols": symbols,
                    }
                )

        return dependencies


ast_parser = ASTParser()
