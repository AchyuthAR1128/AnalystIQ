import ast
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==================================================
# ALLOWED PYTHON OPERATIONS
# ==================================================

ALLOWED_NODES = {
    # Basic structure
    ast.Module,
    ast.Assign,
    ast.Expr,

    # Variables
    ast.Name,
    ast.Load,
    ast.Store,

    # Values
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,

    # DataFrame operations
    ast.Subscript,
    # Enables safe column selection such as df.loc[:, ["product", "price"]].
    ast.Slice,
    ast.Call,
    ast.Attribute,

    # IMPORTANT:
    # Allows keyword arguments such as:
    # ascending=False
    # by="total_revenue"
    # total_revenue="sum"
    ast.keyword,

    # Operators
    ast.BinOp,
    ast.UnaryOp,

    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,

    # Comparisons
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,

    # Boolean operations
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
}


# ==================================================
# BLOCKED NAMES
# ==================================================

BLOCKED_NAMES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "requests",
    "socket",

    "open",
    "eval",
    "exec",
    "__import__",
    "compile",

    "globals",
    "locals",

    "input",

    "help",
    "dir",
}

BLOCKED_ATTRIBUTES = {
    "read_csv", "read_excel", "read_json", "read_parquet", "read_pickle",
    "to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "to_sql",
    "to_html", "to_markdown", "write_html", "write_image", "write_json",
    "save", "show",
}


# ==================================================
# VALIDATE GENERATED CODE
# ==================================================

def validate_code(code):

    try:
        tree = ast.parse(code)

    except SyntaxError as error:

        return (
            False,
            f"Invalid Python syntax: {error}"
        )

    for node in ast.walk(tree):

        # ------------------------------------------
        # Check allowed AST operations
        # ------------------------------------------

        if type(node) not in ALLOWED_NODES:

            return (
                False,
                f"Blocked Python operation: "
                f"{type(node).__name__}"
            )

        # ------------------------------------------
        # Block dangerous names
        # ------------------------------------------

        if isinstance(node, ast.Name):

            if node.id in BLOCKED_NAMES:

                return (
                    False,
                    f"Blocked name: {node.id}"
                )

        # ------------------------------------------
        # Block dunder attributes
        # ------------------------------------------

        if isinstance(node, ast.Attribute):

            if node.attr.startswith("__"):

                return (
                    False,
                    "Dunder attributes are not allowed."
                )

            if node.attr in BLOCKED_ATTRIBUTES:
                return False, f"Blocked attribute: {node.attr}"

    return True, None


# ==================================================
# EXECUTE GENERATED CODE
# ==================================================

def execute_code(code, df, return_figure=False):

    # ------------------------------------------
    # Validate first
    # ------------------------------------------

    is_valid, error = validate_code(code)

    if not is_valid:

        return (None, error, None) if return_figure else (None, error)

    # ------------------------------------------
    # Restricted execution environment
    # ------------------------------------------

    safe_globals = {
        "__builtins__": {},
        "pd": pd,
        "px": px,
        "go": go,
    }

    safe_locals = {
        "df": df.copy()
    }

    # ------------------------------------------
    # Execute
    # ------------------------------------------

    try:

        exec(
            compile(
                code,
                "<ai_analysis>",
                "exec"
            ),
            safe_globals,
            safe_locals
        )

    except Exception as error:

        error_message = str(error)
        return (None, error_message, None) if return_figure else (None, error_message)

    # ------------------------------------------
    # Require result variable
    # ------------------------------------------

    if "result" not in safe_locals:

        error_message = "AI code did not create a variable named 'result'."
        return (None, error_message, None) if return_figure else (None, error_message)

    fig = safe_locals.get("fig")
    if fig is not None and not isinstance(fig, go.Figure):
        error_message = "The AI chart must be a Plotly figure stored in 'fig'."
        return (None, error_message, None) if return_figure else (None, error_message)

    if return_figure:
        return safe_locals["result"], None, fig
    return safe_locals["result"], None
