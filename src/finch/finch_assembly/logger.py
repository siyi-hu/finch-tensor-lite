from typing import Any


def debug(message: str, *expressions: Any):
    """
    Print a message and a list of expressions with their values.

    Example:
        TODO

    Output:
        TODO

    """
    # Evaluate in the caller's frame
    parts = [f"{expr}\n\n" for expr in expressions]
    if parts:
        print(f"{message}: " + ", ".join(parts))
    else:
        print(message)
