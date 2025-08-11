import inspect


def debug(message: str, *expressions: str):
    """
    Print a message and a list of expressions with their values.

    Example:
        debug("lower: inputs", "node.op", "len(node.inputs)", "node.inputs[0].shape")

    Output (to stderr by default):
        lower: inputs: node.op='Add', len(node.inputs)=2, node.inputs[0].shape=(2, 3)
    """
    # Evaluate in the caller's frame
    frame = inspect.currentframe()
    caller = frame.f_back
    glb, loc = caller.f_globals, caller.f_locals
    parts: list[str] = []

    for expr in expressions:
        val = eval(expr, glb, loc)  # nosec - debug-only evaluator
        parts.append(f"{expr}={val!r}")

    if parts:
        print(f"{message}: " + ", ".join(parts))
    else:
        print(message)
