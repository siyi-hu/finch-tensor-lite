import ctypes
import shutil
import subprocess
import tempfile
from functools import lru_cache
from operator import methodcaller
from pathlib import Path

from ..util import config
from ..util.cache import file_cache


@file_cache(ext=config.get("shared_library_suffix"), domain="c")
def create_shared_lib(filename, c_code, cc, cflags):
    """
    Compiles a C function into a shared library and returns the path.

    :param c_code: The C code as a string.
    :return: The result of the function call.
    """
    tmp_dir = Path(config.get("data_path")) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Create a temporary directory to store the C file and shared library
    with tempfile.TemporaryDirectory(prefix=str(tmp_dir)) as staging_dir:
        staging_dir = Path(staging_dir)
        c_file_path = staging_dir / "temp.c"
        shared_lib_path = Path(filename)

        # Write the C code to a file
        c_file_path.write_text(c_code)

        # Compile the C code into a shared library
        compile_command = [
            str(cc),
            *cflags,
            "-o",
            str(shared_lib_path),
            str(c_file_path),
        ]
        if not shutil.which(cc):
            raise FileNotFoundError(
                f"Compiler '{cc}' not found. Ensure it is installed and in your PATH."
            )
        subprocess.run(compile_command, check=True)
        assert shared_lib_path.exists(), f"Compilation failed: {compile_command}"


@lru_cache(maxsize=10_000)
def get_c_function(function_name, c_code):
    """
    :param function_name: The name of the function to call.
    :param c_code: The code to compile
    """
    shared_lib_path = create_shared_lib(
        c_code,
        config.get("cc"),
        [*config.get("cflags").split(), *config.get("shared_cflags").split()],
    )

    # Load the shared library using ctypes
    shared_lib = ctypes.CDLL(str(shared_lib_path))

    # Get the function from the shared library
    return getattr(shared_lib, function_name)


class CKernel:
    """
    A class to represent a C kernel.
    """

    def __init__(self, function_name, c_code):
        self.function_name = function_name
        self.c_code = c_code
        self.c_function = get_c_function(function_name, c_code)

    def __call__(self, *args):
        """
        Calls the C function with the given arguments.
        """
        return self.c_function(*map(methodcaller("to_c"), args))
