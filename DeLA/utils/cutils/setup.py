from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCES = [
    str(path) for path in sorted((ROOT / "srcs").iterdir())
    if path.suffix in {".cpp", ".cu"}
]

setup(
    name="dela-cutils",
    version="1.0",
    ext_modules=[CUDAExtension(
        name="dela_cutils_ext",
        sources=SOURCES,
        extra_compile_args={
            "cxx": ["-O3", "-mavx2", "-funroll-loops"],
            "nvcc": ["-Xptxas", "-v"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
