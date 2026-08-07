from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="Cukd",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["Cukd"],
    package_dir={"Cukd": "CukdTorch"},
    ext_modules=[
        CUDAExtension(
            "Cukd.CuFun",
            ["cukd/api.cu"],
            extra_compile_args={
                "cxx": ["-g", "-O3", "-mavx2", "-funroll-loops"],
                "nvcc": ["-O2", "-Xptxas", "-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_dirs=["./"],
)
