"""setup.py — Build the neuro-analog CUDA extension.

Run on RunPod (GPU required):
    pip install -e .

Environment variables:
    TORCH_CUDA_ARCH_LIST: GPU architectures to compile for (default: 8.0;8.6)
        - A100: 8.0
        - RTX 3090: 8.6
        - V100: 7.0

This produces neuro_analog_cuda.cpython-3XX-x86_64-linux-gnu.so
"""
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUDA arch list from environment, default to A100 + RTX 3090
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0;8.6')

# Parse arch list into gencode flags
gencode_flags = []
for arch in arch_list.split(';'):
    gencode_flags.append(f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}')

setup(
    name='neuro_analog_cuda',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='neuro_analog_cuda',
            sources=[
                'analog_linear_cuda.cpp',
                'analog_linear_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xptxas', '-dlcm=ca',  # Use L1 cache for loads
                ] + gencode_flags,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            no_python_abi_suffix=True,
            build_directory=os.path.join(os.path.dirname(__file__), 'build'),
        )
    },
    python_requires='>=3.8',
)
