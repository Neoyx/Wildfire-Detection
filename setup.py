from setuptools import setup, Extension
import pybind11

# Define compiler flags for optimization
cpp_args = ['-std=c++17', '-O3']

ext_modules = [
    Extension(
        # The name of the module in Python: import sequential_regioning_cpp
        'sequential_regioning_cpp', 
        # The source C++ file
        ['sequential_regioning.cpp'], 
        # Add pybind11's include directory
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='sequential_regioning_cpp',
    version='1.0.0',
    ext_modules=ext_modules,
)