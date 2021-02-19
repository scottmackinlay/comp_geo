import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="comp_geo",  # TODO: Change this to the name of your module
    version="0.0.1",
    author="Scott Mackinlay",  # TODO: Change this to your name
    description="Computational geometry notes and exercises.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # TODO: Change this to your repo URL
    url="https://github.com/scottmackinlay/comp_geo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # Pip-installable packages which are required for your
    # module to function are listed here
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
