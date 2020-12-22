import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bigsimr",
    version="0.0.1",
    author="Alex Knudson",
    author_email="aknudson@nevada.unr.edu",
    description="simulate high-dimensional multivariate data with arbitrary marginal distributions",
    long_description=long_description,
    url="https://github.com/SchisslerGroup/python-bigsimr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    python_requires='>=3.6'
)