import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NBSVM",
    version="0.1",
    author="Chris Wallace",
    author_email="cjwallace@cloudera.com",
    description="Tool for benchmarking NLP classification problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fastforwardlabs/nbsvm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: new BSD",
        "Operating System :: OS Independent",
    ],
)