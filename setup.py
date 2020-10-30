import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLToolsShed",
    version="1.0.0",
    author="marcoromanelli-github",
    author_email="marcoromane@gmail.com",
    description="A set of tools to make it easier to program with TensorFlow and Keras.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcoromanelli-github/MLToolsShed",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
