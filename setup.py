import codecs
import os

from setuptools import find_packages, setup

NAME = "Jeta"
DESCRIPTION = "A JAX based meta learning library"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

PROJECT = os.path.abspath(os.path.dirname(__file__))
EXCLUDE = ()
VERSION = "0.1.0"
AUTHOR = "Society for Artificial Intelligence and Deep Learning"
EMAIL = "vedantshah2012@gmail.com"
LICENSE = "MIT"
REPO_URL = "https://github.com/SforAiDl/jeta"
PROJECT_URLS = {"Source": REPO_URL}
CLASSIFIERS = (
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
)


def read(*parts):
    """
    returns contents of file
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as file:
        return file.read()


def get_requires(path):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


INSTALL_REQUIRES = list(get_requires("requirements.txt"))


CONFIG = {
    "name": NAME,
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "packages": find_packages(where=PROJECT, exclude=EXCLUDE),
    "version": VERSION,
    "author": AUTHOR,
    "author_email": EMAIL,
    "license": LICENSE,
    "url": REPO_URL,
    "project_urls": PROJECT_URLS,
    "classifiers": CLASSIFIERS,
    "install_requires": INSTALL_REQUIRES,
    "python_requires": ">=3.6",
}

if __name__ == "__main__":
    setup(**CONFIG)
