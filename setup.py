from setuptools import setup, find_packages

setup(
    name="mind",
    version="0.0.1",
    description="Robodog Mind",
    author="Your Name",
    author_email="yourname@gamil.com",
    package_dir={"": "src"},  # Tell setuptools that code is inside src/
    packages=find_packages(where="src"),
)