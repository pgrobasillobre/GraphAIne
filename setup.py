from setuptools import setup, find_packages

setup(
    name="GraphAIne",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "tensorflow",
        "pandas",
    ],
    include_package_data=True,
    description="A deep learning framework for graphene simulations",
    author="Pablo Grobas Illobre",
    author_email="pgrobasillobre@gmail.com",
    url="https://github.com/pgrobasillobre/GraphAIne",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)

