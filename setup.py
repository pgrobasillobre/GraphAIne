from setuptools import setup, find_packages

setup(
    name="GraphAIne",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19",
        "tensorflow>=2.0",
    ],
    include_package_data=True,
    description="A deep learning framework for graphene simulations",
    author="Pablo Grobas Illobre",
    author_email="pgrobasillobre@gmail.com",
    url="https://github.com/pgrobasillobre/GraphAIne",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

