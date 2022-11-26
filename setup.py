from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",  # How you named your package folder (MyLib)
    version="0.0.1",  # Start with a small number and increase it with every change you make
    author="Hari",  # Type in your name
    description="A small package for DVC_DL_Tensorflow_demo",  # Give a short description about your library
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/demohari/DVC_DL_Tensorflow_demo",  # Provide either the link to your github or to your website
    author_email="kishore_koneru@yahoo.com",  # Type in your E-Mail
    packages=["src"],
    python_requires=">=3.7",  # Specify which pyhton versions that you want to support
    install_requires=[
        "dvc",
        "tensorflow",
        "matplotlib",
        "numpy",
        "pandas",
        "tqdm",
        "PyYAML",
        "boto3" ,  # requirements
    ]
)