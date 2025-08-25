from setuptools import setup,find_packages

with open("requirements.txt") as f:
    Requirements = f.read().splitlines()

setup (
    name="NutriPlan-AI",
    version="0.1",
    author="shiva",
    packages=find_packages(),
    install_requires = Requirements,
)
