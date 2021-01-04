from setuptools import setup

setup(
    name="vizdoomgym",
    version="1.0",
    description="OpenAI Gym wrapper for ViZDoom",
    author="Simon Hakenes",
    author_email="simon.hakenes@ini.rub.de",
    install_requires=[
        "gym",
        "numpy",
        "pre-commit",
    ],
)
