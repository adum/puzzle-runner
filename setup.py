from setuptools import find_packages, setup

setup(
    name="puzzle-runner",
    version="0.1.0",
    description="Iterative benchmark orchestrator for AI coding agents.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "puzzle-runner=puzzle_runner.cli:main",
            "puzzle-runner-watch=puzzle_runner.watch:main",
        ]
    },
)
