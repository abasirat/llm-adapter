from setuptools import setup, find_packages

setup(
    name="llm-adapter",
    version="0.1.0",
    description="A library for adapting language models to new languages and tasks",
    author="Ali Basirat",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "wechsel",
        "tqdm",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
