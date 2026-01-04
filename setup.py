from setuptools import setup, find_packages

setup(
    name='ash',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'requests',
        'torchvision',
        'tqdm',
        'wandb',
        'torchinfo'
    ],
    description='Shivam\'s (arshiv) custom Pytorch research toolkit',
)