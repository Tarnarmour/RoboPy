from setuptools import setup, find_packages

setup(
    name='RoboPy',
    version='1.0',
    description='Python robotics package for serial robot arms',
    author='John Morrell',
    author_email='Tarnarmour@gmail.com',
    packages=find_packages('RoboPy'),
    install_requires=['numpy', 'scipy', 'pyqtgraph', 'pyopengl', 'matplotlib', 'sympy']
)
