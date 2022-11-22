from setuptools import setup, find_packages

setup(name='ipm', version='1.0', packages=find_packages(), install_requires=['torch', 'numpy', 'gym', 'stable-baselines3[extra]'])

