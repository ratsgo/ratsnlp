import setuptools

setuptools.setup(
    name="ratsnlp",
    version="0.0.5",
    license='MIT',
    author="ratsgo",
    author_email="ratsgo@naver.com",
    description="tools for Natural Language Processing",
    long_description=open('README.md').read(),
    url="https://github.com/ratsgo/ratsnlp",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==1.5.1",
        "pytorch-lightning==0.8.5",
        "transformers==3.0.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)