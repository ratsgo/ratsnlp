import setuptools

setuptools.setup(
    name="ratsnlp",
    version="0.0.997",
    license='MIT',
    author="ratsgo",
    author_email="ratsgo@naver.com",
    description="tools for Natural Language Processing",
    long_description=open('README.md').read(),
    url="https://github.com/ratsgo/ratsnlp",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data = {
        'ratsnlp.nlpbook.classification': ['*.html'],
        'ratsnlp.nlpbook.ner': ['*.html'],
        'ratsnlp.nlpbook.qa': ['*.html'],
        'ratsnlp.nlpbook.paircls': ['*.html'],
        'ratsnlp.nlpbook.generation': ['*.html'],
    },
    install_requires=[
        "torch>=1.7.1",
        "pytorch-lightning==1.2.3",
        "transformers==4.2.2",
        "tqdm>=4.46.0",
        "Korpora>=0.1.0",
        "flask==1.1.2",
        "flask_ngrok>=0.0.25",
        "flask_cors>=3.0.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)