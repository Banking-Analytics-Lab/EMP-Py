import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="EMP",
        version="0.0.1",
        author="Cristián Bravo",
        author_email="cbravoro@uwo.ca",
        description="Functions for estimating EMP (Expected Maximum Profit Measure) in Credit Risk Scoring and Customer Churn Prediction, according to Verbraken et al (2013, 2014) <DOI:10.1109/TKDE.2012.50>, <DOI:10.1016/j.ejor.2014.04.001>.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/Banking-Analytics-Lab/EMP-Py",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: GPL (>= 3)",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8'
    )
