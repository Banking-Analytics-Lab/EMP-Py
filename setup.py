import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="EMP-PY",
        version="2.0.4",
        author="Cristián Bravo",
        author_email="cbravoro@uwo.ca",
        description="Functions for estimating EMP (Expected Maximum Profit Measure) in Credit Risk Scoring and Customer Churn Prediction, according to Verbraken et al (2013, 2014).",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/Banking-Analytics-Lab/EMP-Py",
        packages=setuptools.find_packages(),
        setup_requires=["numpy"],  
        install_requires=["numpy",'pandas','pandas','scipy','scikit-learn'],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8'
    )
