import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EMP-PY",
    version="2.0.5",
    author="Cristián Bravo",
    author_email="cbravoro@uwo.ca",
    description="Functions for estimating EMP (Expected Maximum Profit Measure) in Credit Risk Scoring and Customer Churn Prediction, according to Verbraken et al (2013, 2014).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Banking-Analytics-Lab/EMP-Py",
    license_expression="GPL-3.0-or-later",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "scipy", "scikit-learn"],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
