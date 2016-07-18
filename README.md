##ABCPRC
ABCPRC is an Approximate Bayesian Computation Particle Rejection Scheme designed to perform model fitting
on individual-based models.

##Setup

To setup, first download a local copy and then run
```python
python setup.py install
```

##Introduction

Import as
```python
import ABCPRC as prc
```
A fitting class is setup using
```python
m = prc.ABC()
```
You can then either use the built-in tolerances or fit your own using
```python
m.fit()
```
The fitting can then be performed using
```python
m.run(num_particles)
```
and the results shown (using seaborn),
```python
m.trace(plot=True)
```
##Tutorials

Two example tutorials accompany this package.
* [ecology example](Tutorial_Ecology.ipynb)
* [epidemiology example](Tutorial_Epidemiology.ipynb)

##Testing

Tests are run using the ```python nose2 ``` package. To install run
```python
pip install nose2
pip install cov-core
```
and tests can be performed running the command

```python
nose2 --with-coverage
```
