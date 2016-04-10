##ABCPRC
ABCPRC is an Approximate Bayesian Computation Particle Rejection Scheme designed to perform model fitting
on individual-based models.

##Setup

to setup run:
python setup.py install

##Introduction

Import as
import ABCPRC as prc

A fitting class is setup using

m = prc.ABC()

You can then either use the built-in tolerances or fit your own using

m.fit()

The fitting can then be performed using

m.run(num_particles)

and the results shown (using seaborn),

m.trace(plot=True)