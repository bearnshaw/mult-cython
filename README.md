# mult-cython

This is a demonstration of an issue using the [parallel
package](http://cython.readthedocs.io/en/latest/src/userguide/parallelism.html)
in [Cython](http://cython.readthedocs.io/en/latest/index.html). Cython uses
[OpenMP](http://www.openmp.org/) to implement parallel computing.  In November
2015, [OpenMP
4.5](http://www.openmp.org/uncategorized/openmp-45-specs-released/) was released
and included the ability to do reductions on c/c++
arrays. GCC 6.1 is the [first version of
GCC](http://www.openmp.org/resources/openmp-compilers/) to support OpenMP 4.5.
Apparently, Cython still hasn't implemented these reductions, as demonstrated by
the code in this repo.

I opened an [issue](https://github.com/cython/cython/issues/1504) on Cython's
github page asking about adding support.

## Installation
Install and activate the conda environment
```bash
$ conda env create
$ source activate mult
```
Compile the Cython code by running
```bash
CC=/path/to/c python setup.py build_ext --inplace
```
where `/path/to/c` is a path to a c compiler with OpenMP support. Best way to
get one of those in MacOSX is to use [homebrew](http://brew.sh/) and install gcc
without multilib support:
```bash
$ brew install --without-multilib gcc
```

## Running
```bash
$ python test.py
```
Typical output is
```
a is 10 x 10
b is 10 x 10
norm of c: 26.341118998648835
compute a x b 1000 times using...

mult
number correct: 1000
number wrong  : 0
avg rel error : None
avg time per  : 0:00:00.000249

mult_broken
number correct: 868
number wrong  : 132
avg rel error : 0.022022445149214236
avg time per  : 0:00:00.000255
```
Notice that, roughly 10% of the time, `mult_broken` does not correctly calculate
the product of `a` and `b`. This is because, unlike `mult`, it requires a
non-trivial reduction (i.e., needs to gather non-zero results across multiple
threads) on arrays. It creates a race condition which only manifests itself a
fraction of the time.
