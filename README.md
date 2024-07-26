# Block cubic Newton with greedy selection

This repository contains the files related to the experiments reported in

[A. Cristofari. Block cubic Newton with greedy selection. _arXiv:2407.18150_](https://arxiv.org/abs/2407.18150).

In the above paper, a second-order block coordinate descent method is proposed, named _Inexact Block Cubic Newton_ (IBCN) method,
using a greedy rule for the block selection and cubic models for the block update.

## Author

Andrea Cristofari (e-mail: [andrea.cristofari@uniroma2.it](mailto:andrea.cristofari@uniroma2.it))

## Licensing

IBCN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
IBCN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with IBCN. If not, see <http://www.gnu.org/licenses/>.

## Usage

All codes are in Matlab. Two classes of unconstrained problems are considered, as described in the above paper.

1. For sparse least squares (non-convex problems), just run the file `main_sp_ls.m`.

2. For l2-regularized logistic regression (convex problems), first download the datasets `gisette`, `leu` and `madelon`
   from <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>. Using the `libsvmread` software, which can be downloaded from therein as well,
   convert the files into matlab files and save them as `gisette.mat`, `leu.mat` and `madelon.mat`, respectively.
   In each matlab file, the instance matrix must be a sparse matrix named `A` and the label vector must be a vector named `b`.
   Then, run the file `main_l2_log_reg.m`.