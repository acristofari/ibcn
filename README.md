# Block cubic Newton with greedy selection

This repository contains the files related to the experiments reported in

[A. Cristofari (2026). Block cubic Newton with greedy selection. Optimization Methods and Software, to appear](https://doi.org/10.1080/10556788.2026.2692465).

In this paper, a second-order block coordinate descent method is proposed, named _Inexact Block Cubic Newton_ (IBCN) method,
using a greedy rule for block selection and a cubic Newton strategy for block updates.

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

Copyright 2025-2026 Andrea Cristofari

## Usage

All codes are in Matlab. Two classes of unconstrained problems are considered, as described in the above paper.

1. For sparse least squares, run the file `main_sp_ls.m`.

2. For l2-regularized logistic regression, first download the datasets as indicated in the file `main_l2_log_reg.m`, then run it.