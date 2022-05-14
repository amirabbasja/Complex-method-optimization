# Complex-method-optimization
Complex optimization method by Box

The Complex method was first presented by Box [1], and later improved by Guin [2]. The method is a constraint simplex method, hence the name Complex, developed from the Simplex method by Spendley et al [3] and Nelder Mead. Similar related methods go under names such as Nelder-Mead Simplex. The main difference between the Simplex method and the complex method is that the Complex method uses more points during the search process.

For the full description of the method, please see [Here](http:/https://complexmethod.readthedocs.io/en/latest/Description.html#the-complex-method/ "Here")

Also in the code, there are two refinements to the original idea:
1. Sometimes Xc is also not acceptable, so there is no point in moving towards it.
2.Some times despite decreasing the alpha, we are not solving the problem in Xr, so we have to leave the point and start from a new one.

------------


Refs:
1. Box, M.J., “A new method of constrained optimization and a comparison with other method”, Computer Journal, Volume 8, No. 1, pages 42-52, 1965.
2. Guin J. A., “Modification of the Complex method of constraint optimization”, Computer Journal, Volume 10, pages 416-417, 1968.
3.Spendley W., Hext G. R., and Himsworth F. R., “Sequential application of Simplex designs in optimisation and evolutionary operation”, Technometrics, Volume 4, pages 441-462, 1962
