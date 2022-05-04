# Ornstein Uhlenbeck


# Feller Diffusion
can we do FSA and find expected value of extinction time
also prove that extinction time for expected value is finite

# Matrix Completion
The Euler Maruyama scheme for this is as follows:
$v_{k + 1} - v_{k} = \sqrt{Pbeta}dB^n$
Covariance Pbeta

this is nontrivial n dimensions
we know we want var(\sqrt{Pbeta} dBn) 
this expands to
\sqrt{Pbeta} var(dBn) \sqrt{Pbeta}^T
then we know that var(dBn) is covariance matrix of n by n iid with variance h (step size)
so we know that dBn is just diag(h) 
im pretty sure the final covariance is just hPbeta (elementwise)

