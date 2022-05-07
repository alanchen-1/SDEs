# Ornstein Uhlenbeck


# Feller Diffusion
can we do FSA and find expected value of extinction time
can we use martingale too?
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

it makes sense that shit is a degenerate process when beta = inf
all residuals start at 1. in diag case, this becomes identity, but in this case, this becomes
full matrix of all 1s.
in a matrix with all elements same, the sampled random vector will be another constant vector
so then we will get a bunch of things the same -> residuals same -> repeat
so it's a degenerate process - all the elements are the same for all time (until extinct, of course)

try using t-SNE projection to plot these in 2d? see what that means idk lol
or other projection methods

higher step sizes result in longer processes

# Filtered Matrix Completion

