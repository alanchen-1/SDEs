import sys
sys.path.append('..')
import unittest
import numpy as np
from utils.filtered_matrix_completion import random_matrix_wishart, rescale

class TestFiltered(unittest.TestCase):
    """
    Testing for FilteredMatrix class.
    """
    def test_rescale(self):
        """
        Quick test case for singular value rescaling.
        """
        matrix = random_matrix_wishart(100, 50)
        _, s, _ = np.linalg.svd(rescale(matrix))
        self.assertAlmostEqual(np.amax(s), 0.5, places=6)

if __name__ == '__main__':
    unittest.main()

