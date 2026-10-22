import numpy as np
import pytest


@pytest.fixture(scope="module")
def div(load_script):
    return load_script("diversity")


def test_distance_matrices_haploid_with_missing(div):
    # (n_variants, n_samples, ploidy), haploid parsed as diploid "x/."
    g = np.array(
        [
            [[0, -1], [1, -1], [1, -1]],
            [[0, -1], [0, -1], [-1, -1]],
            [[1, -1], [0, -1], [1, -1]],
        ]
    )
    divergence, differences = div.compute_distance_matrices(g)
    # pairs (0,1), (0,2), (1,2); sample 2 is missing at variant 2
    np.testing.assert_array_equal(differences, [2, 1, 1])
    np.testing.assert_allclose(divergence, [2 / 3, 1 / 2, 1 / 2])
