import numpy as np


def _construct_jitter(A, ϕ, ω, N=100, τ=0):
    if isinstance(N, float): N = int(np.round(N))
    x = np.linspace(0, 2 * np.pi, N)
    x += τ
    y = np.zeros_like(x)
    for idx, val in enumerate(range(A.size)):
        y += A[idx] * np.sin(ω[idx] * (x + ϕ[idx]))
    return y


def create_artificial_jitter_1d(N, n=2):
    # create random amplitude, phase, frequency
    A, ϕ, ω = np.zeros((n)), np.zeros((n)), np.zeros((n))
    for idx, val in enumerate(range(n)):
        A[idx] = np.random.rand(1) * np.power(val + 2, 2)
        ϕ[idx] = np.random.rand(1) * 2 * np.pi
        ω[idx] = np.random.uniform(low=np.power(val + 1, 2),
                                   high=np.power(val + 2, 2),
                                   size=(1,))
    A = np.flip(A)

    τ = np.random.randint(low=3, high=13)
    #    τ = np.random.uniform(low=np.minimum(.5, N/1e2),
    #                          high=np.maximum(1, N/1e1), size=(1,))

    # make reconstruction
    y_0 = _construct_jitter(A, ϕ, ω, N)
    y_1 = _construct_jitter(A, ϕ, ω, N, τ)
    g = y_0 - y_1
    return g, τ, A, ϕ, ω, N
