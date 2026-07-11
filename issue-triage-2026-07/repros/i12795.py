import jax.numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
rng = np.random.RandomState(0)
N = 3

x_L = np.tril(rng.randn(N, N) + 1j * rng.randn(N, N))
x_sym = x_L + x_L.conj().T
x_sym[np.diag_indices(N)] /= 2

def eigvalsh_lower(x):
  return jax.lax.linalg.eigh(x, lower=True, symmetrize_input=False)[1]

def first_eigvalh_lower(x):
  return eigvalsh_lower(x)[0]

print("eigvals equal:", np.allclose(eigvalsh_lower(x_L), eigvalsh_lower(x_sym)))
g_L = jax.grad(first_eigvalh_lower)(x_L)
g_sym = jax.grad(first_eigvalh_lower)(x_sym)
print("grad at x_L:\n", g_L)
print("grad at x_sym:\n", g_sym)
print("grads allclose:", np.allclose(g_L, g_sym))

# finite-difference check of the x_L gradient (only lower triangle matters
# since symmetrize_input=False reads only lower triangle)
eps = 1e-6
fd = np.zeros((N, N), dtype=complex)
for a in range(N):
  for b in range(a + 1):
    for part, delta in [(1.0, eps), (1j, 1j * eps)]:
      dx = np.zeros((N, N), dtype=complex); dx[a, b] = delta
      d = (first_eigvalh_lower(x_L + dx) - first_eigvalh_lower(x_L - dx)) / (2 * eps)
      if part == 1.0: fd[a, b] += d.real
      else: fd[a, b] += 1j * d.real
# jax.grad for real-valued f of complex input returns conj of the CR-gradient such that
# df = Re(sum(grad * conj(dx)))? Convention: df ~ Re(vdot(grad, dx)) with grad = conj of derivative.
print("finite-diff (df/dRe + i df/dIm) lower triangle:\n", fd)
print("conj(grad_L):\n", np.conj(g_L))
print("fd matches conj(grad at x_L):", np.allclose(fd, np.conj(g_L), atol=1e-5))
print("fd matches grad at x_L:", np.allclose(fd, g_L, atol=1e-5))
