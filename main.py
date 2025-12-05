# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def random_isotropic(rng):
    """Return a random unit vector uniformly distributed over the sphere."""
    u = rng.uniform(-1, 1)
    phi = rng.uniform(0, 2*np.pi)
    sqrt_one_minus_u2 = np.sqrt(1 - u*u)
    return np.array([sqrt_one_minus_u2*np.cos(phi),
                     sqrt_one_minus_u2*np.sin(phi),
                     u])

def simulate_photon(L, mu_s, rng):
    """Simulate a single photon in a slab of thickness L.
    Returns total traveled distance and exit side (1 for transmitted, -1 for reflected)."""
    pos = np.array([0.0, 0.0, 0.0])
    direction = random_isotropic(rng)
    total_dist = 0.0
    while True:
        step = -np.log(rng.random()) / mu_s
        pos += direction * step
        total_dist += step
        if pos[2] < 0.0:
            return total_dist, -1
        if pos[2] > L:
            return total_dist, 1
        direction = random_isotropic(rng)

def experiment1():
    """Monte Carlo photon random walk: histogram of total path lengths."""
    L = 1.0
    mu_s = 1.0
    N = 200000
    rng = np.random.default_rng(42)
    path_lengths = np.empty(N)
    for i in range(N):
        pl, _ = simulate_photon(L, mu_s, rng)
        path_lengths[i] = pl
    plt.figure()
    counts, bins, _ = plt.hist(path_lengths, bins=100, density=True, alpha=0.6, label='Simulation')
    x = np.linspace(0, bins[-1], 500)
    lam = mu_s
    pdf = lam * np.exp(-lam * x)
    plt.plot(x, pdf, 'r-', label='Exponential (mean free path)')
    plt.xlabel('Total path length')
    plt.ylabel('Probability density')
    plt.title('Photon total path length distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('path_length_distribution.png')
    plt.close()
    return path_lengths.mean()

def experiment2():
    """Transmission vs slab thickness and comparison with Beer–Lambert law."""
    mu_s = 1.0
    thicknesses = np.linspace(0.1, 3.0, 30)
    N = 50000
    rng = np.random.default_rng(123)
    transmitted = []
    for L in thicknesses:
        count = 0
        for _ in range(N):
            _, side = simulate_photon(L, mu_s, rng)
            if side == 1:
                count += 1
        transmitted.append(count / N)
    transmitted = np.array(transmitted)
    theory = np.exp(-mu_s * thicknesses)
    plt.figure()
    plt.plot(thicknesses, transmitted, 'bo', label='Monte Carlo')
    plt.plot(thicknesses, theory, 'r-', label='Beer–Lambert')
    plt.xlabel('Slab thickness')
    plt.ylabel('Transmitted fraction')
    plt.title('Transmission vs slab thickness')
    plt.legend()
    plt.tight_layout()
    plt.savefig('transmission_vs_thickness.png')
    plt.close()

def main():
    avg_path = experiment1()
    experiment2()
    print('Answer:', avg_path)

if __name__ == '__main__':
    main()

