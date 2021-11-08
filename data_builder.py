"""
Author: Shaan Desai
Code to build simulated physics datasets

Adapted,in part, from https://github.com/greydanus/hamiltonian-nn
"""

import numpy as np
from scipy.integrate import solve_ivp as rk
import autograd
import autograd.numpy as np
solve_ivp = rk


def get_dataset(data_name, num_samples, T_max, dt, noise_std=0, seed=0, typev=1):
    """
    Args:
        data_name: str, from list 'mass_spring','n_spring','n_grav','pendulum','dpendulum','heinon'
        num_samples: total number of initial conditions to sample
        T_max: maximimum time to integrate to
        dt: integration time step
        noise_std: noise scaling on gaussian
        seed: random seed for numpy
    """

    dataset_list = ['mass_spring', 'pendulum', 'dpendulum', 'heinon', 'painleve_I', 'duffing', 'forced_mass_spring',
                    'forced_pendulum', 'dpend', 'damped', 'relativity', 'spring_coupled']
    if data_name not in dataset_list:
        raise ValueError('data name not in data list')

    if data_name == 'mass_spring':
        return mass_spring(num_samples, T_max, dt, noise_std, seed)
    if data_name == 'pendulum':
        return pendulum(num_samples, T_max, dt, noise_std, seed)
    if data_name == 'heinon':
        return heinon_heiles(num_samples, T_max, dt, noise_std, seed)
    if data_name == 'painleve_I':
        return painleve_I(num_samples, T_max, dt, noise_std, seed)
    if data_name == 'forced_mass_spring':
        return forced_mass_spring(num_samples, T_max, dt, noise_std, seed, typev)
    if data_name == 'forced_pendulum':
        return forced_pendulum(num_samples, T_max, dt, noise_std, seed, typev)
    if data_name == 'duffing':
        return duffing(num_samples, T_max, dt, noise_std, seed, typev)
    if data_name == 'dpend':
        return dpend_adapted(num_samples, T_max, dt, noise_std, seed, yflag=False)
    if data_name == 'damped':
        return damped(num_samples, T_max, dt, noise_std, seed)
    if data_name == 'relativity':
        return relativity(num_samples, T_max, dt, noise_std, seed, typev)
    if data_name == 'spring_coupled':
        return spring_coupled(num_samples, T_max, dt, noise_std, seed, typev)


def spring_coupled(num_samples, T_max, dt, noise_std=0, seed=3, type=1):
    def dynamics_fn(t, coords):
        m = 2
        k = 3

        ma1 = -k * coords[0] + k * (coords[1] - coords[0]) + np.cos(t)
        ma2 = -k * (coords[1] - coords[0]) + k * (-coords[1])
        return np.stack([coords[2] / m, coords[3] / m, ma1, ma2])

    def hamiltonian_eval(coords):
        m = 2
        k = 3
        q1 = coords[:, 0]
        q2 = coords[:, 1]
        p1 = coords[:, 2]
        p2 = coords[:, 3]
        H = (p1 ** 2) / (2 * m) + (p2 ** 2) / (2 * m) + k * q1 ** 2 + k * q2 ** 2 - k * q1 * q2
        return H

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        qs = np.random.uniform(-0.5, 0.5, size=2)
        vs = np.random.uniform(-0.2, 0.2, size=2)
        m = 2
        k = 3
        y0 = np.concatenate([qs, m * vs])
        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][:2], spring_ivp['y'][2:]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.concatenate([x, y]).T)
        energies.append(hamiltonian_eval(np.concatenate([x, y]).T))
        dxs.append(np.concatenate([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def pendulum(num_samples, T_max, dt, noise_std=0, seed=3):
    """simple pendulum"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)
        H = 9.81 * (1 - cos(q)) + (p ** 2) / 2  # pendulum hamiltonian
        return H

    def hamiltonian_eval(coords):
        H = 9.81 * (1 - np.cos(coords[:, 0])) + (coords[:, 1] ** 2) / 2  # pendulum hamiltonian
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt, radius=None, y0=None, **kwargs):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        if y0 is None:
            y0 = np.random.rand(2) * 2. - 1
        if radius is None:
            radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius

        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    # tuneable subsampling rate
    ssr = 1
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append(hamiltonian_eval(xs[-1]))
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def heinon_heiles(num_trajectories, T_max, dt, noise_std=0, seed=0):
    """heinon heiles data generator"""

    def hamiltonian_fn(coords):
        x, y, px, py = np.split(coords, 4)
        lambda_ = 1
        H = 0.5 * px ** 2 + 0.5 * py ** 2 + 0.5 * (x ** 2 + y ** 2) + lambda_ * (
                (x ** 2) * y - (y ** 3) / 3)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=1):

        # get initial state
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        px = np.random.uniform(-.5, .5)
        py = np.random.uniform(-.5, .5)

        y0 = np.array([x, y, px, py])

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-10)
        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]
        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))

        return accum, np.array(daccum), energies, np.arange(0, t_span[1], timescale)

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    data = {}
    ssr = 1  # int(sub_sample_rate / dt)

    xs, dxs, energies, times = [], [], [], []
    for s in range(num_trajectories):
        x, dx, energy, time = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate)

        x += np.random.randn(*x.shape) * noise_std
        dx += np.random.randn(*dx.shape) * noise_std

        xs.append(x)
        dxs.append(dx)
        energies.append(energy)
        times.append(time)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs)
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = times

    return data


from autograd.numpy import cos, sin


def dpend_adapted(num_trajectories, T_max, dt, noise_std=0, seed=0, yflag=False):
    """heinon heiles data generator"""

    def hamiltonian_fn(coords):
        t1, t2, pt1, pt2 = np.split(coords, 4)
        numerator = pt1 ** 2 + 2 * pt2 ** 2 - 2 * pt1 * pt2 * cos(t1 - t2)
        denominator = 2 * (1 + sin(t1 - t2) ** 2)
        H = (numerator / denominator) - 2 * 9.81 * cos(t1) - 9.81 * cos(t2)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=dt, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        t1 = np.random.uniform(-np.pi, np.pi)
        t2 = np.random.uniform(-np.pi, np.pi)
        pt1 = 0  # np.random.uniform(-np.pi/10, np.pi/10)
        pt2 = 0  # np.random.uniform(-np.pi/10, np.pi/10)

        y0 = np.array([t1, t2, pt1, pt2])

        if yflag:
            y0 = [-0.53202021, -0.38343444, -2.70467816, 0.98074028]

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-10)
        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]
        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))
        # print(energies[-1] - energies[0], len(energies))
        return accum, np.array(daccum), energies, np.arange(0, t_span[1], timescale)

    def get_dataset(num_trajectories, T_max, dt, seed=seed, test_split=0.5,
                    **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs

        data = {}
        ssr = 1  # int(sub_sample_rate / dt)

        xs, dxs, energies = [], [], []
        time = []
        for s in range(num_trajectories):
            x, dx, energy, times = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=dt)

            x += np.random.randn(*x.shape) * noise_std
            dx += np.random.randn(*dx.shape) * noise_std
            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            time.append(times)

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['tvalues'] = np.concatenate(time)

        return data

    np.random.seed(seed)
    return get_dataset(num_trajectories, T_max, dt)


def mass_spring(num_trajectories, T_max, dt, noise_std, seed):
    """1-body mass spring system"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)

        H = (p ** 2) / 2 + (q ** 2) / 2  # spring hamiltonian (linear oscillator)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=dt, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        if y0 is None:
            y0 = np.random.rand(2) * 2 - 1
        if radius is None:
            radius = np.sqrt(np.random.uniform(1, 4.5))  # np.random.rand() * 0.9 + 0.1  # sample a range of radii
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * (radius)  ## set the appropriate radius

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-9, atol=1e-9)

        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]

        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))

        return accum, np.array(daccum), energies, np.arange(0, t_span[1], timescale)

    def get_dataset(num_trajectories, T_max, dt, seed=seed):
        data = {}

        # randomly sample inputs
        np.random.seed(seed)
        data = {}
        ssr = 1  # int(sub_sample_rate / dt)

        xs, dxs, energies, times = [], [], [], []
        for s in range(num_trajectories):
            x, dx, energy, time = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=dt)

            x += np.random.randn(*x.shape) * noise_std
            # dx += np.random.randn(*dx.shape) * noise_std

            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            times.append(time)

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['tvalues'] = np.concatenate(times)

        return data

    return get_dataset(num_trajectories, T_max, dt)


def painleve_I(num_samples, T_max, dt, noise_std=0, seed=3):
    """simple pendulum"""

    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        H = p ** 2 / 2 - 2 * q ** 3 - t * q
        return H

    def hamiltonian_eval(coords, t):
        H = (coords[:, 1] ** 2) / 2 - 2 * coords[:, 0] ** 3 - t * coords[:, 0]
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords, t)

        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        y0 = np.random.rand(2) * 2 - 1
        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)

    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append(hamiltonian_eval(xs[-1], t))
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def relativity(num_samples, T_max, dt, noise_std=0, seed=3, type=1):
    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        c = 1
        m = 1
        if type == 1:
            omega = 1.2
            delta = 0
            gamma = 0.2
            alpha = 1
            beta = 1
        if (type == 2) or (type == 3):
            alpha = -1
            beta = 1
            omega = 1.2
            delta = 0
            gamma = 10
        H = c * np.sqrt(p ** 2 + m ** 2 * c ** 2) + alpha * (q ** 2) / 2 + beta * (q ** 4) / 4 - q * gamma * sin(
            omega * t)

        return H

    # autograd slows the computation for long trajectories
    def dynamics_fn(t, coords):
        q, p = np.split(coords, 2)

        c = 1
        m = 1

        if type == 1:
            omega = 1.2
            delta = 0
            gamma = 0.2
            alpha = 1
            beta = 1

        if (type == 2) or (type == 3):
            alpha = -1
            beta = 1
            omega = 1.2
            delta = 0
            gamma = 10

        xdot = c * p * (p ** 2 + (m ** 2) * (c ** 2)) ** (-.5)
        pdot = -alpha * q - beta * q ** 3 - delta * xdot + gamma * np.sin(omega * t)

        S = np.concatenate([xdot, pdot], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        y0 = np.random.uniform(0, 2, size=2)
        if type == 2:
            # y0 = [0,0.9]#np.random.rand(2) * 2 - 1
            y0 = [0., 0.]
            omega = 1.2
            dt_per_period = 1000
            period = 2 * np.pi / omega
            dt = 2 * np.pi / omega / dt_per_period
            t_span = [0, 10 * period]
            t_eval = np.arange(0, 10 * period, dt)
        if type == 3:
            # y0 = np.random.rand(2) * 2 - 1
            y0 = [0., 0.]
            omega = 1.2
            dt_per_period = 100
            period = 2 * np.pi / omega
            dt = 2 * np.pi / omega / dt_per_period
            t_span = [0, T_max]
            t_eval = np.arange(0, T_max, dt)
        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append([hamiltonian_fn(xs[-1][i], t[i]) for i in range(len(xs[-1]))])
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def forced_mass_spring(num_samples, T_max, dt, noise_std=0, seed=3, type=1):
    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        if type == 1:
            omega = 3
            H = q ** 2 / 2 + p ** 2 / 2 - q * sin(omega * t)  # pendulum hamiltonian

        if type == 2:
            omega = 3
            H = q ** 2 / 2 + p ** 2 / 2 - q * sin(omega * t) * sin(2 * omega * t)  # pendulum hamiltonian

        if type == 3:
            omega = 3
            H = q ** 2 / 2 + p ** 2 / 2 - q * 10  # pendulum hamiltonian

        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords, t)

        dqdt, dpdt = np.split(dcoords, 2)
        if type == 5:
            S = np.concatenate([np.array([coords[1]]), -dqdt], axis=-1)
        else:
            S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        y0 = np.random.rand(2) * 2 - 1
        radius = np.sqrt(np.random.uniform(1., 4.5))  # np.random.rand() * 0.9 + 0.1  # sample a range of radii
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * (radius)

        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append([hamiltonian_fn(xs[-1][i], t[i]) for i in range(len(xs[-1]))])
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def duffing(num_samples, T_max, dt, noise_std=0, seed=1, type=1):
    """simple pendulum"""

    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        alpha = 1
        beta = 1
        omega = 1.2
        delta = 0
        gamma = .2
        H = alpha * (q ** 2) / 2 + (p ** 2) / 2 + beta * (q ** 4) / 4 - q * gamma * sin(
            omega * t)  # pendulum hamiltonian
        return H

    def dynamics_fn(t, coords):
        # dcoords = autograd.grad(hamiltonian_fn)(coords, t)

        # dqdt, dpdt = np.split(dcoords, 2)
        # S = np.concatenate([dpdt, -dqdt], axis=-1)
        q, p = np.split(coords, 2)
        alpha = -1
        beta = 1
        omega = 1.2
        delta = 0.3
        gamma = .2
        if type == 1:
            S = np.concatenate([p, -alpha * q - beta * q ** 3 - delta * p + gamma * np.sin(omega * t)], axis=-1)
        if type == 2:
            gamma = 0.5
            S = np.concatenate([p, -alpha * q - beta * q ** 3 - delta * p + gamma * sin(omega * t)], axis=-1)
        if type == 3:
            # alpha =1
            # beta = 5
            omega = 1.4
            gamma, delta = 0.39, 0.1
            S = np.concatenate([p, q - q ** 3 - delta * p + gamma * cos(omega * t)], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        y0 = np.random.rand(2) * 5 - 2.5
        # radius = np.sqrt(np.random.uniform(0.5, 1.5))  # np.random.rand() * 0.9 + 0.1  # sample a range of radii
        # y0 = y0 / np.sqrt((y0 ** 2).sum()) * (radius)
        if type == 3:
            y0 = np.random.rand(2) * 2 - 1
            # y0 = [0., 0.]
            omega = 1.4
            dt_per_period = 100
            period = 2 * np.pi / omega
            dt = 2 * np.pi / omega / dt_per_period
            t_span = [0, period]
            t_eval = np.arange(0, period, dt)
        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append([hamiltonian_fn(xs[-1][i], t[i]) for i in range(len(xs[-1]))])
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def damped(num_samples, T_max, dt, noise_std=0, seed=1, type=1):
    """simple pendulum"""

    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        alpha = 1
        beta = 1
        omega = 1.2
        delta = 0
        gamma = .2
        H = alpha * (q ** 2) / 2 + (p ** 2) / 2 + beta * (q ** 4) / 4 - q * gamma * sin(
            omega * t)  # pendulum hamiltonian
        return H

    def dynamics_fn(t, coords):
        # dcoords = autograd.grad(hamiltonian_fn)(coords, t)

        # dqdt, dpdt = np.split(dcoords, 2)
        # S = np.concatenate([dpdt, -dqdt], axis=-1)
        q, p = np.split(coords, 2)
        alpha = 1
        beta = 1
        omega = 1.2
        delta = 0.3
        gamma = 0
        # if type == 1:
        S = np.concatenate([p, -q - delta * p], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        y0 = np.random.rand(2) * 2 - 1
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append([hamiltonian_fn(xs[-1][i], t[i]) for i in range(len(xs[-1]))])
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


def forced_pendulum(num_samples, T_max, dt, noise_std=0, seed=3, type=1):
    """simple pendulum"""

    def hamiltonian_fn(coords, t):
        q, p = np.split(coords, 2)
        omega = 6
        H = 9.81 * (1 - cos(q)) + (p ** 2) / 2 - q * sin(omega * t)  # pendulum hamiltonian
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords, t)

        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        y0 = np.random.rand(2) * 2 - 1
        radius = np.sqrt(np.random.uniform(0.5, 1.5))  # np.random.rand() * 0.9 + 0.1  # sample a range of radii
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * (radius)

        spring_ivp = rk(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn(t_eval[i], y) for i, y in enumerate(spring_ivp['y'].T)]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    ssr = 1  # int(srate / dt)
    energies = []
    tvalues = []
    for s in range(num_samples):
        x, y, dx, dy, t = get_trajectory()
        x = x[::ssr]
        y = y[::ssr]
        dx = dx[::ssr]
        dy = dy[::ssr]
        xs.append(np.stack([x, y]).T)
        energies.append([hamiltonian_fn(xs[-1][i], t[i]) for i in range(len(xs[-1]))])
        dxs.append(np.stack([dx, dy]).T)
        tvalues.append(t)

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['energy'] = np.concatenate(energies)
    data['tvalues'] = np.concatenate(tvalues)
    return data


if __name__ == '__main__':
    train_data = get_dataset('mass_spring', 1, 10.01, 0.01, noise_std=0, seed=1, typev=1)
    print(train_data['energy'][0] - train_data['energy'][-1])
