#!/usr/bin
#!nvidia-smi

# some necessary packages
#!pip install -q dm-haiku  # neural network library
#!pip install -q optax   # optimization library

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional
from functools import partial
import optax
import matplotlib.pyplot as plt 
import time
from IPython import display

class EGNN(hk.Module):

    def __init__(self, 
                 depth :int,
                 F:int, 
                 remat: bool = False, 
                 init_stddev:float = 0.1,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        self.depth = depth
        self.F = F
        self.remat = remat
        self.init_stddev = init_stddev
  

    def phie(self, x, d):
        n, dim = x.shape 
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        rij = jnp.sum(jnp.square(rij), axis=-1).reshape(n, n, 1)
        mlp = hk.nets.MLP([self.F, 1], 
                          w_init=hk.initializers.TruncatedNormal(0.01), 
                          b_init=hk.initializers.TruncatedNormal(0.1),
                          activation=jax.nn.silu, 
                          name=f"edge_mlp_{d}") 
        return mlp(rij) 

    def phi0(self, x, d):
        n, dim = x.shape 
        ri2 = jnp.sum(jnp.square(x), axis=-1).reshape(n, 1)
        mlp = hk.nets.MLP([self.F, 1], 
                          w_init=hk.initializers.TruncatedNormal(0.01), 
                          b_init=hk.initializers.TruncatedNormal(0.1),
                          activation=jax.nn.silu,
                          name=f"point_mlp_{d}") 
        return mlp(ri2)

    def __call__(self, x):
        
        x = jnp.reshape(x, [int(x.size/2), 2])
        assert x.ndim == 2
        n, dim = x.shape

        def block(x, d): 
            mij = self.phie(x, d)
            weight = mij.reshape(n, n)/(n-1)
            
            xij = jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim))
            xij = xij.reshape(n, n, dim) 
            
            sum_mij_xij = jnp.einsum('ijd,ij->id', xij, weight)
            
            #sum_xi_xi = 0.01 * jnp.sum(jnp.square(x), axis=-1).reshape(n, 1) * x
            sum_xi_xi = self.phi0(x, d) * x /(n-1)
            
            x = x + sum_mij_xij + sum_xi_xi
            return x
        
        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for d in range(self.depth):
            x = block(x, d)
            
        x = jnp.reshape(x, [x.size])
        return x

#%%
"""Hamiltonian"""
def energy_fn(x, n, dim):
    i, j = np.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    return jnp.sum(x**2) + v_ee

def make_network(key, n, dim, hidden_sizes):
    depth, F = hidden_sizes
    @hk.without_apply_rng
    @hk.transform
    def network(z):
        net = EGNN(depth, F)
        return net(z)
    z = jax.random.normal(key, (n*dim, ))
    params = network.init(key, z)
    return params, network.apply


#%%
### x, logp
def make_flow(network):
    def flow(params, z):
        x = network(params, z)
        jac = jax.jacfwd(network,argnums=1)(params, z)
        _, logabsdet = jnp.linalg.slogdet(jac)
        logp = jnp.sum(jax.scipy.stats.norm.logpdf(z)) - logabsdet
        return x, logp
    return flow


def make_loss(batch_flow, n, dim, beta):

    batch_energy = jax.vmap(energy_fn, (0, None, None), 0)

    def loss(params, z):

        x, logp = batch_flow(params, z)

        energy = batch_energy(x, n, dim)
        f = logp/beta + energy
        return jnp.mean(f), (jnp.std(f)/jnp.sqrt(x.shape[0]), x)
    return loss

#%%
"""Initial"""

batchsize = 1024 #8192
n = 6 #20
dim = 2 
beta = 10.0

hidden_sizes = [4, 64]

### key42
key = jax.random.PRNGKey(42)

params, network = make_network(key, n, dim, hidden_sizes)
flow = make_flow(network)
batch_flow = jax.vmap(flow, (None, 0), (0, 0))
loss = make_loss(batch_flow, n, dim, beta)
value_and_grad = jax.value_and_grad(loss, has_aux=True)


from jax.flatten_util import ravel_pytree
print(ravel_pytree(params)[0].size)


"""Optimization"""

#optimizer = optax.adam(1e-3)
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)


#%%
"""Training"""
@jax.jit
def step(key, params, opt_state):
    z = jax.random.normal(key, (batchsize, n*dim)) 
    value, grad = value_and_grad(params, z)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return value, params, opt_state

t0 = time.time()
dy = []
loss_history = []

#fig = plt.figure(figsize=(5, 5))
for i in range(2000):
    t0 = time.time()
    
    key, subkey = jax.random.split(key)
    value,  params, opt_state = step(subkey, params, opt_state)
    f_mean, (f_err, x) = value
    loss_history.append([f_mean, f_err])
    
    t1 = time.time()
    print(i, f_mean, 't=',round(t1-t0, 2),'s')

####
    x = jnp.reshape(x, (batchsize*n, dim)) 
    display.clear_output(wait=True)
    

    if i%10 == 0:
        fig = plt.figure(figsize=(12, 5))
        ## figure 1
        plt.subplot(1, 2, 1)
        H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], 
                                           bins=100, 
                                           range=((-4, 4), (-4, 4)),
                                           density=True)
        plt.imshow(H, interpolation="nearest", 
                   extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                   cmap="inferno")
    
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        
        ## figure 2
        plt.subplot(1, 2, 2)
        
        y = np.reshape(np.array(loss_history), (-1, 2))
        plt.plot(np.arange(i+1), y[:, 0], marker='.')
        plt.xlabel('epochs')
        plt.ylabel('variational free energy')
        plt.pause(0.001)
        
print(loss_history[-1])

#%%
x = x.reshape(batchsize, n, dim)
print(jax.vmap(energy_fn, (0, None, None), 0)(x, n, dim).mean())
print(jax.vmap(energy_fn, (0, None, None), 0)(x, n, dim))

#%%
x = x.reshape(batchsize, n, dim)
plt.figure(figsize=(4, 4))
plt.scatter(x[:, :, 0], x[:, :, 1], alpha=0.1, s=1, color='r')
plt.xlim([-5, 5])
plt.ylim([-5, 5])

#%%
x = x.reshape(batchsize, n, dim)
plt.figure(figsize=(4, 4))
b = 3
plt.scatter(x[b, :, 0], x[b, :, 1], alpha=0.5)
plt.xlim([-5, 5])
plt.ylim([-5, 5])

