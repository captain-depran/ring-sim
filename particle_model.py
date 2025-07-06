# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:48:17 2025

@author: Luke
"""

import moon_mesh_tools as mmt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def ring_spawn():
    rng = np.random.default_rng(simconfig["seed"])
    radii= rng.integers(low=int(74e6), high=143e6, size=simconfig["particles"])
    theta=np.linspace(0,2*np.pi,simconfig["particles"])
    particle_states[0,:,0],particle_states[0,:,1]=mmt.ellipse_draw(radii, 0, theta, 0)
    #scatter=np.random.normal(1,simconfig["scatter_mult"],len(theta))
    #particle_states[0,:,0]=particle_states[0,:,0]*scatter
    #particle_states[0,:,1]=particle_states[0,:,1]*scatter


    

def run(t):
    m0=5.683e26
    m=0.379e20
    mt=1345e20
    
    a=185.52e6
    at=1221.87e6
    sat=mmt.body("Saturn",60268000,m0,m0)
    mim=mmt.body("Mimas",208000,m,m0)
    tit=mmt.body("Titan",2575000,mt,m0)
    prom=mmt.body("prometheus",68000,0.0016e20,m0)
    
    mim.orbit(t,a,0.0202,(2*np.pi)/10,0)
    tit.orbit(t,at,0.0292,(2*np.pi)/3,0)
    prom.orbit(t,139.353e6,0.00204,(2*np.pi)/7,0)
    
    global moons
    #moons=[prom,tit,mim]
    moons=[]
    
    ring_spawn()
    global fs
        
    
    particle_speeds[0,:,1],particle_speeds[0,:,0]=sat.orbit_speed(particle_states[0,:,0], particle_states[0,:,1], 0, 0)
    particle_speeds[0,:,1]=particle_speeds[0,:,1]*(-1)
    
    moon_accel=lambda moon,x,y,i: moon.f_accel(x,y,moon.pos[0][i],moon.pos[1][i])
    
  
    
    
    for n in range(0,steps-1):
        accels = np.empty((len(moons),simconfig["particles"],2))
        fs=sat.f_accel(particle_states[n,:,0], particle_states[n,:,1], 0, 0).T
        for m in range(0,len(moons)):
            accels[m] = moon_accel(moons[m],particle_states[n,:,0], particle_states[n,:,1],0).T
        accels=np.sum(accels,axis=0)+fs
        
        particle_speeds[1,:,:]=particle_speeds[0,:,:]+(accels*simconfig["dt"])
        particle_states[n+1,:,0:2]=(particle_speeds[1,:,:]*simconfig["dt"])+particle_states[n,:,0:]
        particle_speeds[0,:,:]=particle_speeds[1,:,:]
    

def cleanup(states):
    #print("Cleanup starting...")
    nan_check=np.count_nonzero(~np.isnan(particle_states), axis=0)
    warning_list=[]
    for i in range(0,simconfig["particles"]):
        if nan_check[i,0]!=steps:
            warning_list.append(i)
    
    states=np.delete(states,warning_list,axis=1)
    #print("Cleanup done!")
    
    return states
    


global particle_states
global particle_speeds
global simconfig
global steps

simconfig = {
    "seed": 12345,
    "particles" : 30000,
    "PRad" : int(120e6),
    "timespan": 2419000,
    "dt" : 500,
    "scatter_mult" : 0.1,
    "blocks" : 96
    }



steps=int(simconfig["timespan"]//simconfig["dt"])
t=np.linspace(0,simconfig["timespan"],steps)



particle_states=np.empty((steps,simconfig["particles"],2))
particle_speeds=np.empty((2,simconfig["particles"],2))

"""
Speed and position indexs 0,1 are x,y
"""

for n in tqdm(range(0,simconfig["blocks"])):
    t=t+(simconfig["timespan"]*n)
    run(t)
    particle_states[0,:,:]=particle_states[-1,:,:]



particle_states=cleanup(particle_states)

#plt.scatter(particle_states[-1,:,0],particle_states[-1,:,1],marker=".")
plt.hexbin(particle_states[:,:,0],particle_states[:,:,1],gridsize=2000)
#plt.axis("scaled")
plt.show()