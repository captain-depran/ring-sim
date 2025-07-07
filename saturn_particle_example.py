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
    radii= rng.integers(low=simconfig["PRadIn"], high=simconfig["PRadOut"], size=simconfig["particles"])
    theta=rng.integers(low=0,high=np.pi*2,size=simconfig["particles"])
    particle_states[0,:,0],particle_states[0,:,1]=mmt.ellipse_draw(radii, 0, theta, 0)


    

def run(t):
    m0=5.683e26 #Saturn Mass
    
    """
    Here are some moon values for Mimas and Titan. I have moved the numbers directly into the object initalisation, but am leaving these variables here commented incase I mess up and leave them called somewhere
    m=0.379e20
    mt=1345e20
    
    a=185.52e6
    at=1221.87e6
    """

    
    sat=mmt.body("Saturn",60268000,m0,m0)
    mim=mmt.body("Mimas",208000,0.379e20,m0)
    tit=mmt.body("Titan",2575000,1345e20,m0)
    prom=mmt.body("prometheus",68000,0.0016e20,m0)
    
    mim.orbit(t,185.52e6,0.0202,(2*np.pi)/10,0)
    tit.orbit(t,1221.87e6,0.0292,(2*np.pi)/3,0)
    prom.orbit(t,139.353e6,0.00204,(2*np.pi)/7,0)
    
    global moons
    moons=[mim,tit,prom] #Creates a list of moons for the simulation to consider
    
    ring_spawn()
    global fs #force of gravity due to saturn
        
    #Intialise the orbital velocities of every particle, assuming it begins in a stable circular orbit
    particle_speeds[0,:,1],particle_speeds[0,:,0]=sat.orbit_speed(particle_states[0,:,0], particle_states[0,:,1], 0, 0)
    particle_speeds[0,:,1]=particle_speeds[0,:,1]*(-1)

    #Create lambda function to get acceleration everywhere due to given moon, for later use
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
    


    


global particle_states
global particle_speeds
global simconfig
global steps

simconfig = {
    "seed": 12345,
    "particles" : 5000,
    "PRadIn" : int(74e6),      #particle spawning inner radius, meters
    "PRadOut" : int(130e6),    #particle spawning outer radius, meters
    "timespan": 2419000,       #Length of time to simulate, seconds
    "dt" : 500,                #Size of time step, seconds
    "blocks" : 196
    }



steps=int(simconfig["timespan"]//simconfig["dt"])
t=np.linspace(0,simconfig["timespan"],steps)



particle_states=np.empty((steps,simconfig["particles"],2))
particle_speeds=np.empty((2,simconfig["particles"],2))

"""
Speed and position indexs 0,1 are x,y
"""

#runs the sim for a month, then clears memory and restarts with particle intial state set by last months state
for n in tqdm(range(0,simconfig["blocks"])):
    t=t+(simconfig["timespan"]*n)
    run(t)
    particle_states[0,:,:]=particle_states[-1,:,:]


#remove all np.nan value inducing particles
particle_states=mmt.cleanup(particle_states)

np.save("moons.npy",particle_states)
fig=plt.hexbin(particle_states[:,:,0],particle_states[:,:,1],gridsize=1000)
#plt.scatter(particle_states[-1,:,0],particle_states[-1,:,1],marker=".")
#plt.hexbin(particle_states[:,:,0],particle_states[:,:,1],gridsize=1000)
plt.axis("scaled")
plt.savefig("saturn_rings.png",dpi=1000)
plt.show()
