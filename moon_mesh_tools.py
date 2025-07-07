# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:46:44 2025

@author: Luke
"""

import numpy as np
import scipy.constants as const


global G
G=const.G



class body:
    def __init__(self,name,rad,m,parent_mass):
        self.name=name
        self.m=m
        self.m0=parent_mass
        self.rad=rad
        
    def orbit(self,t,a,e,rot,anom):
        self.p=period(a,self.m0)
        self.pos=ellipse_draw(a,e,(2*np.pi*(t/self.p))+anom,rot)
        
    def f_accel(self,xs,ys,mx,my):       
        
        xdif=np.subtract(xs,mx)
        ydif=np.subtract(ys,my)
        
        dif_stack=np.stack(np.broadcast_arrays(xdif,ydif),dtype="float64")

        diff=(np.linalg.norm(dif_stack,axis=0))
        
        unit=np.where(diff!=0,dif_stack/diff,0)

        
        grav_field=np.where(diff<self.rad,np.nan,grav(G,self.m,diff)*unit)
        
        return grav_field
    
    def surf_grav(self,rad):
        global G
        g=grav(G,self.m,rad)
        print("Surface g-accel= ",g)
        
    def orbit_speed(self,xs,ys,mx,my):
        xdif=np.subtract(xs,mx)
        ydif=np.subtract(ys,my)
        
        dif_stack=np.stack(np.broadcast_arrays(xdif,ydif),dtype="float64")

        diff=(np.linalg.norm(dif_stack,axis=0))
        
        unit=np.where(diff!=0,dif_stack/diff,0)
        
        speed=np.sqrt((G*self.m)/diff)*unit
        return speed

def grav(G,M,r):
    return(-1*(G*M)/(r**2))



def array_mask(rad_in,rad_out,xs,ys):
    return (((xs**2+ys**2)**0.5)<=rad_out) & (((xs**2+ys**2)**0.5)>=rad_in)



def period(a,m):
    global G
    top=4*(np.pi**2)*(a**3)
    bottom=G*m
    return np.sqrt(top/bottom)
    

def rotate(x,y,rot):
    xrot=(np.cos(rot)*x)-(np.sin(rot)*y)
    yrot=(np.sin(rot)*x)+(np.cos(rot)*y)
    return xrot,yrot


def ellipse(a,e,theta):
    b=a*np.sqrt(1-e**2)
    c=a*e
    x=(a*np.cos(theta))+c
    y=b*np.sin(theta)    
    return(x,y)
    
def ellipse_draw(a,e,theta,rot):
    ex,ey=ellipse(a,e,theta)
    ex,ey=rotate(ex,ey,rot)
    return ex,ey

def cleanup(states):
    """
    Clean out any particle that *ever* reports a np.nan value for its velocity and/or position as this indicates the particle collided with a moon, and was removed from the rings.
    States is a matrix of particle positions and/or speeds/accelerations, and should any particle value *ever* be nan, it should be removed from the entire simulation.
    """
    nan_check=np.count_nonzero(~np.isnan(states), axis=0)
    warning_list=[]
    for i in range(0,len(states[0,:])):
        if nan_check[i,0]!=len(states[:,0]):
            warning_list.append(i)
    
    states=np.delete(states,warning_list,axis=1)
    
    return states

def radial_space(x,y):
    return np.sqrt((x**2)+(y**2))


def compare_runs(file1,file2):
    """
    Compare the final block of two previously ran simulations, importing their .npy files of the particle states.
    The returned list is a list of the difference in orbital radius for each particle in the sim.
    """
    moons=np.load(file1)
    no_moons=np.load(file2)
    moons=radial_space(moons[-1,:,0],moons[-1,:,1])
    no_moons=radial_space(no_moons[-1,:,0],no_moons[-1,:,1])
    compare=np.abs(moons-no_moons)
    return compare