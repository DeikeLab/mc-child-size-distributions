# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:26:07 2022

@author: druth
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import pandas as pd
import pickle

integrate = scipy.integrate.trapz

delta_min = 0.001

def pick_given_powerlaw(exponent,x_min,x_max,n=1):

    a = (exponent+1) / (x_max**(exponent+1) - x_min**(exponent+1))
    F = np.random.uniform(0,1,size=n)
    val = ( (exponent+1) * F / a + x_min**(exponent+1))** (1./(exponent+1))
    return np.squeeze(val)

def pick_given_exponential(decay_rate,x_min,x_max,n=1):
    
    b = decay_rate # = 1/<m>
    a = -1*b * np.exp(x_min*b*-1)
    
    min_possible = 0
    max_possible = a/b * ( np.exp(b*x_max) - np.exp(b*x_min))
    F = np.random.uniform(min_possible,max_possible,size=n)
    val = np.log(F*b/a + np.exp(b*x_min)) / b
    return np.squeeze(val)

def pick_uniform(x_min,x_max,n=1):
    return np.squeeze(np.random.uniform(x_min,x_max,size=n))

class Distribution:
    '''
    A class to compute child size distributions
    '''
    
    def __init__(self,values,bin_edges,n_observations=1):
        self.m = len(values)/n_observations # the average number of bubbles formed
        y,x = np.histogram(values,bins=bin_edges,density=True)
        delta_centers = x[:-1] + np.diff(x)/2
        # make it integrate to m
        p = y*self.m

        self.bin_edges = bin_edges
        self.bin_centers = delta_centers
        self.bin_widths = np.diff(bin_edges)
        self.p = p
        
    @property
    def number_int(self):
        return np.sum(self.p*self.bin_widths)
    
    @property
    def volume_int(self):
        return np.sum(self.p*self.bin_centers*self.bin_widths)
    
class Breakup:
    '''
    Class for one stochastically-simulated break-up
    '''
    
    def __init__(self,Delta,delta_min,m_picker):
        
        self.Delta = Delta
        self.delta_min = delta_min
        self.m_picker = m_picker # function to pick the number of child bubbles
        
    def gen_child_sizes(self):
        
        Delta = self.Delta
        delta_min = self.delta_min
        
        # pick the size of the inertial lobes
        inertial_max = Delta
        inertial_1 = pick_uniform(delta_min,inertial_max-delta_min)
        inertial_2 = inertial_max - inertial_1 # inertial_1 + inertial_2 = inertial_max
        
        # pick the number of bubbles produced
        m = self.m_picker(self.Delta)
        
        # get the sizes of the m-2 capillary bubbles
        largest_cap = Delta#/2 # initialize the largest value a capillary bubble might be
        caps = []
        for i in range(m-2):
            # only add on a bubble if we can fit in two smallest bubbles
            if (largest_cap-delta_min)>delta_min:
                # pick size from powerlaw distribution between delta_min and largest_cap-delta_min
                split_1 = pick_given_powerlaw(-7./6,delta_min,largest_cap-delta_min)
                # make split_1 the smaller of the two
                split_1 = np.min([split_1,largest_cap-split_1])
                # store it and update the largest capillary bubble which will break up next
                caps.append(split_1)
                largest_cap = largest_cap-split_1
        
        # adjust the volume of the inertial volumes to account for the capillary ones
        V_cap_total = np.sum(caps)
        chi_cap = V_cap_total / Delta
        inertial_1 = inertial_1 * (1-chi_cap)
        inertial_2 = inertial_2 * (1-chi_cap)
        
        child_sizes = np.squeeze([inertial_1,inertial_2] + caps)
        self.child_sizes = child_sizes[child_sizes>0]
        self.m = len(self.child_sizes)
        self.chi_cap = chi_cap
        
    def get_distribution(self,delta_edges):
        self.dist = Distribution(self.child_sizes,delta_edges)
        


Delta_vec_default = np.geomspace(0.1**3,1000,160)
class DataGeneration:
    
    def __init__(self,m_picker_params,n_MC=5e4,Delta_vec=None,):
        
        self.m_picker_params = m_picker_params # dict with m_min,b1,b2
        
        if Delta_vec is None:
            Delta_vec = Delta_vec_default.copy()
            
        self.n_MC = int(n_MC)
        self.Delta_vec = Delta_vec
        self.interpolating_points = None
    
    def m_given_Delta(self,Delta):
        
        # pick a value of m given Delta and the parameters in m_picker_params
        max_num = Delta/delta_min
        d0dH = Delta**(1./3)
        m_avg = self.m_picker_params['m_min'] + 1./self.m_picker_params['b1'] * d0dH**self.m_picker_params['b2']
        m_prime_avg = m_avg - 2
        decay_rate = -1/m_prime_avg
        m_prime_by_m_prime_avg = pick_given_exponential(decay_rate,-0.5,max_num)
        m_prime = m_prime_by_m_prime_avg #* m_prime_avg
        m = np.int(np.round(m_prime+2))
        return m
    
    #def gen_interpolating_data(self,):
    def gen_breakups(self):
        
        n_MC = self.n_MC
        Delta_vec = self.Delta_vec        
        
        def get_breakups(Delta):
            breakups = []
            for _ in range(n_MC):    
                bu = Breakup(Delta,delta_min,self.m_given_Delta)
                bu.gen_child_sizes()
                breakups.append(bu)
            return breakups
        
        # get all the size dists for each break-up
        # this should be parallelized
        breakupss = [get_breakups(Delta) for Delta in Delta_vec]
        self.breakupss = breakupss
        
    def gen_sizedist_data(self,):
        
        Delta_vec = self.Delta_vec
        df = pd.DataFrame(index=Delta_vec)
        dists = []
        
        breakupss = self.breakupss
                
        # go through and process each break-up
        for di,(Delta,breakups) in enumerate(zip(Delta_vec,breakupss)):
            
            # define delta values for this Delta
            half_end = np.max([delta_min*2,Delta/5])
            delta_half = np.geomspace(delta_min,half_end,21)
            dx = np.diff(delta_half)[-1]
            delta_edges = np.concatenate([delta_half,np.arange(delta_half[-1]+dx,Delta,dx)]) # +dx
                
            all_child_sizes = np.concatenate([bu.child_sizes for bu in breakups])
            overall_dist = Distribution(all_child_sizes,delta_edges,n_observations=len(breakups))
            dists.append(overall_dist)
            
            # get chi_cap
            chi_cap_vals = np.zeros(len(breakups))*np.nan
            for bi,breakup in enumerate(breakups):
                assert Delta == breakup.Delta
                chi_inertial = np.sum(breakup.child_sizes[:2]) / Delta
                chi_cap = 1-chi_inertial
                chi_cap_vals[bi] = chi_cap
            chi_cap_mean = np.mean(chi_cap_vals)
            
            alpha = -7./6
            def distribution_fit_chicap(bin_centers,gamma,chi_cap,Delta,delta_min):
                a = (1 - chi_cap) * Delta * (gamma+2) / (Delta**(gamma+2)-delta_min**(gamma+2))
                ap2 = alpha+2
                b = (ap2/(Delta**ap2 - delta_min**ap2)) * (Delta - (a/(gamma+2))*(Delta**(gamma+2) - delta_min**(gamma+2)))
                return a*bin_centers**gamma + b*bin_centers**alpha
            
            # fit coefficients of the two powerlaws
            bin_centers = overall_dist.bin_centers[:-1]
            x = bin_centers
            y = np.log(overall_dist.p[:-1])
            min_delta_for_fit=delta_min
            y = y[bin_centers>min_delta_for_fit]
            x = x[bin_centers>min_delta_for_fit]
            x = x[~np.isinf(y)]
            y = y[~np.isinf(y)]
            logged_dist_fit = lambda bin_centers,gamma: np.log(distribution_fit_chicap(bin_centers,gamma,chi_cap_mean,Delta,delta_min))
            popt,_ = scipy.optimize.curve_fit(logged_dist_fit,x,y,bounds=((-2,),(0)),maxfev=1000,p0=(-0.5,)) # 
            
            df.loc[Delta,'inertial_exp'] = popt[0]
            df.loc[Delta,'chi_cap'] = chi_cap_mean
            
        df['m_avg'] = [d.m for d in dists]
        
        # store metadata in the dataframe
        df.attrs['n_MC'] = self.n_MC
        df.attrs['m_picker_params'] = self.m_picker_params
        
        self.interpolating_points = df
        self.dists = dists
        
        
    def save(self,folder,fname):
        
        # save the interpolating points as a dataframe
        self.interpolating_points.to_pickle(folder+fname+r'.pkl')
        
        # save the simulated breakups
        with open(folder+fname+r'_individual_breakups.pkl','wb') as f:
            pickle.dump(self.breakupss, f)