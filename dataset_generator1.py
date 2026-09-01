# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:25:01 2026

@author: codett
"""

# Cleaned version of dataset_generator_pandapower_v2.py

import os
import argparse
import numpy as np
import pandapower as pp
from tqdm import tqdm

# Configuration
NUMBER_OF_SAMPLES = 2000
NUMBER_OF_PROCESSES = 10

# Helper Functions
def remove_c_nf(net):
    """Set line capacitance to zero, as in PowerFlowNet."""
    net.line["c_nf_per_km"] = 0.0
    
def get_line_z_pu(net):
    """Return line resistance and reactance in per-unit"""
    r = net.line["r_ohm_per_km"].values * net.line["length_km"].values
    x = net.line["x_ohm_per_km"].values * net.line["length_km"].values
    from_bus = net.line["from_bus"].values
    vn_kv = net.bus["vn_kv"].values[from_bus]
    z_base = vn_kv ** 2 / net.sn_mva
    r_pu = r / z_base
    x_pu = x / z_base
    return r_pu, x_pu

def get_trafo_z_pu(net):
    """Return transformer reactance and resistance in per-unit."""
    for trafo_id in net.trafo.index:
        net.trafo.loc[trafo_id, "i0_percent"] = 0.0
        net.trafo.loc[trafo_id, "pfe_kw"] = 0.0
        
    z_pu = (
        net.trafo["vk_percent"].values / 100
        * 1000.0 / net.sn_mva)
    
    r_pu = (
        net.trafo["vkr_percent"].values / 100.0 
        * 1000.0 / net.sn_mva)
    
    x_pu = np.sqrt(z_pu ** 2 - r_pu ** 2)
    return x_pu, r_pu

# Data Generation
def generate_data(num_samples, case):
    edge_features_list = []
    node_features_x_list = []
    node_features_y_list = []
    
    for sample_idx in tqdm(range(num_samples)):
        
        # Start from fresh default Pandapower case
        if case == '14': net = pp.networks.case14()
        elif case == '118': net = pp.networks.case118()
        elif case == '6470rte': net = pp.networks.case6470rte()
    
        # Set line capacitance to zero
        remove_c_nf(net)
        n = len(net.bus)
        
        # Give buses integer names, matching PowerFlowNet
        net.bus['name'] = net.bus.index
        
        # Save default operating-point quantities
        r = net.line['r_ohm_per_km'].values.copy()
        x = net.line['x_ohm_per_km'].values.copy()
        le = net.line['length_km'].values.copy()
        
        Pg = net.gen['p_mw'].values.copy()
        Pd = net.load['p_mw'].values.copy()
        Qd = net.load['q_mvar'].values.copy()
        
        # PowerFlowNet perturbations
        
        # Line parameters: ±20%
        r = np.random.uniform(0.8 * r, 1.2 * r)
        x = np.random.uniform(0.8 * x, 1.2 * x)
        le = np.random.uniform(0.8 * le, 1.2 * le)
        
        # Generator voltage: 1.00–1.05 p.u.
        Vg = np.random.uniform(1.00, 1.05, size=len(net.gen))
        
        # Generator active power: Gaussian, std = 10% of |default Pg|
        Pg = np.random.normal(Pg, 0.1 * np.abs(Pg))
        
        # Local active power: Gaussian, std = 10% of |default Pd|
        Pd = np.random.normal(Pd, 0.1 * np.abs(Pd))
        
        # Local reactive power: Gaussian, std = 10% of |default Qd|
        Qd = np.random.normal(Qd, 0.1 * np.abs(Qd))
        
        # Apply Perturbations
        net.line["r_ohm_per_km"] = r
        net.line["x_ohm_per_km"] = x
        net.line["length_km"] = le

        net.gen["vm_pu"] = Vg
        net.gen["p_mw"] = Pg

        net.load["p_mw"] = Pd
        net.load["q_mvar"] = Qd

        # Solve Power Flow
        try:
            pp.runpp(
                net,
                algorithm='nr',
                init='results',
                numba=False)
        except pp.LoadflowNotConverged:
            # Do not count failed samples
            continue
        
        # Edge features
        
        edge_features = np.zeros((len(net.line), 7))
        edge_features[:, 0] = (net.line['from_bus'].values + 1)
        edge_features[:, 1] = (net.line['to_bus'].values + 1)
        edge_features[:, 2], edge_features[:, 3] = (get_line_z_pu(net))
        
        # Columns 4-6 remain zero: [b, tau, angle]
        
        # Tranformer features
        trafo_features = np.zeros((len(net.trafo),7))
        
        if len(net.trafo) > 0:
            trafo_features[:, 0] = (net.trafo['hv_bus'].values + 1)
            trafo_features[:, 1] = (net.trafo['lv_bus'].values + 1)
            trafo_features[:, 2], trafo_features[:, 3] = (get_trafo_z_pu(net))
            
        edge_features = np.concatenate(
            [edge_features, trafo_features],
            axis=0)
        
        # Node input features X
        # [index, type, Vm, Va, Pd, Qd, Gs, Bs, Pg]
        
        node_features_x = np.zeros((n,9))
        node_features_x[:, 0] = (net.bus['name'].values + 1)
        node_features_x[:, 3] = 0.0 # Va
        
        vm = np.ones(n)
        types = np.ones(n) * 2
        
        # Generator buses
        for j in range(len(net.gen)):
            bus = int(net.gen['bus'].values[j])
            vm[bus] = net.gen['vm_pu'].values[j]
            types[bus] = 1
            node_features_x[bus, 8] = (
                net.gen['p_mw'].values[j] / net.sn_mva)
            
        node_features_x[:, 2] = vm
        node_features_x[:, 1] = types
        
        # Load buses
        for j in range(len(net.load)):
            bus = int(net.load['bus'].values[j])
            node_features_x[bus, 4] = (Pd[j] / net.sn_mva)
            node_features_x[bus, 5] = (Qd[j] / net.sn_mva)
            
        # Node targte features Y
        # [index, type, Vm, Va, P, Q, Gs, Bs]
        
        node_features_y = np.zeros((n, 8))
        node_features_y[:, 0] = (net.bus['name'].values + 1)
        node_features_y[:, 1] = types
        node_features_y[:, 2] = (net.res_bus['vm_pu'].values)
        node_features_y[:, 3] = (net.res_bus['va_degree'].values)
        node_features_y[:, 4] = (net.res_bus['p_mw'].values / net.sn_mva)
        node_features_y[:, 5] = (net.res_bus['q_mvar'].values / net.sn_mva)
        
        # Store sample
        edge_features_list.append(edge_features)
        node_features_x_list.append(node_features_x)
        node_features_y_list.append(node_features_y)
        
    return (edge_features_list, node_features_x_list, node_features_y_list)
    
# Main
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case',
        type=str,
        default='14',
        choices=['14', '118', '6470rte'])
    parser.add_argument(
        '--samples',
        type=int,
        default=NUMBER_OF_SAMPLES)
    
    args = parser.parse_args()
    print(f"Generating {args.samples} samples")
    print(f"Case: {args.case}")
    
    # Generate Data
    (edge_features_list, node_features_x_list, node_features_y_list) = generate_data(args.samples, args.case)
    
    # Conver to arrays
    edge_features = np.array(edge_features_list)
    node_features_x = np.array(node_features_x_list)
    node_features_y = np.array(node_features_y_list)
    
    # Save
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    case_name = f'case{args.case}'
    
    np.save(os.path.join(output_dir, f'{case_name}_edge_features.npy'), edge_features)
    np.save(os.path.join(output_dir, f'{case_name}_node_features_x.npy'), node_features_x)
    np.save(os.path.join(output_dir, f'{case_name}_node_features_y.npy'), node_features_y)