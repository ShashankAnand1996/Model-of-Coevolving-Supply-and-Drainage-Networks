from landlab.components import FlowAccumulator, LinearDiffuser
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab import RasterModelGrid
from scipy import optimize
from mpmath import *
import numpy as np
import pickle
import os

def fill(Z):
    Ni = Z.shape[0]
    Nj = Z.shape[1]
    border = np.zeros((Ni, Nj)).astype(np.int8)
    border[:, 0] = 1
    border[:, -1] = 1
    border[0, :] = 1
    border[-1, :] = 1
    W = np.where(border == 0, np.max(Z) + 0.01, Z)
    eps = 0.0000001
    smt_done = 1
    while smt_done == 1:
        smt_done = 0
        proc_ext = np.where((W > Z) & (border == 0), 1, 0).astype(np.int8)
        list_nb = neighbour_list(W, -1)
        for nb in range(0, 8):
            case_1 = np.where((proc_ext == 1) & (Z >= list_nb[nb] + eps), 1, 0).astype(np.int8)
            case_2 = np.where((proc_ext == 1) & (case_1 == 0) & (W > list_nb[nb] + eps), 1, 0).astype(np.int8)
            Wnew = np.where(case_1 == 1, Z, W)
            Wnew = np.where(case_2 == 1, list_nb[nb] + eps, Wnew)
            if np.sum(np.abs(W - Wnew)) > 0:
                smt_done = 1
            W = np.copy(Wnew)
            list_nb = neighbour_list(W, -1)
    return W

def neighbour_list(A, border_value):
    Ni = A.shape[0]
    Nj = A.shape[1]

    A_7 = np.append(border_value * np.ones((1, Nj)), A[0: Ni - 1, :].reshape(Ni - 1, Nj), axis=0)
    A_5 = np.append(border_value * np.ones((Ni, 1)), A[:, 0: Nj - 1].reshape(Ni, Nj - 1), axis=1)
    A_1 = np.append(A[:, 1: Nj].reshape(Ni, Nj - 1), border_value * np.ones((Ni, 1)), axis=1)
    A_3 = np.append(A[1: Ni, :].reshape(Ni - 1, Nj), border_value * np.ones((1, Nj)), axis=0)

    A_6 = np.append(border_value * np.ones((Ni, 1)), A_7[:, 0: Nj - 1].reshape(Ni, Nj - 1), axis=1)
    A_8 = np.append(A_7[:, 1: Nj].reshape(Ni, Nj - 1), border_value * np.ones((Ni, 1)), axis=1)
    A_4 = np.append(border_value * np.ones((Ni, 1)), A_3[:, 0: Nj - 1].reshape(Ni, Nj - 1), axis=1)
    A_2 = np.append(A_3[:, 1: Nj].reshape(Ni, Nj - 1), border_value * np.ones((Ni, 1)), axis=1)

    list_neighbour = []

    list_neighbour.append(A_1)
    list_neighbour.append(A_2)
    list_neighbour.append(A_3)
    list_neighbour.append(A_4)
    list_neighbour.append(A_5)
    list_neighbour.append(A_6)
    list_neighbour.append(A_7)
    list_neighbour.append(A_8)

    return list_neighbour


# Function for updating scalar field, h, using the sink and source terms utilizing D_infinity flow direction method
def DinfEroder_vein_brent(mg, mg_rev, dt, K_sp, K_sp_rev, m_sp, n_sp, m_sp_rev, n_sp_rev):
    '''
    Function arguments:
    mg/mg_rev:                                  Raster grids used for the simulation
    dt:                                                        Time-step to be taken
    K_sp/K_sp_rev:        Erosion coefficient times production rate/utilization rate
    m_sp/m_sp_rev: Exponent of output/input material density in the sink/source term
    n_sp/n_sp_rev:                   Exponent of local slope in the sink/source term
    '''
 
    # Initializing local arrays for the function block
    ele = np.copy(mg.at_node['topographic__elevation'])
    A_neg = np.copy(mg.at_node['drainage_area'])
    A_pos = np.copy(mg_rev.at_node['drainage_area'])
    nodes = np.arange(mg.number_of_nodes)
    neighbour = np.append(mg.diagonal_adjacent_nodes_at_node, mg.adjacent_nodes_at_node, axis = 1)
    
    ## Step 1. Making the of doners
    receiver = mg.at_node['flow__receiver_node']
    receiver_of_neighbour = receiver[neighbour]
    receiver_of_neighbour[:,:,0] = np.where(neighbour == -1 , -1, receiver_of_neighbour[:,:,0])
    receiver_of_neighbour[:,:,1] = np.where(neighbour == -1 , -1, receiver_of_neighbour[:,:,1])
    
    donor = np.zeros_like(neighbour)
    for nei in range(8):
        donor[:, nei] = np.where((receiver_of_neighbour[:, nei, 0] == nodes) |
                                 (receiver_of_neighbour[:, nei, 1] == nodes) , neighbour[:, nei], -1)

    ## Step 2. Processing nodes from bottom to top
    queue = list(np.arange(mg.number_of_nodes)[mg.at_node['flow__sink_flag'] == 1])
    processed_nodes = np.zeros_like(nodes, dtype=bool)
    processed_nodes[queue] = True
    
    while len(queue) > 0:
        node = queue.pop(0)
        for nei in donor[node, donor[node] >= 0]:
            if processed_nodes[nei] == False and np.all(processed_nodes[receiver[nei][receiver[nei] >= 0]]):
                processed_nodes[nei] = True
                h_0 , h_1 = None, None
                
                for res in (receiver[nei]):
                    if res != -1:
                        if res // Nc == nei // Nc or res % Nc == nei % Nc:
                            h_0 = ele[res]
                        else:
                            h_1 = ele[res]
                h_n = ele[nei]
                width = dx
                
                if h_0 is not None and h_1 is not None:
                    f = lambda h_next: h_next - h_n + K_sp * (A_neg[nei] / width)**m_sp *\
                                ((h_0 - h_1)**2 + (h_next - h_0)**2)**(n_sp / 2.) * dt / dx**n_sp
                    min_h = min(h_0, h_1)

                elif h_0 is not None:
                    f = lambda h_next: h_next - h_n + K_sp * (A_neg[nei] / width)**m_sp *\
                                ((h_next - h_0)**2)**(n_sp / 2.) * dt / dx**n_sp
                    min_h = h_0
                elif h_1 is not None:
                    f = lambda h_next: h_next - h_n + K_sp * (A_neg[nei] / width)**m_sp *\
                                (h_next - h_1)**n_sp * dt / (2**.5 * dx)**n_sp

                    min_h = h_1
                    
                ele[nei] = optimize.brenth(f, h_n, min_h)
                
                queue.append(nei)
                  
    ## Step 1. Making the of doners
    receiver_rev = mg_rev.at_node['flow__receiver_node']
    receiver_of_neighbour_rev = receiver_rev[neighbour]
    receiver_of_neighbour_rev[:,:,0] = np.where(neighbour == -1 , -1, receiver_of_neighbour_rev[:,:,0])
    receiver_of_neighbour_rev[:,:,1] = np.where(neighbour == -1 , -1, receiver_of_neighbour_rev[:,:,1])
    
    donor = np.zeros_like(neighbour)
    for nei in range(8):
        donor[:, nei] = np.where((receiver_of_neighbour_rev[:, nei, 0] == nodes) |
                                 (receiver_of_neighbour_rev[:, nei, 1] == nodes) , neighbour[:, nei], -1)


    ## Step 2. Processing nodes from bottom to top
    queue = list(np.arange(mg_rev.number_of_nodes)[mg_rev.at_node['flow__sink_flag'] == 1])
    processed_nodes = np.zeros_like(nodes, dtype=bool)
    processed_nodes[queue] = True
    
    while len(queue) > 0:
        node = queue.pop(0)
        for nei in donor[node, donor[node] >= 0]:
            if processed_nodes[nei] == False and np.all(processed_nodes[receiver_rev[nei][receiver_rev[nei] >= 0]]):
                processed_nodes[nei] = True
                h_0 , h_1 = None, None
                for res in (receiver_rev[nei]):
                    if res != -1:
                        if res // Nc == nei // Nc or res % Nc == nei % Nc:
                            h_0 = ele[res]
                        else:
                            h_1 = ele[res]
                h_n = ele[nei]
                width = dx
                
                if h_0 is not None and h_1 is not None:
                    f = lambda h_next: h_next - h_n - K_sp_rev * (A_pos[nei] / width)**m_sp_rev *\
                                ((h_1 - h_0)**2 + (h_0 - h_next)**2)**(n_sp_rev / 2.) * dt / dx**n_sp_rev
                    min_h = min(h_0, h_1)

                elif h_0 is not None:
                    f = lambda h_next: h_next - h_n - K_sp_rev * (A_pos[nei] / width)**m_sp_rev *\
                                ((h_0 - h_next)**2)**(n_sp_rev / 2.) * dt / dx**n_sp_rev
                    min_h = h_0
                elif h_1 is not None:
                    f = lambda h_next: h_next - h_n - K_sp_rev * (A_pos[nei] / width)**m_sp_rev *\
                                (h_1 - h_next)**n_sp_rev * dt / (2**.5 * dx)**n_sp_rev

                    min_h = h_1
                ele[nei] = optimize.brenth(f, h_n, min_h)
                queue.append(nei)         

    return ele

# Initializing the coefficients, domain-size and model parameters
# Variable names ending with '_rev' correspond to input material density
K_sp     =        0.00004
K_sp_rev =        0.00004
D        =        0.00100
m_sp     =            1.0
n_sp     =            1.0
m_sp_rev =            1.0
n_sp_rev =            1.0
dx       =            1.0
Nr  =  L =            100
Nc       =            100
H        =           10.0

# Calculating non-dimensional "Chi" values for both materials 
chi      = (K_sp * L**(2 + m_sp -n_sp)) / (D * H**(1-n_sp))
chi_rev  = (K_sp_rev * L**(2 + m_sp_rev - n_sp_rev)) / (D * H**(1-n_sp_rev))

# Initializing the scalar field
h = np.zeros((Nr, Nc))
for i in range(Nr):
    h[i,:] = H - (H*i/L)
initial_roughness = np.random.rand(h.size)/100.
h += initial_roughness.reshape(Nr, Nc)
h[ 0, :] = H
h[-1, :] = 0

# Initializing the grid
mg     = RasterModelGrid((Nr,Nc), dx)
mg_rev = RasterModelGrid((Nr,Nc), dx)

mg.at_node['topographic__elevation']            =           h
mg_rev.at_node['topographic__elevation']        = h.max() - h

# Imposing boundary conditions as discussed in the manuscript  
for edge in (mg.nodes_at_left_edge, mg.nodes_at_right_edge):
    mg.status_at_node[edge]     =      CLOSED_BOUNDARY
for edge in (mg.nodes_at_bottom_edge,mg.nodes_at_top_edge):
    mg.status_at_node[edge]     = FIXED_VALUE_BOUNDARY
    
for edge in (mg_rev.nodes_at_left_edge, mg_rev.nodes_at_right_edge):
    mg_rev.status_at_node[edge] =      CLOSED_BOUNDARY
for edge in (mg_rev.nodes_at_bottom_edge,mg_rev.nodes_at_top_edge):
    mg_rev.status_at_node[edge] = FIXED_VALUE_BOUNDARY


fc     =     FlowAccumulator(mg, flow_director='FlowDirectorDINF')
fc_rev = FlowAccumulator(mg_rev, flow_director='FlowDirectorDINF')

fc.run_one_step()
fc_rev.run_one_step()

fd = LinearDiffuser(mg,linear_diffusivity=D)

dt           =          1.0
t            =          0.0
i            =           -1
count        =            0
min_try      =          500
min_time     =       1000.0
count_max    =          500
diff_list    =           []
steady_state =        False

while steady_state is False:
    
    i += 1
    alpha = min((i + 1) * 0.01 , 1.)
    dt = alpha * min(dx / (K_sp * np.max((mg.at_node['drainage_area'] / dx) ** m_sp)) , 20. * dx ** 2 / (2 * D))
    dt = min(dt, 500.)

    ele_1 = np.copy(mg.at_node['topographic__elevation'])
    
    erode_done = False
    while erode_done is False:
        try:
            fc_rev = FlowAccumulator(mg_rev, flow_director='FlowDirectorDINF')
            fc_rev.run_one_step()    
            fc     =     FlowAccumulator(mg, flow_director='FlowDirectorDINF')
            fc.run_one_step()
            mg.at_node['topographic__elevation'] = DinfEroder_vein_brent(mg, mg_rev, dt, K_sp,K_sp_rev, m_sp, n_sp, m_sp_rev, n_sp_rev)
            fd.run_one_step(dt)
    
            for k in range(Nr):
                mg.at_node['topographic__elevation'][k*Nc]   = mg.at_node['topographic__elevation'][k*Nc+1]
                mg.at_node['topographic__elevation'][k*Nc-1] = mg.at_node['topographic__elevation'][k*Nc-2]
            erode_done = True

        except ValueError:
            print('Adjusting dt')
            mg.at_node['topographic__elevation'] = ele_1
            dt = dt / 2.
                
    ele_2 = np.copy(mg.at_node['topographic__elevation'])
    t    += dt

    ele_diff_max      =         np.abs(ele_1 - ele_2).max()
    ele_diff_mean     = np.abs(ele_1.mean() - ele_2.mean())
    diff_list.append([t, ele_diff_max, ele_diff_mean])
    
    mg_rev.at_node['topographic__elevation'] =  ele_2.max() - ele_2
    
    if ele_diff_mean < 0.00000001:
        count = count + 1
    else:
        count = 0
        
    if count == count_max:
        steady_state = True
    
    # Checking if the steady-state is reached    
    if  i >= min_try and t >= min_time and ele_diff_max < 0.000001 and ele_diff_mean < 0.00000001:
        steady_state = True

    # Printing the time and other important information to see the simulation's progress
    print(i, t, ele_diff_max, ele_diff_mean)

# Saving the steady-state solutions
array_name_1 = './' + str(int(round(chi))) + '_' + str(int(round(chi_rev))) + '_steady_mg_at_'     + str(int(t)) + '.p'
array_name_2 = './' + str(int(round(chi))) + '_' + str(int(round(chi_rev))) + '_steady_mg_rev_at_' + str(int(t)) + '.p'
pickle.dump(      mg, open( array_name_1, "wb"))
pickle.dump(  mg_rev, open( array_name_2, "wb"))
