#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/riai2018/ELINA/python_interface/')

import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time
from datetime import datetime
from pprint import pprint
import copy
import warnings

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')


# In[2]:


class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.rank = []
        self.use_LP = []


# In[3]:


def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v
    
def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            res.rank.append(np.zeros((W.shape[0],1)))
            res.use_LP.append(np.full((W.shape[0],1), False))
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res

def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = copy.deepcopy(data[:,0])
    high = copy.deepcopy(data[:,1])
    return low,high


# In[4]:


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon
     
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


# In[5]:


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0


# In[6]:


def analyze(nn, LB_N0, UB_N0, label):   
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)

            dimadd = elina_dimchange_alloc(0,num_out_pixels)    
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels
            # handle affine layer
            for i in range(num_out_pixels):
                tdim= ElinaDim(var)
                linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                var+=1
            dimrem = elina_dimchange_alloc(0,num_in_pixels)
            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)
            # handle ReLU layer 
            if(nn.layertypes[layerno]=='ReLU'):
                element = relu_box_layerwise(man,True,element,0, num_out_pixels)
            nn.ffn_counter+=1 

        else:
            print(' net type not supported')
   
    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)

           
    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True
            for j in range(output_size):
                if(j!=i):
                    sup = bounds[j].contents.sup.contents.val.dbl
                    if(inf<=sup):
                        flag = False
                        break
            if(flag):
                predicted_label = i
                break    
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if(j!=label):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, verified_flag


# ## Function to read the neural network file and get perturned image bounds

# In[7]:


def load_nn(netname, specname, epsilon):
    '''
    Retrieve the perturbed label and load the neural network
    
    INPUT:
        - netname: file name for the neural network
        - specname: file name for the image to work on 
        - epsilon: perturbation for the image
    OUTPUT:
        - LB_N0: lower bound of the perturbed image
        - UB_N0: upper bound of the perturbed image
        - nn: neural network
        - label: predicted label of the input
        - flag_wrong_label: flag to verify if the predictionis correct or not 
    '''
    flag_wrong_label = False
    
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)
    
    label, _ = analyze(nn,LB_N0,UB_N0,0) # Get label of unperturbed image, i.e. eps=0
    
    print("True label: " + str(label))
    if(label == int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
        flag_wrong_label =  True
        
    return LB_N0, UB_N0, nn, label, flag_wrong_label


# ## Define operations on abstract domain using linear approximations

# __Paper:__ "Provable defenses against adversarial examples via the convex outer adversarial polytope"; J. Zico Kolter, Eric Wong [[PDF]](https://machine-learning-and-security.github.io/papers/mlsec17_paper_42.pdf)
# 
# We follow the following notation: 
# ```
# Consider a k-layer feedforward ReLU-based NN, f_θ : R^|x| → R^|y|. Given equations: 
#         z_hat_{i} = W_i * z_i + b_i, where i = {1, . . . , k}
#         z_i = max(z_hat_{i-1} , 0),      where i = {2, . . . , k}
# and, 
#         z_1 = x
#         f_θ(x) ≡ z_hat_{k}
# ```

# In[8]:


def get_model(single_thread=False):
    """
    Get Gurobi model
    """
    m = Model("LP")
    m.setParam("outputflag", False)

    # disable parallel Gurobi solver
    m.setParam("Method", 1)  # dual simplex
    if single_thread:
        m.setParam("Threads", 1) # only 1 thread
    return m


# In[9]:


def add_all_vars(m, numlayer, LB_N0, UB_N0, UB_hidden_box_list, verbose=False):
    """
    Add and create all variables of neural network to gurobi model.
    INPUT:
        - m: Gurobi model
        - numlayer: Number of Layers
        - LB_N0: Lower Bound of perturbed image input
        - UB_N0: Upper Bound of perturbed image input
        - UB_hidden_box_list: List of upper Bounds from box approximation (needed to set upper bound of ReLU outputs)
    OUTPUT:
        - m: Gurobi model with newly added variables
        - z: List of Gurobi variables corresponding to pre-ReLU Layer (hidden)
        - z_hat: List of Gurobi variables corresponding to post-ReLU Layer

    """
    
    # for output of each ReLU
    z = []
    # for output of each hidden layer
    z_hat = []
    
    # Create variables of input image
    num_in_pixels = len(LB_N0)
    img_vars = m.addVars(num_in_pixels, lb=LB_N0, ub=UB_N0,                  vtype=GRB.CONTINUOUS, name="input_layer")
    z.append(img_vars)
    
    # Create variables for all layers and append to the list 
    for i in range(numlayer):
        # for layers before the final layer, z_hat and z exists
        if i < (numlayer - 1):

            UB_relu = UB_hidden_box_list[i].squeeze().copy()
            for j in range(len(UB_hidden_box_list[i])):
                bound = UB_hidden_box_list[i][j]
                UB_relu[j] = max(0, bound)
            UB_relu.squeeze() 

            # middle layer, has both z and z hat
            z_hat_hidden = m.addVars(len(UB_hidden_box_list[i]), lb=-np.inf, ub=np.inf,                                      vtype=GRB.CONTINUOUS, name="hidden_layer_" + str(i))
            z_relu = m.addVars(len(UB_hidden_box_list[i]), lb=0.0, ub = UB_relu,                               vtype=GRB.CONTINUOUS, name="relu_layer_" + str(i))
            # append to the list
            z_hat.append(z_hat_hidden)
            z.append(z_relu)
        # for last layer, only z_hat exists
        else: 
            z_hat_hidden = m.addVars(len(UB_hidden_box_list[i]), lb=-np.inf, ub=np.inf,                                      vtype=GRB.CONTINUOUS, name="output_layer") 
            # append to the list
            z_hat.append(z_hat_hidden)

    m.update()
    
    if verbose:
        # Sanity check!
        # Size of z should be number of relu activation layers + 1 (for input)
        print("Number of relu layers: {0}".format(len(z)))
        # Size of z_hat should be number of hidden layers
        print("Number of hidden layers: {0}".format(len(z_hat)))
        print("Size of last hidden layer: {0}".format(len(z_hat[-1])))
    
    return m, z, z_hat


# In[10]:


def add_hidden_constraint(model, layerno, z, z_hat, weights, biases):
    """
    This function computes “which side” of the ReLU the pre-ReLU activations lies on.
    INPUT:
        - model: gurobi model
        - layerno: layer number from which z_hat belong
        - z: gurobi variables for hidden layer input
        - z_hat: gurobi variables for hidden layer output
        - weights: weights for the hidden layer
        - bias: bias in the hidden layer
    OUTPUT:
        - model: gurobi model with new hidden constrains
   """
    # Sanity check!
    assert len(z) == weights.shape[1]
    assert len(z_hat) == weights.shape[0]
    
    # add constraint to model
    for i_out in range(len(z_hat)):
        constr = LinExpr() + np.asscalar(biases[i_out])
        for s in range(len(z)):
            constr += z[s] * np.asscalar(weights[i_out, s])

        model.addConstr(z_hat[i_out] == constr,                     name="hidden_constr_" + str(layerno) + "_" + str(i_out))
    
    model.update()
    return model


# In[11]:


def add_relu_activation_constraint(model, layerno, z_hat, z, LB, UB):
    """
    This function computes “which side” of the ReLU the pre-ReLU activations lies on.
    INPUT:
        - model: gurobi model
        - layerno: layer number from which z_hat belong
        - z_hat: gurobi variables for pre-relu input
        - z: gurobi variables for relu output
        - LB: lower bound of inputs to a relu layer
        - UB: upper bound of inputs to a relu layer
    OUTPUT:
        - model: gurobi model with new ReLU constrains
   """
    # Sanity check!
    assert len(z) == len(UB)
    
    # iterate over each pre-relu neuron activation
    for j in range(len(UB)):
        u = np.asscalar(UB[j])
        l = np.asscalar(LB[j])

        if u <= 0:
            model.addConstr(z[j] == 0,                         name="relu_constr_deac_" + str(layerno) + "_" + str(j))
        elif l > 0:
            model.addConstr(z[j] == z_hat[j],                         name="relu_constr_deac_" + str(layerno) + "_" + str(j))
        else:
            alpha = u/(u - l)
            model.addConstr(z[j] >= 0 ,                          name="relu_const_ambi_pos_" + str(layerno) + "_" + str(j))
            model.addConstr(z[j] >= z_hat[j],                          name="relu_const_ambi_hid_" + str(layerno) + "_" + str(j))
            model.addConstr(z[j] <= alpha * (z_hat[j] - l),                          name="relu_const_ambi_lin_" + str(layerno) + "_" + str(j))
    model.update()
    return model


# In[12]:


def call_linear_solver(model, z_hat):
    """
    This function computes lower and upper bound for given objective function and model
    INPUT:
        - model: gurobi model
        - z_hat: gurobi variable to optimize for
    OUTPUT:
        - LB: lower bound of variable
        - UB: upper bound of variable
   """
    # Find Lower Bound
    model.setObjective(z_hat, GRB.MINIMIZE)
    model.update()
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        LB = model.objVal
    else:
        raise(RuntimeError('[Min] Error. Not Able to retrieve bound. Gurobi Model. Not Optimal.'))
    
    # reset model 
    model.reset()

    # Find Upper Bound
    model.setObjective(z_hat, GRB.MAXIMIZE)
    model.update()
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        UB = model.objVal
    else:
        raise(RuntimeError('[Max] Error. Not Able to retrieve bound. Gurobi Model. Not Optimal.'))
    
    # reset model 
    model.reset()
    
    return LB, UB


# ## Define operations on abstract domain using Box approximations

# In[13]:


def get_relu_bounds_using_box(man, input_LB, input_UB, num_in_pixels):
    '''
    This function calculates the bounds of a ReLU operation. 
    INPUT:
        - man: pointer to elina manager
        - input_LB: lower bound of the inputs to the ReLU
        - input_UB: upper bound of the inputs to the ReLU
        - num_in_pixels: number of inputs to ReLU
    
    OUTPUT:
        - output_LB: lower bound of the outputs from ReLU layer
        - output_UB: upper bound of the outputs from ReLU layer
        - num_out_pixels: number of outputs of ReLI layer
    '''
    itv = elina_interval_array_alloc(num_in_pixels)

    ## Populate the interval
    for i in range(num_in_pixels):
        elina_interval_set_double(itv[i], input_LB[i], input_UB[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_in_pixels, itv)
    elina_interval_array_free(itv, num_in_pixels)
    
    # ------------------------------------------------------------------
    # Handle ReLU Layer
    # ------------------------------------------------------------------
    num_out_pixels = num_in_pixels
    
    element = relu_box_layerwise(man, True, element,0, num_in_pixels)
    
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)
    
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)
    
    output_LB = np.zeros((num_out_pixels, 1), float)
    output_UB = np.zeros((num_out_pixels, 1), float)
    for j in range(num_out_pixels):
        output_LB[j] = bounds[j].contents.inf.contents.val.dbl
        output_UB[j] = bounds[j].contents.sup.contents.val.dbl
    
    # free out the memory allocations
    elina_interval_array_free(bounds, num_out_pixels)
    elina_abstract0_free(man, element)
    
    return output_LB, output_UB, num_out_pixels


# In[14]:


def get_hidden_bounds_using_box(man, weights, biases, input_LB, input_UB, num_in_pixels, verbose=False):
    '''
    This function calculates the bounds of a ReLU operation followed by a hidden layer. 
    INPUT:
        - man: pointer to elina manager
        - weights: weights of the hidden layer
        - biases: biases of the hidden layer
        - input_LB: lower bound of the inputs to the hidden layer
        - input_UB: upper bound of the inputs to the hidden layer
        - num_in_pixels: number of inputs to the input layer
    
    OUTPUT:
        - output_LB: lower bound of the outputs from hidden layer
        - output_UB: upper bound of the outputs from hidden layer
        - num_out_pixels: number of outputs of hidden layer
    '''
    itv = elina_interval_array_alloc(num_in_pixels)

    ## Populate the interval
    for i in range(num_in_pixels):
        elina_interval_set_double(itv[i], input_LB[i], input_UB[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_in_pixels, itv)
    elina_interval_array_free(itv, num_in_pixels)
    
    # ------------------------------------------------------------------
    # Handle Affine Layer
    # ------------------------------------------------------------------

    # calculate number of outputs
    num_out_pixels = len(weights)
    
    if verbose:
        print("[Network] Input pixels: " + str(num_in_pixels))
        print("[Network] Shape of weights: " + str(np.shape(weights)))
        print("[Network] Shape of biases: " + str(np.shape(biases)))
        print("[Network] Out pixels: " + str(num_out_pixels))

    # Create number of neurons in the layer and populate it
    # with the number of inputs to each neuron in the layer
    dimadd = elina_dimchange_alloc(0, num_out_pixels)    
    for i in range(num_out_pixels):
        dimadd.contents.dim[i] = num_in_pixels

    # Add dimensions to an ElinaAbstract0 pointer i.e. element
    elina_abstract0_add_dimensions(man, True, element, dimadd, False)
    elina_dimchange_free(dimadd)

    # Create the linear expression associated each neuron
    var = num_in_pixels
    for i in range(num_out_pixels):
        tdim = ElinaDim(var)
        linexpr0 = generate_linexpr0(weights[i], biases[i], num_in_pixels)
        # Parallel assignment of several dimensions of an ElinaAbstract0 by using an ElinaLinexpr0Array
        element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
        var += 1

    # Pointer to which semantics we want to follow.
    dimrem = elina_dimchange_alloc(0, num_in_pixels)
    for i in range(num_in_pixels):
        dimrem.contents.dim[i] = i
        
    # Remove dimensions from an ElinaAbstract0
    elina_abstract0_remove_dimensions(man, True, element, dimrem)
    elina_dimchange_free(dimrem)
    
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)
    
    output_LB = np.zeros((num_out_pixels, 1), float)
    output_UB = np.zeros((num_out_pixels, 1), float)
    for j in range(num_out_pixels):
        output_LB[j] = bounds[j].contents.inf.contents.val.dbl
        output_UB[j] = bounds[j].contents.sup.contents.val.dbl    
    
    # free out the memory allocations
    elina_interval_array_free(bounds, num_out_pixels)
    elina_abstract0_free(man, element)
    
    return output_LB, output_UB, num_out_pixels


# ## Define function to verify the neural network

# In[15]:


def verify_network(LB_N0, UB_N0, LB_NN, UB_NN, label, num_input_pixels = 784, num_out_pixels = 10):
    '''
    This function verifies the network given the bounds of the input layer and the final layer of the network.
    INPUT:
        - LB_N0: lower bounds of the preturbed input image
        - UB_N0: unpper bounds of the preturbed input image
        - LB_NN: lower bounds of the final layer of neural network
        - UB_NN: upper bounds of the final layer of neural network
        - label: true label of the input image
        - num_input_pixels: number of pixels in the input image (for MNIST, default: 784)
        - num_out_pixels: number of neurons in the last layer of the network  (for MNIST, default: 10)
    
    OUTPUT:
        - predicted_label: label predicted by the neural network
        - verified_flag: boolean variable, true if the network is robust to perturbation
    '''
    
    # if epsilon is zero, try to classify else verify robustness 
    verified_flag = True
    predicted_label = 0
    if(LB_N0[0]==UB_N0[0]):
        for i in range(num_out_pixels):
            inf = LB_NN[i]
            flag = True
            for j in range(num_out_pixels):
                if(j!=i):
                    sup = UB_NN[j]
                    if(inf<=sup):
                        flag = False
                        break
            if(flag):
                predicted_label = i
                break    
    else:
        inf = LB_NN[label]
        for j in range(num_out_pixels):
            if(j!=label):
                sup = UB_NN[j]
                if(inf<=sup):
                    predicted_label = label
                    verified_flag = False
                    break
        
    return predicted_label, verified_flag


# In[16]:


def print_verify_output(verified_flag):
    if(verified_flag):
        print("verified")
    else:
        print("can not be verified")  


# ## Functions to perform different analysis

# In[17]:


# function to perform box analysis for all layers in the neural network nn
def perform_box_analysis(nn, LB_N0, UB_N0, verbose = False):
    # create a list to store the bounds found through box approximation
    LB_hidden_box_list = []
    UB_hidden_box_list = []

    # create manager for Elina
    man = elina_box_manager_alloc()

    # initialize variables for the network iteration
    numlayer = nn.numlayer 
    nn.ffn_counter = 0

    # for input image
    input_LB = LB_N0.copy()
    input_UB = UB_N0.copy()
    num_in_pixels = len(LB_N0)
    
    if verbose:
        print("Input Layer, size: " + str(len(LB_N0)))
        print('---------------')

    for layerno in range(numlayer):
        if verbose:
            print("Layer Number: " + str(layerno + 1))

        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            if verbose:
                print("Layer Type: %s" % nn.layertypes[layerno])

            # read the layer weights and biases
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)

            # ------------------------------------------------------------------
            # Handle Affine Layer
            # ------------------------------------------------------------------
            output_LB, output_UB, num_out_pixels = get_hidden_bounds_using_box(man, weights, biases, input_LB, input_UB, num_in_pixels, verbose)

            # Add bounds to the list
            LB_hidden_box_list.append(output_LB.copy())
            UB_hidden_box_list.append(output_UB.copy())
            # Prepare variables for next layer
            input_LB = output_LB.copy()
            input_UB = output_UB.copy()
            num_in_pixels = num_out_pixels
            nn.ffn_counter += 1 

            # ------------------------------------------------------------------
            # Handle ReLU Layer
            # ------------------------------------------------------------------
            if(nn.layertypes[layerno] == "ReLU"):
                output_LB, output_UB, num_out_pixels = get_relu_bounds_using_box(man, input_LB, input_UB, num_in_pixels)

            # Prepare variables for next layer
            input_LB = output_LB.copy()
            input_UB = output_UB.copy()

            if verbose:
                print("[OUTPUT] Bounds: ")
                output_LB, output_UB  = output_LB.squeeze(), output_UB.squeeze()
                pprint(np.stack((output_LB, output_UB), axis=1))
            
            if verbose:
                print('---------------')

        else:
            print(' net type not supported')
    if verbose:
        print("Output Layer, size: " + str(len(output_LB)))

    elina_manager_free(man)
    
    # for last layer of the netowork is ReLU
    LB_NN = LB_hidden_box_list[-1].copy()
    UB_NN = UB_hidden_box_list[-1].copy()

    if nn.layertypes[-1] == "ReLU" :
        num_out = len(LB_hidden_box_list[-1])
        for i in range(num_out):
            if LB_hidden_box_list[-1][i] < 0 :
                LB_NN[i] = 0 
            if UB_hidden_box_list[-1][i] < 0 :
                UB_NN[i] = 0 
            
    return LB_hidden_box_list, UB_hidden_box_list, LB_NN.squeeze(), UB_NN.squeeze()


# In[18]:


def perform_linear_layerwise(nn, numlayer, LB_N0, UB_N0, lp_list, 
                             LB_hidden_box_list, UB_hidden_box_list, verbose=False):
    """
    Get final bounds using linear programming layerwise. If lp_freq > 1 linear bounds are only calculated every 
    lp_freq'th layer.
    INPUT:
        - m: Gurobi model
        - z: List of Gurobi variables corresponding to pre-ReLU Layer (hidden)
        - z_hat: List of Gurobi variables corresponding to post-ReLU Laye
        - nn: Neural Network as defined in initial code (contains layertypes, weights, etc.)
        - numlayer: Number of Layers
        - LB_N0: Lower Bound of perturbed image input
        - UB_N0: Upper Bound of perturbed image input
        - lp_list: Layerno to start solvingbounds by LP 
        - prob: probability to select a neuron
        - LB_hidden_box_list: List of upper Bounds from box approximation
        - UB_hidden_box_list: List of upper Bounds from box approximation
    OUTPUT:
        - LB_NN: Lower bounds of neural network output
        - UB_NN: Upper bounds of neural network output
    """
    
    # Sanity check
    assert len(lp_list) != 0
    assert LB_hidden_box_list is not None 
    assert UB_hidden_box_list is not None


    # create manager for Elina
    man = elina_box_manager_alloc()
    
    # create gurobi model
    m = get_model()
    # create all gurobi variables for the network
    m, z, z_hat = add_all_vars(m, numlayer, LB_N0, UB_N0, UB_hidden_box_list, verbose)
    
    # initialize counter
    nn.ffn_counter = 0
    
    # Adding weights constraints for k layers
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            # read the layer weights and biases
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            np.ascontiguousarray(weights, dtype=np.float)
            np.ascontiguousarray(biases, dtype=np.float)

            # output shape of the layer
            n_in = weights.shape[1]
            n_out = weights.shape[0]

            # create variables to store bounds of hidden layer
            LB_hat = np.zeros(n_out, float)
            UB_hat = np.zeros(n_out, float)

            # add affine constraint
            add_hidden_constraint(m, layerno, z[layerno], z_hat[layerno], weights, biases)
            
            # for initial layers use original box bounds
            if layerno < lp_list[0] or layerno == 0:
                LB_hat, UB_hat = LB_hidden_box_list[layerno].copy() , UB_hidden_box_list[layerno].copy()
            # for the last layer
            elif layerno == numlayer - 1:
                for i_out in range(n_out):
                        LB_hat[i_out], UB_hat[i_out] = call_linear_solver(m, z_hat[layerno][i_out])
                break;
            else:            
                if (layerno in lp_list):
                    # find new bounds for the hidden layer using linear solver
                    for i_out in range(n_out):
                        LB_hat[i_out], UB_hat[i_out] = call_linear_solver(m, z_hat[layerno][i_out])                    
                else:
                    # find new bounds for the hidden layer using box solver
                    LB, UB, n_out = get_relu_bounds_using_box(man, LB_hat_prev, UB_hat_prev, n_in)
                    LB_hat, UB_hat, n_out = get_hidden_bounds_using_box(man, weights, biases, 
                                                                LB, UB, n_out, verbose)
            # add relu constraint
            if layerno < (numlayer - 1) and nn.layertypes[layerno] in ["ReLU"]:
                add_relu_activation_constraint(m, layerno, z_hat[layerno], z[layerno + 1], LB_hat, UB_hat)
            
            # preparation for next iteration    
            LB_hat_prev, UB_hat_prev = LB_hat.copy(), UB_hat.copy()
                
            m.update()

            # update counter for next iteration
            nn.ffn_counter += 1
        else:
            raise("Not a valid layer!")
    
    # Set bounds of last performed layer to output
    LB_NN = LB_hat
    UB_NN = UB_hat
    # If last Layer is RELU change last lower and upper bounds accordingly.
    if nn.layertypes[-1] == "ReLU" :
        num_out = len(UB_hat)

    for i in range(num_out):
        if LB_hat[i] < 0 :
            LB_NN[i] = 0 
        if UB_hat[i] < 0 :
            UB_NN[i] = 0 
           
    return LB_NN, UB_NN


# ## Function to analyse neural network and write output to a file

# In[31]:


def analyse_nn_and_log(net_path, img_path, epsilon, write = True, verbose=False, log_file=None, max_out_time = 320):
    """
    Analyse Neural Network and write the result to a log file. Code heuristic yourself in this function
    INPUT:
        - net_path: Neural Network file path
        - img_path: Image file path
        - epsilon: Perturbation
        - write: boolean flag to write into file
        - verbose: Verbose
        - log_file: File to write results
        - max_out_time: maximum time to which which run the heuristics
    OUTPUT:
        - verified_flag: Verification flag. True if network was verified against perturbation.
    """
    
    log_root_path = '/home/riai2018/log'
    log_base_name = 'log_'
    
    verified_counter, true_label_counter = 0, 0
    
    stamp = '_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.now())
    
    if not os.path.isdir(log_root_path):
        os.makedirs(log_root_path)
    
    if write:
        if log_file is None:
            log_file = os.path.join(log_root_path, log_base_name + net_code + '_epsilon_' + str(epsilon) + stamp +'.txt')
        
        with open(log_file, "+a") as f:
            f.write("Epsilon %s\n\n" % epsilon)        
    elif verbose:
        print("Epsilon: {0}".format(epsilon))
    
    t_start = time.time()
    
    # Load NN and perturbe image
    LB_N0, UB_N0, nn, label, flag_wrong_label = load_nn(net_path, img_path, epsilon)
    true_label_counter += int(not flag_wrong_label)

    if flag_wrong_label:
        return

    # Get number of layers in NN
    numlayer = nn.numlayer

    # Your heuristics come here:
    if numlayer <= 4 and epsilon < 0.05:
        lp_freq = np.int(numlayer)
    else:
        lp_freq = 1
    
    # hard-coded heuristic for the large network 
    if np.shape(nn.weights[0])[0] > 500 and epsilon > 0.045:
        print_verify_output(False)
        return
    
    t1 =time.time()

    # Get rough bounds using box analysis
    LB_hidden_box_list, UB_hidden_box_list, LB_NN, UB_NN = perform_box_analysis(nn, LB_N0, UB_N0, verbose = verbose)

    while ((time.time() - t1) <= max_out_time):

        lp_start = 1
        lp_list = [i for i in range(lp_start, numlayer, lp_freq)]
        
        print("List: " + str(lp_list))

        # Get Bounds
        LB_NN, UB_NN = perform_linear_layerwise(nn, numlayer, LB_N0, UB_N0, lp_list, 
                                     LB_hidden_box_list, UB_hidden_box_list, verbose=verbose)
        # Check if NN was verified
        _, verified_flag = verify_network(LB_N0, UB_N0, LB_NN, UB_NN, label, num_input_pixels = len(LB_N0), num_out_pixels = 10)

        # check verification flag. If true then we succeed! 
        if verified_flag or lp_freq == 1:
            break;
        else:
            lp_freq = np.int(np.ceil(lp_freq/2))

    t2 = time.time()

    print_verify_output(verified_flag)
    verified_counter += int(verified_flag)

    # Write to logfile
    if write or verbose:
        if not flag_wrong_label and verified_flag:
            line = "{:>10} img_{:>2} verified label {:>2} time {:>5.4f} s freq {:>1}\n".format(net_path, img_path, label, t2-t1, lp_freq)
        elif not flag_wrong_label and not verified_flag:
            line = "{:>10} img_{:>2} failed label {:>2} freq {:>1}\n".format(net_path, img_path, label, lp_freq)
        else:
            line = "{:>10} img_{:>2} not considered\n".format(net_path, img_path)

        if write:
            with open(log_file, "+a") as f:
                f.write(line)
        elif verbose:
            print(line)

        # Write final summary
        line = []
        line.append("\n--------------------- \n")
        line.append("Analysis precision:  {:>3} /{:>3}\n".format(verified_counter, true_label_counter))
        line.append("Total Time: {:>5.4f} \n".format(time.time() - t_start))
        if write:
            with open(log_file, "+a") as f:
                for l in line:
                    f.write(l)
        elif verbose:
            for l in line:
                print(l)

    return


# # The main function

# In[ ]:


if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    # c_label = int(argv[4])
    
    # flags for debuging
    logger = False
    verbose = False
    
    start = time.time()
    analyse_nn_and_log(netname, specname, epsilon, logger, verbose)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
