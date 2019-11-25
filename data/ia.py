#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:07:32 2019

@author: joao
"""
from pynput.keyboard import Key, Controller
from random import randint
import random

import numpy as np
import pyscreenshot as ImageGrab
import cv2
import time
from .components import info
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
model_built = False
load_saved = True
save_current_pool = 1
current_pool = []
fitness = []
total_models = 10
generation = 1

'''
Modelo crossover (Recombinação) 
é um operador genético usado para 
variar a programação de um cromossomo
'''

def model_crossover(model_idx1, model_idx2): # modelo 1  , modelo 2
    global current_pool
    pesos1 = current_pool[model_idx1].get_weights()
    pesos2 = current_pool[model_idx2].get_weights()
    novos_pesos1 = pesos1
    novos_pesos2 = pesos2
    novos_pesos1[0] = pesos2[0] # mudança no cromossomo
    novos_pesos2[0] = pesos1[0] # mudança no cromossomo
    return np.asarray([novos_pesos1, novos_pesos2]) # retorna cromossomos modificados 

'''
Para o modelo de mutação ela acontece randomicamente 
'''
def model_mutacao(pesos):
    for xi in range(len(pesos)):
        for yi in range(len(pesos[xi])):
            temp2 = random.uniform(0, 1)
            print("Valor aleatorio da mutacao",temp2," > 0.85")
            if  temp2 > 0.85:
                novo_modelo = random.uniform(-0.5,0.5)
                pesos[xi][yi] += novo_modelo
    return pesos

'''
Modelo para prever a ação futura do player (mario)
'''
def predict_action(neural_input, model_num):
    neural_input = np.expand_dims(neural_input,axis=3)
    neural_input = np.atleast_2d(neural_input) #(560, 750, 3)
    neural_input = np.expand_dims(neural_input,axis=0)
    
    output_prob = current_pool[model_num].predict([neural_input]) # saida da predicao
    x = np.argmax(output_prob) # valor maximo do eixo
    return generated_input[x]


