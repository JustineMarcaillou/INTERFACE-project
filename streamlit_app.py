#%% Imports modélisation
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import cv2
import csv
import streamlit as st

st.set_page_config(page_title="Modeling in epidemiology", layout="wide")

#%% Code modélisation avec restriction
def sir_model_with_masks_SEIR(y, t, params):
    """
    SEIR model differential equations
    y: list of [S, E, I, R] values at time t
    t: time
    params: dictionary of parameters
    """
    S, E, I, R = y
    N = S+E+I+R
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    q = params['q']
    z = params['z']
    dSdt = -beta * S * I/N
    dEdt = beta * S * I/N - sigma * E - q * I
    dIdt = sigma * E - gamma * I - z * I
    dRdt = gamma * I + z * I
    return [dSdt, dEdt, dIdt, dRdt]

def plot_sir_model_simulation_masks_SEIR(result, days, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def generate_sir_model_data_masks_SEIR(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 0, 1, 0]
    # Solve the differential equations
    result = odeint(sir_model_with_masks_SEIR, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    # Introduce seasonal variations using a sinusoidal function if requested
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]
    # Add random noise if requested
    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise
    # Clip negative values to zero
    result[result < 0] = 0
    # Create DataFrame
    df = pd.DataFrame(result, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered'])
    return (df, result)

def mse_loss_SEIR(params, t, data, y0):
    params_dict = {
        'beta': params[0],
        'sigma': params[1],
        'gamma': params[2],
        'q': params[3],
        'z': params[4]
    }
    result = odeint(sir_model_with_masks_SEIR, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Exposed'])**2 +
                   (result[:, 2] - data['Infected'])**2 +
                   (result[:, 3] - data['Recovered'])**2)
    return loss

def optimize_params_SEIR(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Exposed'][0], data['Infected'][0], data['Recovered'][0]]
    bounds = [
        (0, 0.1),   # beta
        (0, 0.5),   # sigma
        (0, 0.5),   # gamma
        (0, 0.5),   # q
        (0, 0.5)    # z
    ]
    result_min = minimize(mse_loss_SEIR, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict_SEIR(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict_SEIR(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'sigma': optimized_params[1],
        'gamma': optimized_params[2],
        'q': optimized_params[3],
        'z': optimized_params[4]
    }
    return params_dict


def sir_model_with_masks_SIR(y, t, params):
    """
    SIR model differential equations
    y: list of [S, I, R] values at time t
    t: time
    params: dictionary of parameters
    """
    S, I, R = y
    N=S+I+R
    beta = params['beta']
    gamma = params['gamma']
    z = params['z']
    dSdt = -beta * S * I/N
    dIdt = beta * S * I/N - gamma * I - z * I
    dRdt = gamma * I + z * I
    return [dSdt, dIdt, dRdt]

def plot_sir_model_simulation_masks_SIR(result, days, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Infected', 'Recovered']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def generate_sir_model_data_masks_SIR(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 1, 0]
    # Solve the differential equations
    result = odeint(sir_model_with_masks_SIR, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    # Introduce seasonal variations using a sinusoidal function if requested
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]
    # Add random noise if requested
    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise
    # Clip negative values to zero
    result[result < 0] = 0
    # Create DataFrame
    df = pd.DataFrame(result, columns=['Susceptible', 'Infected', 'Recovered'])
    return (df, result)

def mse_loss_SIR(params, t, data, y0):
    params_dict = {
        'beta': params[0],
        'gamma': params[1],
        'z': params[2]
    }
    result = odeint(sir_model_with_masks_SIR, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Infected'])**2 +
                   (result[:, 2] - data['Recovered'])**2)
    return loss

def optimize_params_SIR(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Infected'][0], data['Recovered'][0]]
    bounds = [
        (0, 0.1),   # beta
        (0, 0.5),   # gamma
        (0, 0.5)    # z
    ]
    result_min = minimize(mse_loss_SIR, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict_SIR(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict_SIR(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'gamma': optimized_params[1],
        'z': optimized_params[2]
    }
    return params_dict


def sir_model_with_masks_SIRD(y, t, params):
    """
    SIRD model differential equations
    y: list of [S, I, R, D] values at time t
    t: time
    params: dictionary of parameters
    """
    S, I, R, D = y
    N = S+I+R+D
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    z = params['z']
    dSdt = -beta * S * I/N
    dIdt = beta * S * I/N - gamma * I - delta * I - z * I
    dRdt = gamma * I + z * I
    dDdt = delta * I
    return [dSdt, dIdt, dRdt, dDdt]

def plot_sir_model_simulation_masks_SIRD(result, days, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Infected', 'Recovered', 'Dead']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SIRD Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def generate_sir_model_data_masks_SIRD(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 1, 0, 0]
    # Solve the differential equations
    result = odeint(sir_model_with_masks_SIRD, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    # Introduce seasonal variations using a sinusoidal function if requested
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]
    # Add random noise if requested
    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise
    # Clip negative values to zero
    result[result < 0] = 0
    # Create DataFrame
    df = pd.DataFrame(result, columns=['Susceptible', 'Infected', 'Recovered', 'Dead'])
    return (df, result)

def mse_loss_SIRD(params, t, data, y0):
    params_dict = {
        'beta': params[0],
        'gamma': params[1],
        'delta': params[2],
        'z': params[3]
    }
    result = odeint(sir_model_with_masks_SIRD, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Infected'])**2 +
                   (result[:, 2] - data['Recovered'])**2 +
                   (result[:, 3] - data['Dead'])**2)
    return loss

def optimize_params_SIRD(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Infected'][0], data['Recovered'][0], data['Dead'][0]]
    bounds = [
        (0, 0.1),   # beta
        (0, 0.5),   # gamma
        (0, 0.2),   # delta
        (0, 0.5)    # z
    ]
    result_min = minimize(mse_loss_SIRD, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict_SIRD(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict_SIRD(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'gamma': optimized_params[1],
        'delta': optimized_params[2],
        'z': optimized_params[3]
    }
    return params_dict


def sir_model_with_masks_SEIRD(y, t, params):
    """
    SEIRD model differential equations
    y: list of [S, E, I, R, D] values at time t
    t: time
    params: dictionary of parameters
    """
    S, E, I, R, D = y
    N=S+E+I+R+D
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    delta = params['delta']
    q = params['q']
    z = params['z']
    dSdt = -beta * S * I/N
    dEdt = beta * S * I/N - sigma * E - q * E
    dIdt = sigma * E - gamma * I - delta * I - z * I
    dRdt = gamma * I + z * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def plot_sir_model_simulation_masks_SEIRD(result, days, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SEIRD Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def generate_sir_model_data_masks_SEIRD(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 0, 1, 0, 0]
    # Solve the differential equations
    result = odeint(sir_model_with_masks_SEIRD, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    # Introduce seasonal variations using a sinusoidal function if requested
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]
    # Add random noise if requested
    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise
    # Clip negative values to zero
    result[result < 0] = 0
    # Create DataFrame
    df = pd.DataFrame(result, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead'])
    return (df, result)

def mse_loss_SEIRD(params, t, data, y0):
    params_dict = {
        'beta': params[0],
        'sigma': params[1],
        'gamma': params[2],
        'delta': params[3],
        'q': params[4],
        'z': params[5]
    }
    result = odeint(sir_model_with_masks_SEIRD, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Exposed'])**2 +
                   (result[:, 2] - data['Infected'])**2 +
                   (result[:, 3] - data['Recovered'])**2 +
                   (result[:, 4] - data['Dead'])**2)
    return loss

def optimize_params_SEIRD(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Exposed'][0], data['Infected'][0], data['Recovered'][0], data['Dead'][0]]
    bounds = [
        (0, 0.1),   # beta
        (0, 0.5),   # sigma
        (0, 0.5),   # gamma
        (0, 0.2),   # delta
        (0, 0.5),   # q
        (0, 0.5)    # z
    ]
    result_min = minimize(mse_loss_SEIRD, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict_SEIRD(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict_SEIRD(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'sigma': optimized_params[1],
        'gamma': optimized_params[2],
        'delta': optimized_params[3],
        'q': optimized_params[4],
        'z': optimized_params[5]
    }
    return params_dict


#%% Code modélisation avec vaccination
def sir_model_with_vaccination(y, t, params):
        """
        SEIRDV model differential equations
        y: list of [S, E, I, R, V, D] values at time t
        t: time
        params: dictionary of parameters
        """
        S, E, I, R, V, D = y
        N = S + E + I + R + V
        beta = params['beta']
        eta = params['eta']
        gamma = params['gamma']
        phi = params['phi']
        alpha = params['alpha']
        dSdt = -beta * S * I/N - alpha * S
        dEdt = beta * S * I/N - eta * E
        dIdt = eta * E - gamma * I
        dRdt = gamma * (1 - phi) * I
        dDdt = gamma * phi * I
        dVdt = alpha * S
        return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]

def plot_sir_model_simulation(result, days, population_size, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Vaccinated', 'Dead']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SEIRDV Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

def generate_sir_model_data(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 100, 0, 100, 0, 0, 0]
    # Solve the differential equations
    result = odeint(sir_model_with_vaccination, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    # Introduce seasonal variations using a sinusoidal function if requested
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]
    # Add random noise if requested
    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise
    # Clip negative values to zero
    result[result < 0] = 0
    # Create DataFrame
    df = pd.DataFrame(result, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Vaccinated', 'Dead'])
    return (df, result)

def bilateral_filter(df, kernel_size=23, sigma_color=777777, sigma_space=70):
    """
    Applies bilateral filter to each column in the DataFrame.
    Args:
        df: Input DataFrame.
        kernel_size: Size of the filter kernel (odd integer).
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.
    Returns:
        Filtered DataFrame.
    """
    filtered_df = pd.DataFrame(index=df.index)  # Set the index explicitly
    for col in df.columns:
        filtered_col = cv2.bilateralFilter(df[col].values.astype(np.float32), kernel_size, sigma_color, sigma_space)
        filtered_df[col] = filtered_col
    return filtered_df

def mse_loss(params, t, data, y0):
    params_dict = {
        'beta': params[0],
        'eta': params[1],
        'gamma': params[2],
        'phi': params[3],
        'alpha': params[4],
    }
    result = odeint(sir_model_with_vaccination, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Exposed'])**2 +
                   (result[:, 2] - data['Infected'])**2 +
                   (result[:, 3] - data['Recovered'])**2 +
                   (result[:, 4] - data['Vaccinated'])**2 +
                   (result[:, 5] - data['Dead'])**2)
    return loss

def optimize_params(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Exposed'][0], data['Infected'][0], data['Recovered'][0], data['Vaccinated'][0], data['Dead'][0]]
    bounds = [
        (0, 0.1),  # beta
        (0, 0.5),     # eta
        (0, 0.5),     # gamma
        (0, 0.2),     # phi
        (0, 0.5),     # alpha
    ]
    lower_bounds = [0, 0, 0, 0, 0]
    upper_bounds = [1, 1, 1, 1, 1]
    result_min = minimize(mse_loss, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'eta': optimized_params[1],
        'gamma': optimized_params[2],
        'phi': optimized_params[3],
        'alpha': optimized_params[4]
    }
    return params_dict

def calculate_summed_error(og_params, optimized_params):
    summed_error = sum(abs(float(optimized_params[key]) - float(og_params[key])) for key in og_params)
    return summed_error

#%% Set up de la bibliographie
BIBLIOGRAPHY = {
    'Prorok': {
        'authors': 'Aleksandra Prorok & Marcin Dwużnik',
        'year': 2024,
        'title': 'Epidemiology modeling',
        'link': 'https://github.com/prorokaleksandra/EpidemiologyModeling/blob/main/Report.pdf',
},
    'tamasiga': {
        'title':'An extended SEIRDV compartmental model: case studies of the spread of COVID-19 and vaccination in Tunisia and South Africa',
        'authors':'Tamasiga & Phemelo and Onyeaka, Helen and Umenweke, Great C and Uwishema, Olivier',
        'journal' : 'Annals of Medicine and Surgery',
        'volume':'85',
        'number':'6',
        'pages':'2721--2730',
        'year':2023,
        'publisher':'LWW'
},
    'melo':{
        'title':'Modeling COVID-19 spread through the SEIRD epidemic model and optimal control',
        'authors':'Melo & Luz',
        'journal':'Proceedings of GREAT Day',
        'volume':'2021',
        'number':'1',
        'pages':'19',
        'year':2022
},
    'Kukuseva':{
        'title':'Mathematical Analysis and Simulation of Measles Infection Spread with SEIRV+ D Model',
        'authors':'Kukuseva & Maja and Stojkovic, Natasa and Martinovska Bande, Cveta and Koceva Lazarova, Limonka',
        'journal':'ICT Innovations 2023 Web proceedings',
        'pages':'39--47',
        'year':2024
},
    'Antonelli':{
        'title':'Switched forced SEIRDV compartmental models to monitor COVID-19 spread and immunization in Italy',
        'authors':'Antonelli & E and Piccolomini, EL and Zama',
        'journal':'Infectious Disease Modelling',
        'volume':'7',
        'pages':'1--15',
        'year':2022
},
    'vynnycky':{
        'title':'An introduction to infectious disease modelling',
        'authors': 'Vynnycky & Emilia and White, Richard',
        'year':2010,
        'publisher':'Oxford university press'
}
}
# Variable globale pour suivre les citations utilisées
if 'cited_refs' not in st.session_state:
    st.session_state.cited_refs = set()
def cite(key):
    """Fonction pour citer une référence"""
    if key in BIBLIOGRAPHY:
        st.session_state.cited_refs.add(key)
        year = BIBLIOGRAPHY[key]['year']
        authors = BIBLIOGRAPHY[key]['authors'].split(',')[0]  # Premier auteur
        if 'et al.' not in authors and '&' in BIBLIOGRAPHY[key]['authors']:
            return f"({authors.split('&')[0].strip()} et al., {year})"
        return f"({authors}, {year})"
    return f"[{key}?]"
def generate_bibliography():
    """Génère la bibliographie des sources citées"""
    if not st.session_state.cited_refs:
        return "Aucune référence citée."
    html = ""
    for i, key in enumerate(sorted(st.session_state.cited_refs), 1):
        ref = BIBLIOGRAPHY[key]
        authors = ref['authors']
        year = ref['year']
        title = ref['title']
        html += f'<div class="biblio-item">[{i}] {authors} ({year}). <em>{title}</em>'
        if 'journal' in ref:
            html += f'{ref["journal"]}'
            if 'volume' in ref:
                html += f', {ref["volume"]}'
            if 'pages' in ref:
                html += f', {ref["pages"]}'
        elif 'publisher' in ref:
            html += f'{ref["publisher"]}'
        if 'doi' in ref:
            html += f' <a href="{ref["doi"]}" target="_blank">[DOI]</a>'
        elif 'link' in ref:
            html += f' <a href="{ref["link"]}" target="_blank">[Lien]</a>'
        html += '.</div>'
    return (html)


#%% Set up des classes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inclusive+Sans:ital,wght@0,300..700;1,300..700&display=swap');
        html, body, * {
        font-family: 'Inclusive Sans', sans-serif;
            }
/* Sidebar générale */
section[data-testid="stSidebar"] {
    background-color: #C6E8F3;
    color: black;
    font-family: 'Inclusive Sans', sans-serif;
}
/* Titre Sidebar */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 2px;
}
/* Texte des options radio */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 16px;
}
/* Espacement entre options */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    margin-bottom: 6px;
}

.header {
    padding: 20px; /*hauteur*/
    border-radius: 10px;
    margin-bottom: 30px; /*espace pour le texte en dessous*/
    font-family: 'Inclusive Sans', sans-serif;
}
.header h1 {
    color: black;
    text-align: center;
    margin: 10px;
    font-family: 'Inclusive Sans', sans-serif;
    margin-top: -90px;
}

.header h2 {
    color: black;
    text-align: left;
    margin: 0px;
    margin-bottom: -100px;
    margin-top: -50px;
    font-family: 'Inclusive Sans', sans-serif;
    font-size: 40px;
}     
.header h3 {
    color: black;
    text-align: justified;
    margin: 0px;
    margin-bottom: -10px;
    margin-top: 0px;
    font-family: 'Inclusive Sans', sans-serif;
    font-size: 20px;
}    

.paragraph{
    text-align: justify;
    text-indent:30px;
    margin-bottom:10px;
    font-family: 'Inclusive Sans', sans-serif;
}
            
/* Police pour Graphviz */
[data-testid="stGraphVizChart"] text,
[data-testid="stGraphVizChart"] svg text {
    font-family: 'Inclusive Sans', sans-serif !important;
    }             
/* Police pour LaTeX */
    [data-testid="stMarkdown"] .katex,
    [data-testid="stMarkdown"] .katex * {
        text-align:left;
        margin-top: px;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

#%% SIDEBAR
st.sidebar.markdown(
    "<div class='sidebar-title'>Navigation</div>",
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "",
    ["Introduction", "Modèle avec restrictions", "Modèle avec vaccination","Bibliographie"]
)

#%% Page d'introduction
if page == "Introduction":
    st.markdown("""
<div class="header">
    <h1>Le modèle SEIRD(V) en épidémiologie : simulations sous certaines conditions</h1>  
</div>          
""", unsafe_allow_html=True)
    citations = [cite('tamasiga'), cite('Kukuseva')]
    citations_clean = [c.strip('()') for c in citations]
    st.markdown(f"""
        <p class = "paragraph">
    Cette interface va permettre de simuler des modèles épidémiologiques. Deux aspects de l'épidémiologie sont simulables : les restrictions (masques, confinement et isolement) à travers les modèles SIR, SEIR, SIRD et SEIRD, et la vaccination avec un modèle SEIRVD. 
        </p>
        <p class = "paragraph">
    Les modèles avec restrictions sont basés sur la littérature {cite('melo')} et adaptés et simplifiés pour cette interface.
        </p>
        <p class = "paragraph">
    Le modèle avec vaccination est basé sur la littérature {cite('Antonelli')} et adaptés pour cette interface. Il est possible de simuler différentes épidémies avec des paramètres estimés à partir de données réelles ({', '.join(citations_clean)}).
        </p>
                """,unsafe_allow_html=True)
    st.markdown(f"""
        <p class = "paragraph">
    Le code utilisé pour les simulations est celui du projet d'Aleksandra Prorok et de Marcin Dwużnik {cite('Prorok')}. Les modèles et équations sont basées sur les travaux de recherche de l'équipe de recherche de Melo {cite('melo')} pour le modèle SEIRD classique, et de Tamasiga {cite('tamasiga')} pour le modèle SEIRDV.
    Le code et les modèles ont été adaptés pour cette présentation.
        </p>
                """,unsafe_allow_html=True)
    st.markdown("""
        <p class = "paragraph">
    
        </p>
                """,unsafe_allow_html=True)
    st.markdown("""
        <p class = "paragraph">
    Interface réalisée par Justine Marcaillou, M2 MODE, 2025-2026.
        </p>
                """,unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.image("logo-institut-agro.png")
    with col2:
        st.image("OIP.jpg")


#%% Page modèles avec restrictions
elif page == "Modèle avec restrictions":
    st.markdown("""<div class="header">
    <h2 style="text-align: center";>Modèles avec restrictions</h2>
    </div>""",unsafe_allow_html=True)
    st.markdown(f"""
        <p class = "paragraph">
    Il est possible de simuler quatre modèles : SIR, SIRD, SEIR et SEIRD. Ils sont tous basés sur les travaux de recherche sur les modèles SEIRD et l'étude de l'impact du contrôle dessus {cite('melo')}. 
    L'application d'un "contrôle" consiste à appliquer un coefficient de confinement et un coefficient d'isolement respectivement aux exposés et aux infectés. Dans leurs travaux, ces coefficients sont fonction du temps, mais par simplification ici ce sont des paramètres à valeur fixe.
    Pour modéliser le port du masque sans confinement ou isolement, il suffit de diminuer β. Par simplification, ici β est un unique paramètre, mais il peut être décomposé {cite('vynnycky')} en le produit du taux de contact par la probabilité de transmission de la maladie.
    Ainsi, le port du masque permettant de diminuer le taux de contact, il peut être modélisé ici en diminuant β directement.
    Les modèles autres que SEIRD ont donc été adaptés pour la présentation.
        </p>
        <p class = "paragraph">
        Les modèles autres que SEIRD ont été adaptés du modèle SEIRD.
        </p>
                """,unsafe_allow_html=True)
    
    st.sidebar.header("Choix du modèle")
    include_E = st.sidebar.checkbox("Compartiment E (Exposés)", value=True)
    include_D = st.sidebar.checkbox("Compartiment D (Décédés)", value=True)
    if include_E and include_D:
        model_type = "SEIRD"
    elif include_E and not include_D:
        model_type = "SEIR"
    elif not include_E and include_D:
        model_type = "SIRD"
    else:
        model_type = "SIR"
    st.sidebar.info(f"Modèle choisi : {model_type}")
    col1,col2=st.columns([3,3])
    with col1:
        st.markdown("""
    <div class="header">
        <h3>Equations du modèle :</h3>  
    </div>
    <div style="margin-top: -50px;"></div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
    <div class="header">
        <h3>Schéma du modèle :</h3>  
    </div> 
    <div style="margin-bottom: -70px;"></div>
    <div>         
    """, unsafe_allow_html=True)
    
    cola,colb,colc = st.columns([1,1,2])
    with cola:
        if model_type == "SEIRD":
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} &= -\beta S I \\
    \frac{dE}{dt} &= \beta S I - \sigma E - q E\\
    \frac{dI}{dt} &= \sigma E - \gamma I - \delta I - z I \\
    \frac{dR}{dt} &= \gamma I + z I \\
    \frac{dD}{dt} &= \delta I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        elif model_type == "SEIR":
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I \\
    \frac{dE}{dt} = \beta S I - \sigma E - q I \\
    \frac{dI}{dt} = \sigma E - \gamma I - z I \\
    \frac{dR}{dt} = \gamma I + z I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        elif model_type == "SIRD":
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I  \\
    \frac{dI}{dt} = \beta S I - \gamma I - \delta I - z I \\
    \frac{dR}{dt} = \gamma I + z I \\
    \frac{dD}{dt} = \delta I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        else:
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I \\
    \frac{dI}{dt} = \beta S I - \gamma I - z I \\
    \frac{dR}{dt} = \gamma I + z I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)

    with colb:
        if model_type == "SEIRD":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate</p>
    <p><strong>γ:</strong> recovery rate</p>
    <p><strong>σ:</strong> incubation rate</p>
    <p><strong>z:</strong> portion of people going into isolation</p>
    <p><strong>q:</strong> portion of people going into quarantine</p>
    <p><strong>δ:</strong> mortality rate</p>
    </div>
    """, unsafe_allow_html=True)
        elif model_type == "SEIR":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate</p>
    <p><strong>γ:</strong> recovery rate</p>
    <p><strong>σ:</strong> incubation rate</p>
    <p><strong>z:</strong> portion of people going into isolation</p>
    <p><strong>q:</strong> portion of people going into quarantine</p>
    </div>
    """, unsafe_allow_html=True)
        elif model_type == "SIRD":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate</p>
    <p><strong>γ:</strong> recovery rate</p>
    <p><strong>z:</strong> portion of people going into isolation</p>
    <p><strong>δ:</strong> mortality rate</p>
    </div>
    """, unsafe_allow_html=True)
        else:
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate</p>
    <p><strong>γ:</strong> recovery rate</p>
    <p><strong>z:</strong> portion of people going into isolation</p>
    </div>
    """, unsafe_allow_html=True)

    with colc:
        if model_type == "SEIRD":
            st.graphviz_chart("""
    digraph SEIRD {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
        edge [fontname="Inclusive Sans"];
        S [label="S\nSusceptibles", fillcolor="#87CEEB"];
        E [label="E\nExposés", fillcolor="#FFD700"];
        I [label="I\nInfectieux", fillcolor="#FF6347"];
        R [label="R\nRétablis", fillcolor="#90EE90"];
        D [label="D\nDécédés", fillcolor="#696969"];
        S -> E [label="β"];
        E -> I [label="σ"];
        E -> R [label="q"];
        I -> R [label="γ, z"];
        I -> D [label="δ"];
    }
    """)

        elif model_type == "SEIR":
            st.graphviz_chart("""
digraph SEIR {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
    edge [fontname="Inclusive Sans"];
    S [label="S\nSusceptibles", fillcolor="#87CEEB"];
    E [label="E\nExposés", fillcolor="#FFD700"];
    I [label="I\nInfectieux", fillcolor="#FF6347"];
    R [label="R\nRétablis", fillcolor="#90EE90"];
    S -> E [label="β"];
    E -> I [label="σ"];
    E -> R [label="q"];
    I -> R [label="γ, z"];
}
""")
        elif model_type == "SIRD":
            st.graphviz_chart("""
  digraph SIRD {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
    edge [fontname="Inclusive Sans"];
    S [label="S\nSusceptibles", fillcolor="#87CEEB"];
    I [label="I\nInfectieux", fillcolor="#FF6347"];
    R [label="R\nRétablis", fillcolor="#90EE90"];
    D [label="D\nDécédés", fillcolor="#696969"];
    S -> I [label="β"];
    I -> R [label="γ, z"];
    I -> D [label="δ"];
}
""")
        else:
            st.graphviz_chart("""
digraph SIR {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
    edge [fontname="Inclusive Sans"];
    S [label="S\nSusceptibles", fillcolor="#87CEEB"];
    I [label="I\nInfectieux", fillcolor="#FF6347"];
    R [label="R\nRétablis", fillcolor="#90EE90"];
    S -> I [label="β"];
    I -> R [label="γ, z"];
}
""")
    st.markdown("""
<div class="header">
    <h3>Simulation du modèle :</h3>  
</div>
<div style="margin-bottom: -50px;"></div>      
""", unsafe_allow_html=True)
    st.write(f"Modèle choisi : {model_type}. Ajustez les différents paramètres :")

    col1,col2= st.columns(2)
    with st.sidebar:
        st.write("**Paramètres du modèle :**")
        beta = st.slider(
            'Infection Rate (β)',
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01
        )
        if model_type == "SEIRD":
            params = {
                'beta': beta,
                'sigma': st.slider('Incubation Rate (σ)', 0.0, 1.0, 0.14, 0.01),
                'gamma': st.slider('Recovery Rate (γ)', 0.0, 1.0, 1/21, 0.01),
                'delta': st.slider('Mortality Rate (δ)', 0.0, 1.0, 0.03, 0.001),
                'q': None,  # Will be set in col2
                'z': None   # Will be set in col2
            }
        elif model_type == "SEIR":
            params = {
                'beta': beta,
                'sigma': st.slider('Incubation Rate (σ)', 0.0, 1.0, 0.14, 0.01),
                'gamma': st.slider('Recovery Rate (γ)', 0.0, 1.0, 1/21, 0.01),
                'q': None,  # Will be set in col2
                'z': None   # Will be set in col2
            }
        elif model_type == "SIRD":
            params = {
                'beta': beta,
                'gamma': st.slider('Recovery Rate (γ)', 0.0, 1.0, 1/21, 0.01),
                'delta': st.slider('Mortality Rate (δ)', 0.0, 1.0, 0.03, 0.001),
                'z': None   # Will be set in col2
            }
        elif model_type == "SIR":
            params = {
                'beta': beta,
                'gamma': st.slider('Recovery Rate (γ)', 0.0, 1.0, 1/21, 0.01),
                'z': None   # Will be set in col2
            }
    with col1:
        st.write("**Conditions de la simulation :**")
        days = st.slider('Number of Days', min_value=10, max_value=1825, step=10)
        population_size = st.slider('Population Size', min_value=50, max_value=10000, step=100)
        noise_std = st.checkbox('Include Noise Standard Deviation')
        seasonal_amplitude = 'Include Seasonal Amplitude'
    with col2:
        st.write("**Paramètres d'intervention :**")
        if model_type in ["SEIRD", "SEIR"]:
            q = st.slider('Isolation Rate of Exposed (q)', 0.0, 1.0, 0.05, 0.01)
            params['q'] = q
        if model_type in ["SEIRD", "SEIR", "SIRD", "SIR"]:
            z = st.slider('Isolation Rate of Infected (z)', 0.0, 1.0, 0.05, 0.01)
            params['z'] = z
    cola,colb=st.columns([2,1])
    with cola:
        if model_type == "SEIRD":
            df, result = generate_sir_model_data_masks_SEIRD(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)
            plot_sir_model_simulation_masks_SEIRD(result, days)
        elif model_type == "SEIR":
            df, result = generate_sir_model_data_masks_SEIR(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)
            plot_sir_model_simulation_masks_SEIR(result, days)
        elif model_type == "SIRD":
            df, result = generate_sir_model_data_masks_SIRD(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)
            plot_sir_model_simulation_masks_SIRD(result, days)
        elif model_type == "SIR":
            df, result = generate_sir_model_data_masks_SIR(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)
            plot_sir_model_simulation_masks_SIR(result, days)

    if 'stored_plots_models' not in st.session_state:
        st.session_state.stored_plots_models = []
    col_btna, col_btnb = st.columns([1, 4])
    with col_btna:
        if st.button('Stocker ce graphique', key='store_model'):
            st.session_state.stored_plots_models.append({
                'result': result.copy(),
                'days': days,
                'population_size': population_size,
                'params': params.copy(),
                'model_type': model_type,
                'timestamp': pd.Timestamp.now()
            })
            st.success(f'Graphique {len(st.session_state.stored_plots_models)} stocké !')
    with col_btnb:
        if st.button('Effacer tous les graphiques stockés', key='clear_models'):
            st.session_state.stored_plots_models = []
            st.success('Tous les graphiques ont été effacés !')
    if st.session_state.stored_plots_models:
        st.header('Graphiques générés :')
        for idx in range(0, len(st.session_state.stored_plots_models), 2):
            col1, col2 = st.columns(2)
            with col1:
                stored_plot = st.session_state.stored_plots_models[idx]
                st.subheader(f'Graphique {idx + 1} - {stored_plot["model_type"]}')
                if stored_plot['model_type'] == "SEIRD":
                    plot_sir_model_simulation_masks_SEIRD(
                        stored_plot['result'], 
                        stored_plot['days'], 
                        stored_plot['population_size']
                    )
                elif stored_plot['model_type'] == "SEIR":
                    plot_sir_model_simulation_masks_SEIR(
                        stored_plot['result'], 
                        stored_plot['days'], 
                        stored_plot['population_size']
                    )
                elif stored_plot['model_type'] == "SIRD":
                    plot_sir_model_simulation_masks_SIRD(
                        stored_plot['result'], 
                        stored_plot['days'], 
                        stored_plot['population_size']
                    )
                elif stored_plot['model_type'] == "SIR":
                    plot_sir_model_simulation_masks_SIR(
                        stored_plot['result'], 
                        stored_plot['days'], 
                        stored_plot['population_size']
                    )
                
                with st.expander(f'Voir les paramètres du graphique {idx + 1}'):
                    st.write(stored_plot['params'])
            
            if idx + 1 < len(st.session_state.stored_plots_models):
                with col2:
                    stored_plot = st.session_state.stored_plots_models[idx + 1]
                    st.subheader(f'Graphique {idx + 2} - {stored_plot["model_type"]}')
                    
                    # Appeler la fonction de plot correspondante au modèle
                    if stored_plot['model_type'] == "SEIRD":
                        plot_sir_model_simulation_masks_SEIRD(
                            stored_plot['result'], 
                            stored_plot['days'], 
                            stored_plot['population_size']
                        )
                    elif stored_plot['model_type'] == "SEIR":
                        plot_sir_model_simulation_masks_SEIR(
                            stored_plot['result'], 
                            stored_plot['days'], 
                            stored_plot['population_size']
                        )
                    elif stored_plot['model_type'] == "SIRD":
                        plot_sir_model_simulation_masks_SIRD(
                            stored_plot['result'], 
                            stored_plot['days'], 
                            stored_plot['population_size']
                        )
                    elif stored_plot['model_type'] == "SIR":
                        plot_sir_model_simulation_masks_SIR(
                            stored_plot['result'], 
                            stored_plot['days'], 
                            stored_plot['population_size']
                        )
                    
                    with st.expander(f'Voir les paramètres du graphique {idx + 2}'):
                        st.write(stored_plot['params'])

#%% Page modèle avec vaccination
elif page == "Modèle avec vaccination":
    st.markdown("""<div class="header">
        <h2 style="text-align: center";>Modèle avec vaccination</h2>
        </div>""",unsafe_allow_html=True)
    st.markdown(f"""
            <p class = "paragraph">
        Le modèle est un modèle SEIRVD écrit à partir du Covid-19 en Italie {cite('Antonelli')}. Il présente plusieurs hypothèses : la taille de la population est constante, on ne prend pas en compte ni les naissances ni les décès. De plus, l'immunité est acquise à vie, par vaccination ou guérison. 
            </p>
                    """,unsafe_allow_html=True)
    
    col1,col2 = st.columns(2    )
    with col1:
        st.markdown("""
    <div class="header">
        <h3>Equations du modèle :</h3>  
    </div>
    <div style="margin-top: -50px;"></div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="header">
            <h3>Schéma du modèle :</h3>  
            <div style="margin-bottom: -70px;"></div>
        </div>          
        """, unsafe_allow_html=True)
    cola,colb,colc = st.columns([2,2,5])
    with cola:
        st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} &= -\frac{\beta S I}{N} - \alpha S \\
    \frac{dE}{dt} &= \frac{\beta S I}{N} - \eta E \\
    \frac{dI}{dt} &= \eta E - \gamma I \\
    \frac{dR}{dt} &= \gamma (1 - \varphi) I \\
    \frac{dD}{dt} &= \gamma \varphi I \\
    \frac{dV}{dt} &= \alpha S
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
    with colb:
        st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate </p>
    <p><strong>γ:</strong> recovery rate</p>
    <p><strong>η:</strong> incubation rate</p>
    <p><strong>φ:</strong> mortality rate</p>
    <p><strong>α:</strong> vaccination rate</p>              
    </div>
    """, unsafe_allow_html=True)
    with colc:
        st.graphviz_chart("""
            digraph SEIRV {
                rankdir=LR;
                node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
                edge [fontname="Inclusive Sans"];
                S [label="S\nSusceptibles", fillcolor="#87CEEB"];
                E [label="E\nExposés", fillcolor="#FFD700"];
                I [label="I\nInfectieux", fillcolor="#FF6347"];
                R [label="R\nRétablis", fillcolor="#90EE90"];
                D [label="D\nDécédés", fillcolor="#696969"];
                V [label="V\nVaccinés efficaces", fillcolor="#9370DB"];
                S -> E [label="β"];
                E -> I [label="η            ", labelfloat = True];
                I -> R [label="γ(1-φ)"];
                I -> D [label="γφ"];
                S -> V [label ="α"];
            }
        """)
    st.markdown("""
    <div class="header">
        <h3>Simulation du modèle :</h3>  
    </div>
    <div style="margin-bottom: -50px;"></div>      
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div>
        <p class = "paragraph">
                Quatre simulations sont possibles : l'épidémie de Covid-19 en Tunisie {cite('tamasiga')} et en Afrique du Nord {cite('tamasiga')}, l'épidémie de rougeole en Macédoine du Nord {cite('Kukuseva')} et une simulation libre où il est possible de faire varier tous les paramètres.
                Pour les trois simulations issues de la littérature il est possible de changer le taux de vaccination (α) afin d'explorer divers scénarios ; le paramètre est fixé au départ à la valeur associée aux données disponibles.
            </p>
    </div>
       """, unsafe_allow_html=True)

    if 'stored_plots_v' not in st.session_state:
        st.session_state.stored_plots_v = []

    st.write("**Conditions de la simulation :**")
    noise_std = st.checkbox('Include Noise Standard Deviation', key='noise_std_vaccination')

    col1, col2 = st.columns(2)
    with col1:
        days = st.slider('Number of Days', min_value=1, max_value=1825, value=365, step=10, key='days_vaccination')
    with col2:
        population_size = st.slider('Population Size', min_value=50, max_value=10000, value=1000, step=100, key='pop_vaccination')
    seasonal_amplitude = False  

    st.sidebar.write("**Choisir une épidémie :**")
    scenario = st.sidebar.radio(
        "",
        ["Covid-19 - Tunisie", "Covid-19 - Afrique du Sud", "Rougeole - Macédoine du Nord", "Personnalisé"],
        key='scenario_vaccination'
    )

    if scenario == "Covid-19 - Tunisie":
        beta, eta, gamma, phi = 0.0532, 1.49, 0.053, 0.035
        alpha = st.sidebar.slider('Vaccination rate (α)', min_value=0.0, max_value=1.0,
                                value=0.0368, format="%.4f", step=0.001, key='alpha_tunisia')

    elif scenario == "Covid-19 - Afrique du Sud":
        beta, eta, gamma, phi = 0.1, 1.06, 0.0227, 0.06
        alpha = st.sidebar.slider('Vaccination rate (α)', min_value=0.0, max_value=1.0,
                                value=0.00847, format="%.5f", step=0.001, key='alpha_southafrica')

    elif scenario == "Rougeole - Macédoine du Nord":
        beta, eta, gamma, phi = 0.9, 0.125, 0.14285, 0.02
        alpha = st.sidebar.slider('Vaccination rate (α)', min_value=0.0, max_value=1.0,
                                value=0.744, format="%.3f", step=0.001, key='alpha_macedonia')

    elif scenario == "Personnalisé":
        with st.sidebar:
            st.markdown("### Paramètres personnalisés")
            beta = st.slider('Infection Rate (β)', min_value=0.0, max_value=1.0,
                            value=0.14, step=0.01, key='beta_vaccination')
            eta = st.slider('Incubation Rate (η)', min_value=0.0, max_value=1.0,
                            value=0.14, step=0.01, key='eta_vaccination')
            gamma = st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0,
                            value=1/21, step=0.01, key='gamma_vaccination')
            alpha = st.slider('Vaccination rate (α)', min_value=0.0, max_value=1.0,
                            value=0.0055, format="%.3f", step=0.001, key='alpha_vaccination')
            phi = st.slider('Mortality Rate (φ)', min_value=0.0, max_value=1.0,
                            value=0.03, format="%.3f", step=0.001, key='phi_vaccination')

    # Affichage dans la zone principale
    st.write(f"**Paramètres de l'épidémie '{scenario}' :**")
    if scenario == "Covid-19 - Tunisie":
            st.markdown(f"""
            <p class = "paragraph">
                Les paramètres sont issus de travaux de recherche d'une review scientifique de médecine {cite('tamasiga')}.
                Le taux de vaccination (α) était de 0,0368.
            </p>""", unsafe_allow_html=True)
    elif scenario == "Covid-19 - Afrique du Sud":
        st.markdown(f"""
        <p class = "paragraph">
            Les paramètres sont issus de travaux de recherche d'une review scientifique de médecine {cite('tamasiga')}.
            Le taux de vaccination (α) était de 0,00847.
        </p>""", unsafe_allow_html=True)
    elif scenario == "Rougeole - Macédoine du Nord":
        st.markdown(f"""
        <p class = "paragraph">
            Les paramètres sont issus de travaux de recherche d'une équipe de Macédoine du Nord {cite('Kukuseva')}.
            Le taux de vaccination (α) était de 0,744.
        </p>""", unsafe_allow_html=True)


    st.write(f"β = {beta}, η = {eta}, γ = {gamma}, φ = {phi}, α = {alpha}")

    params = {
        'beta': beta,
        'eta': eta,
        'gamma': gamma,
        'phi': phi,
        'alpha': alpha,
    }

    df, result = generate_sir_model_data(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)

    col1,col2 = st.columns([2,1])
    with col1:
        plot_sir_model_simulation(result, days, population_size)

    col_btn1, col_btn2 = st.columns([1,4])
    with col_btn1:
        if st.button('Stocker ce graphique'):
            st.session_state.stored_plots_v.append({
                'result': result.copy(),
                'days': days,
                'population_size': population_size,
                'params': params.copy(),
                'timestamp': pd.Timestamp.now()
            })
            st.success(f'Graphique {len(st.session_state.stored_plots_v)} stocké !')

    with col_btn2:
        if st.button('Effacer tous les graphiques stockés'):
            st.session_state.stored_plots_v = []
            st.success('Tous les graphiques ont été effacés !')

    if st.session_state.stored_plots_v:
        st.header('Graphiques générés :')
        for idx in range(0, len(st.session_state.stored_plots_v), 2):
            col1, col2 = st.columns(2)
            with col1:
                stored_plot = st.session_state.stored_plots_v[idx]
                st.subheader(f'Graphique {idx + 1}')
                plot_sir_model_simulation(
                    stored_plot['result'], 
                    stored_plot['days'], 
                    stored_plot['population_size']
                )
                with st.expander(f'Voir les paramètres du graphique {idx + 1}'):
                    st.write(stored_plot['params'])
            if idx + 1 < len(st.session_state.stored_plots_v):
                with col2:
                    stored_plot = st.session_state.stored_plots_v[idx + 1]
                    st.subheader(f'Graphique {idx + 2}')
                    plot_sir_model_simulation(
                        stored_plot['result'], 
                        stored_plot['days'], 
                        stored_plot['population_size']      
                    )
                    with st.expander(f'Voir les paramètres du graphique {idx + 2}'):
                        st.write(stored_plot['params'])
    
#%% Page bibliographie
elif page == "Bibliographie":
    st.markdown("""<div class="header">
    <h2 style="text-align: center;">Bibliographie</h2>
    </div>""", unsafe_allow_html=True)
    if st.button("Réinitialiser la bibliographie"):
        st.session_state.cited_refs = set()
        st.rerun()
    st.markdown("""
    <style>
    .biblio-item {
        margin-bottom: 15px;
        padding-left: 30px;
        text-indent: -30px;
        line-height: 1.6;
    }
    .biblio-item a {
        color: #0066cc;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(generate_bibliography(), unsafe_allow_html=True)
