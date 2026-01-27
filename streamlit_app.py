#https://github.com/prorokaleksandra/EpidemiologyModeling/blob/main/Report.pdf

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
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
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
        'gamma': params[2]
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
        (0, 0.5)    # gamma
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
        'gamma': optimized_params[2]
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
    beta = params['beta']
    gamma = params['gamma']
    
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    
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
        'gamma': params[1]
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
        (0, 0.5)    # gamma
    ]
    result_min = minimize(mse_loss_SIR, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict_SIR(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict_SIR(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'gamma': optimized_params[1]
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
    beta = params['beta']
    gamma = params['gamma']
    alpha = params['alpha']
    delta = params['delta']
    dSdt = -beta * S * I + alpha * R
    dIdt = beta * S * I - gamma * I - delta * I
    dRdt = gamma * I - alpha * R
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
        'alpha': params[2],
        'delta': params[3]
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
        (0, 0.5),   # alpha
        (0, 0.2)    # delta
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
        'alpha': optimized_params[2],
        'delta': optimized_params[3]
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
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    alpha = params['alpha']
    delta = params['delta']
    dSdt = -beta * S * I + alpha * R 
    dEdt = beta * S * I - sigma * E 
    dIdt = sigma * E - gamma * I - delta * I
    dRdt = gamma * I - alpha * R 
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
        'alpha': params[3],
        'delta': params[4]
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
        (0, 0.5),   # alpha
        (0, 0.2)    # delta
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
        'alpha': optimized_params[3],
        'delta': optimized_params[4]
    }
    return params_dict

#%% Code modélisation avec vaccination
def sir_model_with_vaccination(y, t, params):
    """
    SIR model differential equations with vaccination
    y: list of [S, E, I, R, V, V_failed, D] values at time t
    t: time
    params: dictionary of parameters
    """
    S, E, I, R, V, V_failed, D = y
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    alpha = params['alpha']
    v_rate = params['v_rate']
    v_success = params['v_success']
    alpha_v = params['alpha_v']
    alpha_v_failed = params['alpha_v_failed']
    delta = params['delta']
    dSdt = -beta * S * I - v_rate * S + alpha_v * V + alpha * R + alpha_v_failed * V_failed
    dEdt = beta * S * I - sigma * E + V_failed * beta * I
    dIdt = sigma * E - gamma * I - delta * I
    dRdt = gamma * I - alpha * R 
    dDdt = delta * I
    dVdt = v_rate * S * v_success - alpha_v * V
    dV_failedt = v_rate * S * (1-v_success) - V_failed * beta * I - alpha_v_failed * V_failed
    return [dSdt, dEdt, dIdt, dRdt, dVdt, dV_failedt, dDdt]

def plot_sir_model_simulation(result, days, population_size, title_addon='', highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Vaccinated', 'Failed vaccination', 'Dead']
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])
    plt.xlabel('Time (days)')
    plt.ylabel('Percentage of Population')
    plt.title('SEIRDVVf Model')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

def generate_sir_model_data(population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 0, 1, 0, 0, 0, 0]
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
    df = pd.DataFrame(result, columns=['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Vaccinated', 'Failed vaccination', 'Dead'])
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
        'sigma': params[1],
        'gamma': params[2],
        'alpha': params[3],
        'v_rate': params[4],
        'v_success': params[5],
        'alpha_v': params[6],
        'alpha_v_failed': params[7],
        'delta': params[8]
    }
    result = odeint(sir_model_with_vaccination, y0, t, args=(params_dict,))
    loss = np.mean((result[:, 0] - data['Susceptible'])**2 +
                   (result[:, 1] - data['Exposed'])**2 +
                   (result[:, 2] - data['Infected'])**2 +
                   (result[:, 3] - data['Recovered'])**2 +
                   (result[:, 4] - data['Vaccinated'])**2 +
                   (result[:, 5] - data['Failed vaccination'])**2 +
                   (result[:, 6] - data['Dead'])**2)
    return loss

def optimize_params(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)
    data = bilateral_filter(data)
    y0 = [data['Susceptible'][0], data['Exposed'][0], data['Infected'][0], data['Recovered'][0], data['Vaccinated'][0], data['Failed vaccination'][0], data['Dead'][0]]
    bounds = [
        (0, 0.1),  # beta
        (0, 0.5),     # sigma
        (0, 0.5),     # gamma
        (0, 0.5),     # alpha
        (0, 0.1),     # v_rate
        (0, 1),     # v_success
        (0, 0.5),     # alpha_v
        (0, 0.5),     # alpha_v_failed
        (0, 0.2)      # delta
    ]
    lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    result_min = minimize(mse_loss, initial_params, args=(t, data, y0), bounds=bounds, method='L-BFGS-B')
    params_min = create_params_dict(result_min.x)
    error_min = calculate_summed_error(og_params, params_min)
    errors = [(params_min, 'minimize', error_min)]
    return errors

def create_params_dict(optimized_params):
    params_dict = {
        'beta': optimized_params[0],
        'sigma': optimized_params[1],
        'gamma': optimized_params[2],
        'alpha': optimized_params[3],
        'v_rate': optimized_params[4],
        'v_success': optimized_params[5],
        'alpha_v': optimized_params[6],
        'alpha_v_failed': optimized_params[7],
        'delta': optimized_params[8]
    }
    return params_dict

def calculate_summed_error(og_params, optimized_params):
    summed_error = sum(abs(float(optimized_params[key]) - float(og_params[key])) for key in og_params)
    return summed_error

#%% Set up de la bibliography
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
    ["Introduction", "Modèle avec restrictions", "Modèle avec vaccination", "Conclusion et discussion","Bibliographie"]
)

#%% Page d'introduction
if page == "Introduction":
    st.markdown("""
<div class="header">
    <h1>Le modèle SEIRD(V) en épidémiologie : simulations sous certaines conditions</h1>  
</div>          
""", unsafe_allow_html=True)
    st.markdown("""
        <p class = "paragraph">
        Le modèle choisi est un modèle SEIRD classique que l'on va simuler en modulant plusieurs variables : la population initiale, la probabilité d'infection,... Ces variables sont modulables entre plusieurs valeurs tirées de la bibliographie.
        </p>
        <p class = "paragraph">
        Deux choix de simulations sont possibles : une simulation avec des restrictions (port du masque et confinement) et une simulation avec des vaccins. La simulation avec des vaccins est donc un modèle plus complet et complexe : un modèle SEIRVD.
        </p>
                """,unsafe_allow_html=True)
    st.markdown(f"""
        <p class = "paragraph">
    Le code utilisé pour les simulations est celui du projet d'Aleksandra Prorok et de Marcin Dwużnik {cite('Prorok')} et a été adapté pour cette présentation. Les modèles et équations sont basées sur les travaux de recherche de l'équipe de recherche de Melo {cite('melo')} pour le modèle SEIRD classique, et de Tamasiga {cite('tamasiga')} pour le modèle SEIRDV.
        </p>
                """,unsafe_allow_html=True)


#%% Page modèles avec restrictions
elif page == "Modèle avec restrictions":
    st.markdown("""<div class="header">
    <h2 style="text-align: center";>Modèles avec restrictions</h2>
    </div>""",unsafe_allow_html=True)
    st.markdown("""
        <p class = "paragraph">
    Il est possible de simuler quatre modèles : SIR, SIRD, SEIR et SEIRD. 
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
    \frac{dS}{dt} &= -\beta S I + \alpha R \\
    \frac{dE}{dt} &= \beta S I - \sigma E \\
    \frac{dI}{dt} &= \sigma E - \gamma I - \delta I \\
    \frac{dR}{dt} &= \gamma I - \alpha R \\
    \frac{dD}{dt} &= \delta I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        elif model_type == "SEIR":
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I + \alpha R \\
    \frac{dE}{dt} = \beta S I - \sigma E \\
    \frac{dI}{dt} = \sigma E - \gamma I \\
    \frac{dR}{dt} = \gamma I - \alpha R \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        elif model_type == "SIRD":
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I + \alpha R \\
    \frac{dI}{dt} = \beta S I - \gamma I - \delta I \\
    \frac{dR}{dt} = \gamma I - \alpha R \\
    \frac{dD}{dt} = \delta I \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
        else:
            st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} = -\beta S I + \alpha R \\
    \frac{dI}{dt} = \beta S I - \gamma I \\
    \frac{dR}{dt} = \gamma I - \alpha R \\
    \end{aligned}
    $$
    """, unsafe_allow_html=True)

    with colb:
        if model_type == "SEIRD":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate - it is influenced by the number of contacts and probability of infecting,</p>
    <p><strong>γ:</strong> recovery rate,</p>
    <p><strong>σ:</strong> infection rate in exposed individuals,</p>
    <p><strong>α:</strong> rate at which individuals leave the Recovered compartment,</p>
    <p><strong>δ:</strong> mortality rate.</p>
    </div>
    """, unsafe_allow_html=True)
        elif model_type == "SEIR":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate - it is influenced by the number of contacts and probability of infecting,</p>
    <p><strong>γ:</strong> recovery rate,</p>
    <p><strong>σ:</strong> infection rate in exposed individuals,</p>
    <p><strong>α:</strong> rate at which individuals leave the Recovered compartment,</p>
    </div>
    """, unsafe_allow_html=True)
        elif model_type == "SIRD":
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate - it is influenced by the number of contacts and probability of infecting,</p>
    <p><strong>γ:</strong> recovery rate,</p>
    <p><strong>α:</strong> rate at which individuals leave the Recovered compartment,</p>
    <p><strong>δ:</strong> mortality rate.</p>
    </div>
    """, unsafe_allow_html=True)
        else:
            st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate - it is influenced by the number of contacts and probability of infecting,</p>
    <p><strong>γ:</strong> recovery rate,</p>
    <p><strong>α:</strong> rate at which individuals leave the Recovered compartment,</p>
    </div>
    """, unsafe_allow_html=True)

    with colc:
        if model_type == "SEIRD":
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
            S -> E [label="β"];
            E -> I [label="σ"];
            I -> R [label="γ"];
            I -> D [label="δ"];
            R -> S [label="α"];
        }
    """)
        elif model_type == "SEIR":
            st.graphviz_chart("""
        digraph SEIRV {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
            edge [fontname="Inclusive Sans"];
            S [label="S\nSusceptibles", fillcolor="#87CEEB"];
            E [label="E\nExposés", fillcolor="#FFD700"];
            I [label="I\nInfectieux", fillcolor="#FF6347"];
            R [label="R\nRétablis", fillcolor="#90EE90"];
            S -> E [label="β"];
            E -> I [label="σ"];
            I -> R [label="γ"];
            R -> S [label="α"];
        }
    """)
        elif model_type == "SIRD":
            st.graphviz_chart("""
        digraph SEIRV {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
            edge [fontname="Inclusive Sans"];
            S [label="S\nSusceptibles", fillcolor="#87CEEB"];
            I [label="I\nInfectieux", fillcolor="#FF6347"];
            R [label="R\nRétablis", fillcolor="#90EE90"];
            D [label="D\nDécédés", fillcolor="#696969"];
            S -> I [label="β"];
            I -> R [label="γ"];
            I -> D [label="δ"];
            R -> S [label="α"];
        }
    """)
        else:
            st.graphviz_chart("""
        digraph SEIRV {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inclusive Sans"];
            edge [fontname="Inclusive Sans"];
            S [label="S\nSusceptibles", fillcolor="#87CEEB"];
            I [label="I\nInfectieux", fillcolor="#FF6347"];
            R [label="R\nRétablis", fillcolor="#90EE90"];
            S -> I [label="β"];
            I -> R [label="γ"];
            R -> S [label="α"];
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
    with col1:
        st.write("**Conditions de la simulation :**")
        noise_std = st.checkbox('Include Noise Standard Deviation')
        days = st.slider('Number of Days', min_value=10, max_value = 1825, step=10)
        population_size = st.slider('Population Size', min_value=50, max_value = 10000, step=100)
        seasonal_amplitude = 'Include Seasonal Amplitude'
    with col2:
        st.write("**Paramètres du modèle :**")
        prob_of_infecting = st.number_input('Probability of Infecting', value=1/100, format="%.3f", step=0.001)
        avg_no_contacts_per_individual = st.number_input('Average Number of Contacts per Individual', value=12)
        beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
        if model_type == "SEIRD":
            params = {
        'beta': beta,
        'sigma': st.slider('Infection Rate (σ)', min_value=0.0, max_value=1.0, value=0.14, step=0.01),
        'gamma': st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0, value=1/21, step=0.01),
        'alpha': st.slider('Rate at which individuals lose their immunity (α)', min_value=0.0, max_value=1.0, value=0.0055, format="%.3f", step=0.001),
        'delta': st.slider('Mortality Rate (δ)', min_value=0.0, max_value=1.0, value=0.03, format="%.3f", step=0.001)
    }
        elif model_type == "SEIR":
            params = {
        'beta': beta,
        'sigma': st.slider('Infection Rate (σ)', min_value=0.0, max_value=1.0, value=0.14, step=0.01),
        'gamma': st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0, value=1/21, step=0.01),
        'alpha': st.slider('Rate at which individuals lose their immunity (α)', min_value=0.0, max_value=1.0, value=0.0055, format="%.3f", step=0.001),
    }
        elif model_type == "SIRD":
            params = {
        'beta': beta,
        'gamma': st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0, value=1/21, step=0.01),
        'alpha': st.slider('Rate at which individuals lose their immunity (α)', min_value=0.0, max_value=1.0, value=0.0055, format="%.3f", step=0.001),
        'delta': st.slider('Mortality Rate (δ)', min_value=0.0, max_value=1.0, value=0.03, format="%.3f", step=0.001)
    }
        elif model_type == "SIR":
            params = {
        'beta': beta,
        'gamma': st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0, value=1/21, step=0.01),
        'alpha': st.slider('Rate at which individuals lose their immunity (α)', min_value=0.0, max_value=1.0, value=0.0055, format="%.3f", step=0.001),
    }
    cola,colb=st.columns([5,1])
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
    st.markdown("""
            <p class = "paragraph">
        Le modèle choisi est un modèle où l'on modélise les individus vaccinés par deux nouveaux compartiments, en fonction de l'efficacité du vaccin : le vaccin est soit efficace (V) ou inefficace (V_failed). Cependant, dans tous les cas l'imumunité n'est pas définitive : il est toujours possible de retomber malade (donc de retourner dans le compartiment S).
            </p>
                    """,unsafe_allow_html=True)
    st.markdown("""
    <div class="header">
        <h3>Equations du modèle :</h3>  
    </div>
    <div style="margin-top: -50px;"></div>
    """, unsafe_allow_html=True)
    cola,colb = st.columns(2)
    with cola:
        st.markdown(r"""
    $$
    \begin{aligned}
    \frac{dS}{dt} &= -\beta S I - v_{rate} S + \alpha_v V + \alpha R + \alpha_{v_failed} V_{failed} \\
    \frac{dE}{dt} &= \beta S I - \sigma E + \beta I V_{failed} \\
    \frac{dI}{dt} &= \sigma E - \gamma I - \delta I \\
    \frac{dR}{dt} &= \gamma I - \alpha R \\
    \frac{dD}{dt} &= \delta I \\
    \frac{dV}{dt} &= v_{rate} S \cdot v_{success} - \alpha_v V \\
    \frac{dV_{failed}}{dt} &= v_{rate} S (1 - v_{success}) - \beta I V_{failed} - \alpha_{v_failed} V_{failed}
    \end{aligned}
    $$
    """, unsafe_allow_html=True)
    with colb:
        st.markdown("""
    <div style="line-height: 1;">
    <p><strong>β:</strong> infection rate - it is influenced by the number of contacts and probability of infecting,</p>
    <p><strong>γ:</strong> recovery rate,</p>
    <p><strong>σ:</strong> infection rate in exposed individuals,</p>
    <p><strong>α:</strong> rate at which individuals leave the Recovered compartment,</p>
    <p><strong>α<sub>v</sub>:</strong> rate of immunity loss after the vaccination,</p>
    <p><strong>α<sub>v_failed</sub>:</strong> rate at which individuals leave the Failed vaccination compartment - enables them to revaccinate,</p>
    <p><strong>v<sub>rate</sub>:</strong> vaccination rate,</p>
    <p><strong>v<sub>success</sub>:</strong> rate of vaccine efficacy,</p>
    <p><strong>δ:</strong> mortality rate.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="header">
        <h3>Schéma du modèle :</h3>  
        <div style="margin-bottom: -70px;"></div>
    </div>          
    """, unsafe_allow_html=True)
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
            VF [label="V_failed\nVaccination échouée", fillcolor="#FFA07A"];
            S -> E [label="β"];
            E -> I [label="σ"];
            I -> R [label="γ"];
            I -> D [label="δ"];
            R -> S [label="α"];
            S -> V [label="v_rate · v_success"];
            S -> VF [label="v_rate · (1 - v_success)"];
            V -> S [label="α_v"];
            VF -> S [label="α_v_failed"];
            VF -> E [label="β"];
        }
    """)
    st.markdown("""
    <div class="header">
        <h3>Simulation du modèle :</h3>  
    </div>
    <div style="margin-bottom: -50px;"></div>
    <div>
        <p class = "paragraph">
        Le modèle choisi est un modèle où l'on modélise les individus vaccinés par deux nouveaux compartiments, en fonction de l'efficacité du vaccin : le vaccin est soit efficace (V) ou inefficace (V_failed). Cependant, dans tous les cas l'imumunité n'est pas définitive : il est toujours possible de retomber malade (donc de retourner dans le compartiment S).
            </p>
    </div>        
    """, unsafe_allow_html=True)

    if 'stored_plots_v' not in st.session_state:
        st.session_state.stored_plots_v = []

    col1, col2 = st.columns(2)

    st.write("**Conditions de la simulation :**")
    noise_std = st.checkbox('Include Noise Standard Deviation', key='noise_std_vaccination')
    days = st.slider('Number of Days', min_value=1, max_value=1825, value=365, step=10, key='days_vaccination')
    population_size = st.slider('Population Size', min_value=50, max_value=10000, value=1000, step=100, key='pop_vaccination')
    seasonal_amplitude = False  

    st.write("**Paramètres de la simulation :**")
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        prob_of_infecting = st.slider('Probability of Infecting', min_value=0.0, max_value=1.0, value=0.3, step=0.1, key='prob_infecting_vaccination')
        avg_no_contacts_per_individual = st.slider('Average Number of Contacts per Individual', min_value=1,max_value=100, value=12, step=1, key='contacts_vaccination')
        beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
        sigma = st.slider('Infection Rate (σ)', min_value=0.0, max_value=1.0, value=0.14, step=0.01, key='sigma_vaccination')
        gamma = st.slider('Recovery Rate (γ)', min_value=0.0, max_value=1.0, value=1/21, step=0.01, key='gamma_vaccination')
        delta = st.slider('Mortality Rate (δ)', min_value=0.0, max_value=1.0, value=0.03, format="%.3f", step=0.001, key='delta_vaccination')
    with param_col2:
        alpha = st.slider('Rate at which individuals lose their immunity (α)', min_value=0.0, max_value=1.0, value=0.0055, format="%.3f", step=0.001, key='alpha_vaccination')
        v_rate = st.slider('Vaccination Rate (v_rate)', min_value=0.0, max_value=1.0, value=0.14, step=0.01, key='v_rate_vaccination')
        v_success = st.slider('Vaccination success Rate (v_success)', min_value=0.0, max_value=1.0, value=0.14, step=0.01, key='v_success_vaccination')
        alpha_v = st.slider('Rate at which individuals lose their immunity (acquired from the vaccine) (α_v)', min_value=0.0, max_value=1.0, value=0.14, step=0.01, key='alpha_v_vaccination')
        alpha_v_failed = st.slider('Rate at which individuals lose their immunity (acquired from the failed vaccine) (α_v_failed)', min_value=0.0, max_value=1.0, value=0.14, step=0.01, key='alpha_v_failed_vaccination')

    params = {
        'beta': beta,
        'sigma': sigma,
        'gamma': gamma,
        'alpha': alpha,
        'delta': delta,
        'v_rate': v_rate,
        'v_success': v_success,
        'alpha_v': alpha_v,
        'alpha_v_failed': alpha_v_failed
    }

    df, result = generate_sir_model_data(population_size, params, days, seasonal_amplitude=seasonal_amplitude, noise_std=noise_std)
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
    
#%% Page conclusion et discussion
elif page == "Conclusion et discussion":
    st.subheader(" À propos")
    st.write("Application avec design personnalisé")
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
