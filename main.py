import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Buffer:
    """
    Class for storing and managing rays (light paths) in the simulation.
    Acts like a queue for photon rays.
    """
    
    def __init__(self):
        self.storage = []

    def push(self, ray):
        self.storage.append(ray)

    def pop(self):
        if self.storage:
            return self.storage.pop(0)
        return None

    def __len__(self):
        return len(self.storage)

    def is_empty(self):
        return len(self.storage) == 0

    def to_array(self):
        return np.array([(r.power, r.theta) for r in self.storage])

class Ray:
    """
    Represents a light ray in the UWOC system.
    Stores position, direction, power, and angular properties.
    """

    def __init__(self, position, direction, power, is_direct, theta=0, phi=0):
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float) / np.linalg.norm(direction)
        self.power = power
        self.is_direct = is_direct
        self.theta = theta
        self.phi = phi

    def get_theta(self, orientation):
        dot = np.dot(self.direction, orientation) / (
            np.linalg.norm(self.direction) * np.linalg.norm(orientation))
        return np.arccos(np.clip(dot, -1.0, 1.0))

    def generate_random_direction(self):
        phi = 2 * np.pi * np.random.rand()
        costheta = 2 * np.random.rand() - 1
        sintheta = np.sqrt(1 - costheta**2)
        return np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta], dtype=float)

def output_power(model, params, theta):
    """
    Calculates emitted power based on transmitter model (Lambertian/Gaussian).
    """
    
    if model == 'lambertian':
        m = params.get('m', 1)
        return (m + 1) / (2 * np.pi) * np.cos(theta)**m
    elif model == 'gaussian':
        sigma = params.get('sigma', 0.3)
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-theta**2 / (2 * sigma**2))
    else:
        raise ValueError("Unknown output power model")

def compute_intensity(I0, c_lambda, distance):
    """
    Computes the received light intensity using Beer-Lambert law.
    """
    """
    Hitung intensitas I berdasarkan hukum Beer-Lambert:
    I = I0 * exp(-c(lambda) * d)
    """
    
    return I0 * np.exp(-c_lambda * distance)

def run_monte_carlo(tx, rx, scenario):
    """
    Monte Carlo simulation of light propagation between transmitter and receiver.
    It models both direct and scattered rays with exponential attenuation.
    """
    
    impulse_response = Buffer()
    buffer = Buffer()

    direction = np.array(rx['position'], dtype=float) - np.array(tx['position'], dtype=float)
    # Normalize direction vector
    direction /= np.linalg.norm(direction)
    
    ray = Ray(tx['position'], direction, 1.0, True)
    ray.theta = ray.get_theta(tx['orientation'])
    # Compute travel distance from tx to rx
    ray.path_length = np.linalg.norm(np.array(rx['position']) - np.array(tx['position']))
    # Compute travel distance from tx to rx
    ray.arrival_time = ray.path_length / (3e8 / 1.33)  # speed of light in water
    # Calculate initial power using transmitter model
    ray.power *= output_power(tx['type'], tx['params'], ray.theta)
    buffer.push(ray)

    for _ in range(scenario['N']):
        ray = Ray(tx['position'], tx['orientation'], 1.0 / scenario['N'], False)
        ray.direction = ray.generate_random_direction()
        ray.theta = ray.get_theta(tx['orientation'])
        # Compute travel distance from tx to rx
        ray.path_length = np.linalg.norm(np.array(rx['position']) - np.array(tx['position']))
        # Compute travel distance from tx to rx
        ray.arrival_time = ray.path_length / (3e8 / 1.33)
        alpha = 0.15  # default attenuation coefficient
        # Compute travel distance from tx to rx
        ray.power *= np.exp(-alpha * ray.path_length)
        buffer.push(ray)

    while not buffer.is_empty():
        current_ray = buffer.pop()
        impulse_response.push(current_ray)

    return impulse_response

def save_impulse_response_to_csv(impulse_response, filename="impulse_response_full.csv"):
    """
    Saves the simulation results (impulse response) to a CSV file.
    """
    
    data = [{
        'Power': ray.power,
        'Theta (deg)': np.degrees(ray.theta),
        'Path Length (m)': ray.path_length,
        'Arrival Time (ns)': ray.arrival_time * 1e9
    } for ray in impulse_response.storage]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def visualize_impulse_response(impulse_response):
    """
    Visualizes the impulse response using a histogram of arrival angles.
    """
    
    data = impulse_response.to_array()
    if len(data) == 0:
        print("No rays received.")
        return
    power, theta = data[:, 0], data[:, 1]
    plt.figure(figsize=(8, 5))
    plt.hist(theta * 180 / np.pi, weights=power, bins=30, color='skyblue', edgecolor='k')
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Power (a.u.)")
    plt.title("Impulse Response (Angular Distribution)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_params_from_depth(depth):
    T = 25 - 0.2 * depth
    S = 35 + 0.01 * depth
    return T, S

def get_refractive_index(wavelength, method='McNeil', depth=15):
    """
    Returns refractive index based on depth and wavelength using empirical models.
    """
    
    T, S = get_params_from_depth(depth)
    if method.lower() == 'matthaus':
        L = wavelength / 1000.0
        n = (
            1.447824 + 3.011e-4 * S - 1.8029e-5 * T - 1.6916e-6 * T**2
            - 0.489 * L + 0.728 * L**2 - 0.384 * L**3
            - S * (
                7.9362e-7 * T - 8.06e-9 * T**2 + 4.249e-4 * L
                - 5.847e-4 * L**2 + 2.812e-4 * L**3
            )
        )
    else:
        L = wavelength
        n = (
            1.3247 - 2.5e-6 * T**2 + S * (2e-4 - 8e-7 * T)
            + 3300 / (L**2) - 3.2e7 / (L**4)
        )
    return n

def get_absorption_from_wavelength(wavelength, chlorophyll=0.01, gelbstoff=0.01, minerals=0.01):
    """
    Calculates the absorption coefficient alpha based on wavelength and water contents.
    """
    
    water_data = pd.read_csv("data/Pope_Fry_measurements.csv", sep='\t')
    water = 100 * np.interp(wavelength, water_data["Wavelength"], water_data["Absorption"])
    chl_data = pd.read_excel("data/Bricaud_et_al_2004.xlsx")
    ChlA = chlorophyll * np.interp(wavelength, chl_data["lambda"], chl_data["ChlA"])
    ChlB = chlorophyll * np.interp(wavelength, chl_data["lambda"], chl_data["ChlB"])
    phytoplankton = np.mean([ChlA, ChlB])
    decaying = np.exp(-0.0139 * (wavelength - 400))
    inorganic = np.exp(-0.0069 * (wavelength - 400))
    alpha = water + phytoplankton + gelbstoff * decaying + minerals * inorganic
    return alpha

def simulate_attenuation_vs_distance(d_values, alpha, N=1000, save_path="attenuation_vs_distance.csv"):
    """
    Simulate attenuation over various distances using exponential model.
    Save results to CSV and return received power list.
    """

    results = []
    for d in d_values:
        powers = [np.exp(-alpha * d) for _ in range(N)]
        avg_power = np.mean(powers)
        results.append((d, avg_power))

    # Simpan hasil ke CSV
    df = pd.DataFrame(results, columns=["Distance (m)", "Received Power"])
    df.to_csv(save_path, index=False)
    return [r[1] for r in results]

def plot_attenuation_vs_distance(csv_path):
    """
    Plot attenuation vs distance from CSV file.
    """
 
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Distance (m)"], df["Received Power"], marker='o')
    plt.title("Attenuation vs Transmission Distance (Monte Carlo Approximation)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Received Power (a.u.)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("attenuation_vs_distance_plot.png")
    plt.show()

if __name__ == "__main__":
    scenario = {
        'N': 5000,
        'info_period': 500
    }
    tx = {
        'position': [0, 0, 0],
        'orientation': [0, 0, 1],
        'type': 'lambertian',
        'params': {'m': 1},
        'output_power': lambda theta: output_power('lambertian', {'m': 1}, theta)
    }
    rx = {
        'position': [0, 0, 5]
    }
    wavelength = 660
    depth = 15
    refractive_index = get_refractive_index(wavelength=wavelength, method='McNeil', depth=depth)
    absorption = get_absorption_from_wavelength(wavelength=wavelength)

    # Hitung intensitas berdasarkan jarak propagasi
    I0 = 1.0
    distance = np.linalg.norm(np.array(rx['position']) - np.array(tx['position']))
    received_intensity = compute_intensity(I0, absorption, distance)

    # Log information to console
    print(f"Intensitas diterima pada jarak {distance:.2f} m: {received_intensity:.5f} W/m^2")

    # Log information to console
    print("Running UWOC simulation...")
    impulse_response = run_monte_carlo(tx, rx, scenario)
    
    # Simpan ke CSV
    # Log information to console
    print("Save UWOC simulation...")
    save_impulse_response_to_csv(impulse_response, "impulse_response.csv")

    # Log information to console
    print("Visualizing impulse response...")
    visualize_impulse_response(impulse_response)

    # Simulasi dan visualisasi attenuation vs distance
    distances = np.linspace(1, 50, 50)
    alpha = 0.15
    simulate_attenuation_vs_distance(distances, alpha, N=1000, save_path="attenuation_vs_distance.csv")
    plot_attenuation_vs_distance("attenuation_vs_distance.csv")

