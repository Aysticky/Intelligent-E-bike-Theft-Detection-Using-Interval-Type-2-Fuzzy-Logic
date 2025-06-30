# Code developed by Meadows and Saman
# For partneship project with Bouwtag and Bypoint
# Hanzhi University of Applied Sciences

import numpy as np
import pandas as pd
from fuzzy_logic import (
    fls, print_input_memberships,
    accel_universe, pedal_still_universe, motion_universe,
    tilt_universe, z_orientation_universe
)

def sliding_window_average(data, window_sec=1.0, sample_rate=25):
    window_size = int(window_sec * sample_rate)
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        avg_x = np.mean([d["freeAccx"] for d in window])
        avg_y = np.mean([d["freeAccy"] for d in window])
        avg_z = np.mean([d["freeAccz"] for d in window])
        center_time = window[window_size // 2]["timestamp"]
        result.append({
            "freeAccx": avg_x,
            "freeAccy": avg_y,
            "freeAccz": avg_z,
            "timestamp": center_time
        })
    return result

def compute_latest_pedal_stillness(data, window_sec=2.0, sample_rate=25):
    window_size = int(window_sec * sample_rate)
    if len(data) < window_size:
        raise ValueError("Not enough data for pedal stillness window")
    window = data[-window_size:]
    x_vals = np.array([d["freeAccx"] for d in window])
    y_vals = np.array([d["freeAccy"] for d in window])
    signal = x_vals + y_vals

    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.7/(sample_rate/2), btype='high')
    filtered_signal = filtfilt(b, a, signal)

    mad = np.mean(np.abs(np.diff(filtered_signal)))
    print(f"Pedal MAD: {mad:.4f}")

    return {
        "pedal_stillness": mad,
        "freeAccx": np.mean(x_vals),
        "freeAccy": np.mean(y_vals),
        "freeAccz": np.mean([d["freeAccz"] for d in window])
    }

def compute_latest_car_motion(data, window_sec=3.0, sample_rate=25):
    window_size = int(window_sec * sample_rate)
    if len(data) < window_size:
        raise ValueError("Not enough data for car motion window")
    window = data[-window_size:]
    x_vals = [d["freeAccx"] for d in window]
    y_vals = [d["freeAccy"] for d in window]
    mad_x = np.mean(np.abs(np.diff(x_vals)))
    mad_y = np.mean(np.abs(np.diff(y_vals)))
    car_motion = (mad_x + mad_y) / 2
    return {
        "car_motion": car_motion,
        "freeAccx": np.mean(x_vals),
        "freeAccy": np.mean(y_vals),
        "freeAccz": np.mean([d["freeAccz"] for d in window])
    }

def compute_tilt(accel_data):
    return float(np.clip(max(abs(accel_data["freeAccx"]), abs(accel_data["freeAccy"])), tilt_universe[0], tilt_universe[-1]))

def get_lift_accel_from_window(window):
    return np.max([abs(d["freeAccz"]) for d in window])

def detect_theft_it2(accel_data, recent_data=None):
    if recent_data is not None:
        z = float(np.clip(get_lift_accel_from_window(recent_data), accel_universe[0], accel_universe[-1]))
    else:
        z = float(np.clip(abs(accel_data["freeAccz"]), accel_universe[0], accel_universe[-1]))
    pedal_stillness = float(np.clip(accel_data["pedal_stillness"], pedal_still_universe[0], pedal_still_universe[-1]))
    car_motion = float(np.clip(accel_data["car_motion"], motion_universe[0], motion_universe[-1]))
    tilt = compute_tilt(accel_data)
    z_orientation = float(np.clip(accel_data["freeAccz"], z_orientation_universe[0], z_orientation_universe[-1]))

    print(f"\nInput Values - Z: {z:.2f}g, PedalStillness: {pedal_stillness:.4f}, CarMotion: {car_motion:.2f}, Tilt: {tilt:.2f}, z_orientation: {z_orientation:.2f}")

    inputs = {
        "lift_accel": z,
        "pedal_stillness": pedal_stillness,
        "car_motion": car_motion,
        "tilt": tilt,
        "z_orientation": z_orientation
    }
    print_input_memberships(inputs)
    fired_outputs = fls.evaluate(inputs)

    centroid_lower, centroid_upper, centroid = fls.defuzzify(fired_outputs)
    print(f"Risk interval centroid: lower={centroid_lower:.2f}, upper={centroid_upper:.2f}")
    print(f"Defuzzified Risk: {centroid:.2f}%")
    if centroid >= 60:
        print("THEFT DETECTED")
    else:
        print("No theft")
    return centroid

def load_real_data(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    df = df.rename(columns={
        'ax': 'freeAccx',
        'ay': 'freeAccy',
        'az': 'freeAccz',
        'time': 'timestamp'
    })
    df = df.dropna(subset=['freeAccx', 'freeAccy', 'freeAccz'])
    if 'timestamp' not in df.columns:
        raise ValueError("Could not find 'timestamp' column after renaming. Check your CSV header.")
    if not np.issubdtype(df['timestamp'].dtype, np.number):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    data = df[['freeAccx', 'freeAccy', 'freeAccz', 'timestamp']].to_dict(orient='records')
    return data

if __name__ == "__main__":
    filepath = r"C:\Users\User\Desktop\Data with car\real data\data\Lower rack front tire up.csv"
    real_data = load_real_data(filepath)
    pedal_stillness_feat = compute_latest_pedal_stillness(real_data, window_sec=2.0, sample_rate=25)
    car_motion_feat = compute_latest_car_motion(real_data, window_sec=3.0, sample_rate=25)
    test_sample = {**pedal_stillness_feat, **car_motion_feat}
    detect_theft_it2(test_sample, recent_data=real_data)
