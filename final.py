import spidev
import time
import requests
import numpy as np
import joblib
import json
import math
import serial
import csv
import os
import logging
# from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
# load_dotenv()

CSV_FILE = "sensor_data.csv"
PLOT_AREA = 500
PLOT_AREA_ACRE = PLOT_AREA / 4046.86
API_TOKEN = os.getenv("WAQI_API_TOKEN", "be435989d42640ef15e2cb9c6281fdf2539f0b9e")

class MockSPI:
    def xfer2(self, data): return [0, 0, 0]
    def close(self): pass

class MockSerial:
    def __init__(self): self.in_waiting = 0
    def readline(self): return b""

def init_spi():
    try:
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = 1350000
        return spi
    except (FileNotFoundError, IOError):
        logging.warning("SPI device not found. Mocking SPI for testing purposes.")
        return MockSPI()

def init_serial():
    try:
        ser = serial.Serial('/dev/serial0', 9600, timeout=1)
        ser.flush()
        return ser
    except Exception:
        logging.warning("Serial port not available, using mock serial")
        return MockSerial()

def read_channel(spi, channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

def convert_to_voltage(adc_value, vref=3.3, resolution=1023):
    return (adc_value * vref) / resolution

def voltage_to_nutrient(voltage, nutrient='nitrogen'):
    return (voltage / 3.3) * (200 if nutrient == 'nitrogen' else 100)

def calculate_ph(raw_adc_value):
    return 10 - ((raw_adc_value / 1023.0) * 3.3 * 14 / 3.3)

def read_npk_data(ser):
    try:
        if ser.in_waiting > 0:
            return ser.readline().decode('utf-8').strip()
    except Exception as e:
        logging.error("Error reading NPK data: %s", e)
    return None

def get_air_quality(api_token):
    try:
        response = requests.get(f"http://api.waqi.info/feed/here/?token={api_token}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                iaqi = data.get("data", {}).get("iaqi", {})
                return {
                    'AQI': data['data'].get('aqi'),
                    'PM2.5': iaqi.get("pm25", {}).get("v"),
                    'PM10': iaqi.get("pm10", {}).get("v"),
                    'NO2': iaqi.get("no2", {}).get("v"),
                    'CO': iaqi.get("co", {}).get("v"),
                    'Temp': iaqi.get("t", {}).get("v")
                }
            logging.warning("API error: %s", data.get("message"))
        else:
            logging.warning("HTTP error: %s", response.status_code)
    except Exception as e:
        logging.error("Error fetching air quality data: %s", e)
    return {
        'AQI': 150, 'PM2.5': 80, 'PM10': 150, 'NO2': 25, 'CO': 0.8, 'Temp': 32
    }

def collect_data(spi):
    logging.info("Collecting data for 2 minutes...")
    start_time = time.time()
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ph', 'nitrogen', 'phosphorus', 'potassium'])
        while time.time() - start_time < 120:
            try:
                ph_value = calculate_ph(read_channel(spi, 0))
                nitrogen = voltage_to_nutrient(convert_to_voltage(read_channel(spi, 1)), 'nitrogen')
                phosphorus = voltage_to_nutrient(convert_to_voltage(read_channel(spi, 2)), 'phosphorus')
                potassium = voltage_to_nutrient(convert_to_voltage(read_channel(spi, 3)), 'potassium')
                writer.writerow([time.time(), ph_value, nitrogen, phosphorus, potassium])
            except Exception as e:
                logging.error("Sensor error: %s", e)
            time.sleep(5)

def analyze_data():
    data = np.genfromtxt(CSV_FILE, delimiter=',', skip_header=1)
    return np.mean(data[:, 1]), np.mean(data[:, 2]), np.mean(data[:, 3]), np.mean(data[:, 4])

def predict_plant(features):
    model = joblib.load('knn_plant_identifier.pkl')
    scaler = joblib.load('scaler_knn.pkl')
    encoder = joblib.load('label_encoder_knn.pkl')
    scaled = scaler.transform(features)
    return encoder.inverse_transform(model.predict(scaled))[0]

def generate_output(predicted_plant):
    try:
        with open('rec.json', 'r') as f:
            plant_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error("Error loading plant data: %s", e)
        return "Plant data unavailable."

    details = plant_data.get("plant_impact_timeline", {}).get(predicted_plant, {})
    density = details.get('recommended_plant_density', '100-150')
    lower, upper = map(int, density.split('-'))
    lower_count = math.floor(PLOT_AREA_ACRE * lower)
    upper_count = math.ceil(PLOT_AREA_ACRE * upper)

    return (
        f"The recommended plant is {predicted_plant}. It improves air quality in {details.get('time_to_improve_air_quality', 'an estimated timeframe')}, "
        f"benefits soil by {details.get('impact_on_soil', 'unknown effects')}, and should be planted at {density} trees/acre. "
        f"For a {PLOT_AREA} m² area ({PLOT_AREA_ACRE:.2f} acres), plant approximately {lower_count} - {upper_count} trees. "
        f"Maintain with {details.get('maintenance', 'general plant care')}, test soil every {details.get('recommended_testing_frequency', 'regular intervals')}, "
        f"and note it thrives in {details.get('optimal_climate', 'a suitable climate')}. Carbon sequestration potential: {details.get('carbon_sequestration_potential', 'estimated values')}.")

if __name__ == "__main__":
    spi = init_spi()
    ser = init_serial()
    collect_data(spi)
    mean_ph, mean_n, mean_p, mean_k = analyze_data()
    air = get_air_quality(API_TOKEN)

    feature_vector = np.array([[
        air['AQI'], air['PM2.5'], air['PM10'], air['NO2'], air['CO'], air['Temp'],
        mean_ph, mean_n, mean_p, mean_k, PLOT_AREA
    ]])

    logging.info("Feature Vector: %s", feature_vector)
    prediction = predict_plant(feature_vector)
    logging.info(generate_output(prediction))
    spi.close()
