import spidev
import time
import requests
import numpy as np
import joblib
import json
import math

try:
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1350000
except (FileNotFoundError, IOError):
    print("SPI device not found. Mocking SPI for testing purposes.")
    class MockSPI:
        def xfer2(self, data):
            return [0, 0, 0]
        def close(self):
            pass
    spi = MockSPI()

def read_channel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def convert_to_voltage(adc_value, vref=3.3, resolution=1023):
    return (adc_value * vref) / resolution

def voltage_to_ph(voltage):
    return (voltage / 3.3) * 14

def voltage_to_nutrient(voltage, nutrient='nitrogen'):
    if nutrient == 'nitrogen':
        return (voltage / 3.3) * 200
    else:
        return (voltage / 3.3) * 100

def get_air_quality(api_token):
    url = f"http://api.waqi.info/feed/here/?token={api_token}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                station_data = data.get("data", {})
                aqi = station_data.get("aqi")
                iaqi = station_data.get("iaqi", {})
                pm25 = iaqi.get("pm25", {}).get("v")
                pm10 = iaqi.get("pm10", {}).get("v")
                no2 = iaqi.get("no2", {}).get("v")
                co = iaqi.get("co", {}).get("v")
                temp = iaqi.get("t", {}).get("v")
                return {
                    'AQI': aqi,
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'NO2': no2,
                    'CO': co,
                    'Temp': temp
                }
            else:
                print("API error:", data.get("message"))
        else:
            print("HTTP error:", response.status_code)
    except Exception as e:
        print("Error fetching air quality data:", e)
    return None

ph_adc = read_channel(0)
ph_voltage = convert_to_voltage(ph_adc)
ph_value = voltage_to_ph(ph_voltage)

nitrogen_adc = read_channel(1)
phosphorus_adc = read_channel(2)
potassium_adc = read_channel(3)
nitrogen_voltage = convert_to_voltage(nitrogen_adc)
phosphorus_voltage = convert_to_voltage(phosphorus_adc)
potassium_voltage = convert_to_voltage(potassium_adc)
nitrogen_value = voltage_to_nutrient(nitrogen_voltage, 'nitrogen')
phosphorus_value = voltage_to_nutrient(phosphorus_voltage, 'phosphorus')
potassium_value = voltage_to_nutrient(potassium_voltage, 'potassium')

plot_area = 500
plot_area_acre = plot_area / 4046.86

api_token = "be435989d42640ef15e2cb9c6281fdf2539f0b9e"
air_data = get_air_quality(api_token)

if air_data is None:
    air_data = {
        'AQI': 150,
        'PM2.5': 80,
        'PM10': 150,
        'NO2': 25,
        'CO': 0.8,
        'Temp': 32
    }

feature_vector = np.array([[
    air_data['AQI'],
    air_data['PM2.5'],
    air_data['PM10'],
    air_data['NO2'],
    air_data['CO'],
    air_data['Temp'],
    ph_value,
    nitrogen_value,
    phosphorus_value,
    potassium_value,
    plot_area
]])

print("Feature Vector:", feature_vector)

model = joblib.load('knn_plant_identifier.pkl')
scaler_model = joblib.load('scaler_knn.pkl')
label_encoder = joblib.load('label_encoder_knn.pkl')

feature_vector_scaled = scaler_model.transform(feature_vector)

pred = model.predict(feature_vector_scaled)
predicted_plant = label_encoder.inverse_transform(pred)[0]

with open('rec.json', 'r') as json_file:
    plant_data = json.load(json_file)

plant_details = plant_data.get("plant_impact_timeline", {}).get(predicted_plant, {})
recommended_density = plant_details.get('recommended_plant_density', 100)
lower_limit = math.floor(plot_area_acre * int(recommended_density.split('-')[0]))
upper_limit = math.ceil(plot_area_acre * int(recommended_density.split('-')[1]))
num_trees = "{:.0f} - {:.0f}".format(lower_limit, upper_limit)

formatted_output = (
    f"The recommended plant for the given environmental conditions is {predicted_plant}. "
    f"This selection is based on current air quality parameters such as AQI, PM2.5, NO2 levels, "
    f"and soil nutrients including nitrogen, phosphorus, and potassium. "
    f"By planting {predicted_plant}, air quality is expected to improve within {plant_details.get('time_to_improve_air_quality', 'an estimated timeframe')}. "
    f"This plant will also have the following impact on the soil: {plant_details.get('impact_on_soil', 'unknown effects')}. "
    f"For best results, it should be planted at a density of {recommended_density} trees per acre. "
    f"Given the available plot area of {plot_area} square meters ({plot_area_acre:.2f} acres), approximately {num_trees} trees can be planted. "
    f"Proper maintenance includes {plant_details.get('maintenance', 'general plant care')}. "
    f"To ensure optimal growth and continued benefits, soil testing should be performed every {plant_details.get('recommended_testing_frequency', 'regular intervals')}. "
    f"This plant thrives best in {plant_details.get('optimal_climate', 'a suitable climate')}. "
    f"Additionally, it has a carbon sequestration potential of {plant_details.get('carbon_sequestration_potential', 'estimated values')}, contributing positively to environmental sustainability."
)

print(formatted_output)

spi.close()
