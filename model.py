import spidev
import time
import requests
import numpy as np
import joblib

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

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

def get_air_quality(api_token, city):
    url = f"http://api.waqi.info/feed/{city}/?token={api_token}"
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

api_token = "be435989d42640ef15e2cb9c6281fdf2539f0b9e"
city = "bangalore"  
air_data = get_air_quality(api_token, city)

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
predicted_plant = label_encoder.inverse_transform(pred)

print("Recommended Plant Type:", predicted_plant[0])

spi.close()