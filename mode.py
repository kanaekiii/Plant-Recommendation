import spidev
import serial
import time

# Setup SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI device 0 (CE0)
spi.max_speed_hz = 50000  # Set the speed for SPI

# Setup Serial for RS-485 (NPK sensor)
ser = serial.Serial('/dev/serial0', 9600, timeout=1)  # Adjust serial port and baud rate
ser.flush()  # Flush any previous data in the buffer

# Function to read from MCP3008 (used for pH and soil moisture sensors)
def read_adc(channel):
    assert 0 <= channel <= 7, 'Channel must be between 0 and 7'
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

# Function to read NPK data from RS-485
def read_npk_data():
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            return data
    except Exception as e:
        print(f"Error reading NPK data: {e}")
        return None

# Function to calculate pH from raw ADC value (calibration required)
def calculate_ph(raw_adc_value):
    voltage = (raw_adc_value / 1023.0) * 3.3  # Convert ADC to voltage
    # Example calibration: (This needs to be adjusted based on your pH sensor calibration)
    ph_value = 10 - (voltage * 14 / 3.3) # Example calibration, adjust for your sensor
    return ph_value

# Function to process soil moisture data
def calculate_soil_moisture(raw_adc_value):
    # Example conversion (you may need to calibrate this depending on your sensor)
    moisture_value = (raw_adc_value / 1023.0) * 100  # Scale to percentage (0-100)
    return moisture_value

# Main loop
while True:
    # Read pH sensor (connected to channel 0 of MCP3008)
    pH_raw = read_adc(0)  # Read from pH sensor
    pH_value = calculate_ph(pH_raw)

    # Read soil moisture sensor (connected to channel 1 of MCP3008)
    soil_moisture_raw = read_adc(1)  # Read from soil moisture sensor
    soil_moisture_value = calculate_soil_moisture(soil_moisture_raw)

    # Read NPK sensor (RS-485 communication)
    npk_data = read_npk_data()

    # Print the readings
    print(f"pH Sensor: Raw ADC: {pH_raw}, pH Value: {pH_value:.2f}")
    print(f"Soil Moisture Sensor: Raw ADC: {soil_moisture_raw}, Moisture Value: {soil_moisture_value:.2f}%")

    #if npk_data:
    print(f"NPK Sensor Data: {npk_data}")  # Print NPK sensor data received over RS-485

    # Sleep for 1 second before the next reading
    time.sleep(1)