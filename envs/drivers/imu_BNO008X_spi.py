# Author : Geoffrey ISHIMARU - MA1 ELN - ISIB
# References : [1] BNO080 Datasheet from Hillcrest labs, [2] Sparkfun_BNO080.cpp updated by Guillaume Villee, [3] Sparkfun_BNO080.h, [4] Reference Manual from Hillcrestlabs
#              [5] IMUManager.cpp modified by Guillaume Villee

import RPi.GPIO as GPIO
import time
import spidev
import os
from periphery import SPI

# -------------------------------------------------------------------
# ---------------------- VARIABLES TO STORE -------------------------
# -------------------------------------------------------------------

_cs = None
_wake = None
_int = None
_rst = None

# - Connexion
spi = None
MAX_PACKET_SIZE=128


# - Debug
debug_print = True
timeStamp = 0  #32-bit value
firstPackReceived = False
end_prog = False
# -- Debug record file
file = None
record_file = False
# --- to always write the Record File in the same dir as the script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_path = os.path.join(__location__, "DataTest.csv")

# - Low Level Communication with sensor
SHTP_REPORT_COMMAND_RESPONSE = 0xF1
SHTP_REPORT_COMMAND_REQUEST = 0xF2
SHTP_REPORT_PRODUCT_ID_REQUEST = 0xF9  # [3]L59
SHTP_REPORT_PRODUCT_ID_RESPONSE = 0xF8  # [3]L58
SHTP_REPORT_BASE_TIMESTAMP = 0xFB  # [3]L60
SHTP_REPORT_FRS_READ_REQUEST = 0xF4  # [3]L57
SHTP_REPORT_FRS_READ_RESPONSE = 0xF3  # [3]L56
SHTP_REPORT_SET_FEATURE_COMMAND = 0xFD  # [3]L61

SENSOR_REPORTID_ACCELEROMETER = 0x01  # [3] beginning from L65 and for functionparseInputReport at L212
SENSOR_REPORTID_GYROSCOPE = 0x02
SENSOR_REPORTID_MAGNETIC_FIELD = 0x03
SENSOR_REPORTID_LINEAR_ACCELERATION = 0x04
SENSOR_REPORTID_ROTATION_VECTOR = 0x05
SENSOR_REPORTID_GRAVITY = 0x06
SENSOR_REPORTID_GAME_ROTATION_VECTOR = 0x08
SENSOR_REPORTID_GEOMAGNETIC_ROTATION_VECTOR = 0x09
SENSOR_REPORTID_STEP_COUNTER = 0x11
SENSOR_REPORTID_STABILITY_CLASSIFIER = 0x13
SENSOR_REPORTID_PERSONAL_ACTIVITY_CLASSIFIER = 0x1E

COMMAND_DCD = 6  # [3] L91
COMMAND_ME_CALIBRATE = 7  # [3] L92

CALIBRATE_ACCEL = 0  # [3] from L97
CALIBRATE_GYRO = 1
CALIBRATE_MAG = 2
CALIBRATE_PLANAR_ACCEL = 3
CALIBRATE_ACCEL_GYRO_MAG = 4
CALIBRATE_STOP = 5

channel = ["Command", "Executable", "Control", "Sensor-report", "Wake-report", "Gyro-vector"]  # Meaning got from [2] L1083

# - Communication Data Storage
shtpHeader = []
shtpData = []
dataLength = 0  # length of shtpData
seqNbr = [0, 0, 0, 0, 0, 0]  # There are 6 comm channels. Each channel has its own seqNum [3]L198

# - Raw sensor values
rawAccelX, rawAccelY, rawAccelZ, accelAccuracy = None, None, None, None  # [3] L214
rawLinAccelX, rawLinAccelY, rawLinAccelZ, accelLinAccuracy = None, None, None, None
rawGyroX, rawGyroY, rawGyroZ, gyroAccuracy = None, None, None, None
rawMagX, rawMagY, rawMagZ, magAccuracy = None, None, None, None
rawQuatI, rawQuatJ, rawQuatK, rawQuatReal, rawQuatRadianAccuracy, quatAccuracy = None, None, None, None, None, None
stepCount = None
timeStamp = None
stabilityClassifier = None
activityClassifier = None

_activityConfidence = []  # Array that store the confidences of the 9 possible activities
for i in range(9):
    _activityConfidence.append(None)

rotationVector_Q1 = 14  # [3]L227 These Q values are defined in the datasheet but can also be obtained by querying the meta data records
accelerometer_Q1 = 8
linear_accelerometer_Q1 = 8
gyro_Q1 = 9
magnetometer_Q1 = 4

# - Calibration
calibrationStatus = None  # Byte R0 of ME Calibration Response
calibrationSavedStatus = None  # Check if save confirmed [4] 6.4.6.2
isCalibrating = False
calibrated = False
calPrecisionRate = 0
commandSequenceNumber = 0


# -------------------------------------------------------------------
# --------------------------- DEBUG ---------------------------------
# -------------------------------------------------------------------

def enableDebugging():
    global debug_print
    debug_print = True
    
    
def enableDebugRecordFile():
    global record_file
    record_file = True
    

# -------------------------------------------------------------------
# --------------------- COMMUNICATION INIT --------------------------
# -------------------------------------------------------------------

def beginSPI(user_bcm_CSPin, user_bcm_WAKPin, user_bcm_INTPin, user_bcm_RSTPin, user_spiPortSpeed, user_spiPort):  # [2]L33
    # We want the global variable pins declared above
    global _cs, _wake, _int, _rst, spi, shtpData

    # Get user settings
    if debug_print: print("[BNO080] Setting up SPI communication")

    _spiPort = user_spiPort
    _spiPortSpeed = user_spiPortSpeed  # up to 3MHz allowed by BNO but RPi offers 3.9MHz or 1.953MHz (because of clock divider values)

    if _spiPortSpeed > 3000000:
        _spiPortSpeed = 3000000  # BNO080 max SPI freq is 3MHz

    _cs = 37
    _wake = 31
    _int = 33
    _rst = 35

    if debug_print: print("[BNO080] Setting up RPi Pins")

    # Setting RPi pins
    GPIO.setmode(GPIO.BOARD)  # use BCM numbering (GPIOX)
    GPIO.setup(_cs, GPIO.OUT)
    GPIO.setup(_wake, GPIO.OUT)
    GPIO.setup(_int, GPIO.IN, pull_up_down=GPIO.PUD_UP) # if nothing connected to the input, it will read GPIO.HIGH (pull-up resistor)
    GPIO.setup(_rst, GPIO.OUT)

    # Deselect BNO080
    GPIO.output(_cs, GPIO.HIGH)


    # Config BNO080 for SPI communication
    GPIO.output(_wake, GPIO.HIGH)  # Before boot up the pS0/WAK pin mus must be high to enter SPI Mode
    GPIO.output(_rst, GPIO.LOW)  # Reset BO080
    time.sleep(0.002) # Waits 15ms for the BNO080 to reset -min length not specified in BNO080 Datasheet-
    GPIO.output(_rst, GPIO.HIGH)  # Bring out of reset 

    waitForSPI(_int)

    spi=SPI("/dev/spidev0.0", 3, 1953000)

    # spi=spidev.SpiDev()
    # spi.open(0, 0)  # first param = 0 is the bus (corresponding to the spiPort), second param = 0* is the device
    # spi.mode = 0b11 
    # spi.max_speed_hz = 1953000
    # spi.bits_per_word=8
    # spi.lsbfirst=False
    
    if debug_print: print("[BNO080] Setting SPI communication parameters")

    receiveSPIPacket() 
    receiveSPIPacket()

    shtpData[0] = SHTP_REPORT_PRODUCT_ID_REQUEST  # Request the product ID and reset info
    shtpData[1] = 0                               # Reserved
    if debug_print: print("[BNO080] RPi changing shtpData to send test packet")
    if debug_print: print("[BNO080] Check bytes 0 = 0xF9 and byte 1 = 0 :", shtpData)  # just to check if the global shtpData has been used

    # Transmit packet on channel 2 (= control), 2 bytes [2]L103
    if debug_print: print("[BNO080] Test Packet ready ! Launching sendSPIPacket() method")
    sendSPIPacket(channel.index("Control"), 2)

    waitForSPI(_int)

    if debug_print: print("[BNO080] Response to Test Packet awaited")
    if receiveSPIPacket():
        if debug_print: print("[BNO080] Data received : " + str(shtpData))
        if shtpData[0] == SHTP_REPORT_PRODUCT_ID_RESPONSE:
            if debug_print: print("[BNO080] sent back the CORRECT ID response")
            print("[BNO080] Successfully connected via SPI")
            return True  # Will stop here the function

    print("[BNO080] Something went wrong")
    return False


# -------------------------------------------------------------------
# ------------------- ASK/RECEIVE SPI PACKETS -----------------------
# -------------------------------------------------------------------

def waitForSPI(_int):
    for i in range(125):
        if GPIO.input(_int)==GPIO.LOW:
            return True
        if debug_print:
                print("SPI Wait ")
        time.sleep(0.001)
    if debug_print:
            print("SPI INT timeout")
    return False    

def receiveSPIPacket():
    global shtpData, dataLength, seqNbr  # because we want to use the global shtpData defined in the first lines of this code
    if GPIO.input(_int) == GPIO.HIGH:
        if debug_print: print("[BNO080] Data not available")
        return False
    
    if debug_print: print("[BNO080] Data received!")

    # Select BNO080
    GPIO.output(_cs, GPIO.LOW)

    # Get the first 4 bytes, aka the packet header (in shtp = sensor hub transport protocol from Hillcrest)


    shtpHeader = spi.transfer([0x00,0x00,0x00,0x00])
    # shtpHeader = spi.readbytes(4)

    print(shtpHeader)
    packetLSB = shtpHeader[0]
    packetMSB = shtpHeader[1]
    channelNbr = shtpHeader[2]
    seqNbr[channelNbr] = shtpHeader[3]

    # Calculate the number of data bytes in this packet
    if packetMSB >= 128:
        packetMSB -= 128 # the first bit indicates if this package is continuation of the last. Ignore it for now.
    dataLength = packetMSB*256 + packetLSB # in C++ : ((uint16_t)packetMSB << 8 | packetLSB) --> shift MSB to 8 first bits of (new casted) 16 bits then add 8 LSB.
    if debug_print: print("[BNO080] Data Length (including header) is " + str(dataLength))
    if dataLength == 0:
        GPIO.output(_cs, GPIO.HIGH)        
        return False  # packet empty, done
    # Remove the header bytes from data count --> it is only length of data, not the packet
    dataLength -= 4

    # Read incoming data
    shtpData.clear()
    incoming=spi.transfer([0xFF]*dataLength)
    if dataLength<128:
        shtpData=incoming
        print(shtpData)

    GPIO.output(_cs, GPIO.HIGH)        
    return True

def sendSPIPacket(channelNbr, data_length):  # [2]L1017
    # We want to use global variables to be able to write and save the changes about them
    global seqNbr, shtpData
    # packet length contains header (4 bytes) + data
    packetLength = data_length + 4
    # Wait for BNO080 to indicate (through INTerrupt pin) if it is ready for communication
    if waitForSPI(_int) == False:
        if debug_print: print("[BNO080] not ready to receive data")
        return False

    # Select BNO080
    GPIO.output(_cs, GPIO.LOW)

    # 1. Prepare the 4 bytes packet header
    if debug_print: print("[BNO080] RPi Preparing header to send")
    headerBuffer = [0,0,0,0]
    if packetLength < 256:
        headerBuffer[0] = packetLength # packet length LSB
        headerBuffer[1] = 0            # packet length MSB
    elif packetLength >= 256 and packetLength < 65535:
        headerBuffer[0] = packetLength % 256  # packet length LSB
        headerBuffer[1] = packetLength // 256  # packet length MSB
    else:
        print("           !!!!           ")
        print("PACKET TO SEND IS TOO LONG")
        print("           !!!!           ")
    headerBuffer[2] = channelNbr
    headerBuffer[3] = seqNbr[channelNbr] 
    seqNbr[channelNbr] =  seqNbr[channelNbr] + 1
    # 2. Send the header to BNO080
    if debug_print: print("[BNO080] RPi Sending headerBuffer : " + str(headerBuffer))

    spi.transfer(headerBuffer)

    # 3. Prepare user's data packet
    buffer = []
    if data_length > len(shtpData):
        for i in range(data_length-len(shtpData)):
            shtpData.append(0)                      
    for i in range(data_length):
        buffer.append(shtpData[i]%256)

    # 4. Send user's data to BNO080
    if debug_print: print("[BNO080] RPi Sending data : " + str(buffer))
    spi.transfer(buffer)    

    # Release BNO080
    GPIO.output(_cs, GPIO.HIGH)
    if debug_print: print("[BNO080] RPi data sent.")

    # We're done!
    return True


# -------------------------------------------------------------------
# ------------------------ START/RESET ------------------------------
# -------------------------------------------------------------------

def StartIMU(user_bcm_CSPin, user_bcm_WAKPin, user_bcm_INTPin, user_bcm_RSTPin, user_bcm_spiPortSpeed, user_bcm_spiPort):
    count = 0
    while beginSPI(user_bcm_CSPin, user_bcm_WAKPin, user_bcm_INTPin, user_bcm_RSTPin, user_bcm_spiPortSpeed, user_bcm_spiPort) == False and count < 100:  # freq for Linux Kernel 1953000, but BNO080 can go up to 3000000
        print("[BNO080] over SPI not detected. Restart #", count + 1)
        count += 1
    print(".")
    print("Asking IMU to send data...")

# -------------------------------------------------------------------
# --------------------------- MAIN ----------------------------------
# -------------------------------------------------------------------

try:
    enableDebugRecordFile()
    
    if record_file:
        # save data to a file
        # /!\ if run from Terminal, it will write the file in the directory chosen with "cd" command
        # else if run from Thonny IDE, it will write the file in the same dir as the script 
        file = open(file_path, "w")
        # blank the file
        file.write("START IMU - STABILITY RECORD\n")
        file.close()
    
    # user defined values 
    bcm_CSPin = 12
    bcm_WAKPin = 11
    bcm_INTPin = 13
    bcm_RSTPin = 15
    spiPortSpeed = 1953000  # 1953000 recommended for Linux Kernel, max 3000000 for BO080
    spiPort = 0

    # debugging
    enableDebugging()
    
    # start procedure
    StartIMU(bcm_CSPin, bcm_WAKPin, bcm_INTPin, bcm_RSTPin, spiPortSpeed, spiPort)
            
    print("Everything went SMOOTH")
except Exception as e:
    print("An error occurred. Error message : " + str(e))
finally:
    print('.')  # lighten log
    print("Closing SPI communication")
    spi.close()
    print("GPIO cleanup")
    GPIO.cleanup()
    if not file.closed:
        print("Closing the Debug Record File")
        file.close()
    print("Exiting Program")