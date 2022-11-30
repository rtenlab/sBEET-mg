/*
 * Adafruit nRF52 BLE
 *
 * Arduino output to CSV: https://circuitjournal.com/arduino-serial-to-spreadsheet
 * Sync PC time to Arduino: sudo echo "T$(($(date +%s)+60*60*$TZ_adjust))" >/dev/ttyACM0
 * 
 */

#include <Adafruit_INA260.h>
#include <SoftwareSerial.h>

const int sensor_count = 3;
const int start_count = 0;	// default 0

/* 
0x40: T400 PCIE
0x41: 3070 PCIE
0x44: 3070 PSU
*/ 
const int i2c_address[4] = {0x40, 0x41, 0x44, 0x45};

typedef struct sensor_t {
	Adafruit_INA260 ina260;
	float current = 0;
	float voltage = 0;
	float power = 0;
}sensor_t;

typedef struct overall_data_t {
	float current = 0;
	float voltage = 0;
	float power = 0;
}overall_data_t;

sensor_t sensor[3];
overall_data_t overall_data;

void setup() {
	Serial.begin(1000000);
	while (!Serial) { delay(10); }
	Wire.setClock(400000);
	/* Setting up I2C for INA260 */
	// Wait until serial port is opened
	for (int i = start_count; i < sensor_count + start_count; i++) {
		sensor[i].ina260 = Adafruit_INA260();
	}
	for (int i = start_count; i < sensor_count + start_count; i++) {
		sensor[i].ina260.begin(i2c_address[i]);		
		sensor[i].ina260.setCurrentConversionTime(INA260_TIME_140_us);
		sensor[i].ina260.setVoltageConversionTime(INA260_TIME_140_us);
//		Serial.println(sensor[i].ina260.getCurrentConversionTime());
//		Serial.println(sensor[i].ina260.getVoltageConversionTime());
	}
	// output csv headers
	// unit: mA, mV, mW
	// for (int i = start_count; i < sensor_count + start_count; i++) {
	//  	Serial.print("s");
	//  	Serial.print(i);
	// 	Serial.print(".current");
	// 	Serial.print(",");
	// 	Serial.print("s");
	// 	Serial.print(i);
	// 	Serial.print(".voltage");
	// 	Serial.print(",");
	// 	Serial.print("s");
	// 	Serial.print(i);
	//  	Serial.print(".power");
	//  	Serial.print(",");
	// }
	// Serial.print("overall.current");
	// Serial.print(",");
	// Serial.print("overall.voltage");
	// Serial.print(",");
	//Serial.println("overall.power");
}

int row_count = 0;

void loop() {
	// overall_data.current = 0;
	// overall_data.voltage = 0;
	overall_data.power = 0;
	for (int i = start_count; i < sensor_count + start_count; i++) {
		// sensor[i].current = sensor[i].ina260.readCurrent();
		// sensor[i].voltage = sensor[i].ina260.readBusVoltage();
		sensor[i].power = sensor[i].ina260.readPower();
		// overall_data.current += sensor[i].current;
		// overall_data.voltage += sensor[i].voltage;
		overall_data.power += sensor[i].power;
	}	

	// print in csv format
	// for (int i = start_count; i < sensor_count + start_count; i++){
	// 	Serial.print(sensor[i].current);
	// 	Serial.print(",");
	// 	Serial.print(sensor[i].voltage);
	// 	Serial.print(",");
	// 	Serial.print(sensor[i].power);
	//  	Serial.print(",");
	// }
	// Serial.print(overall_data.current);
	// Serial.print(",");
	// Serial.print(overall_data.voltage);
	// Serial.print(",");
	Serial.print(overall_data.power);
	Serial.println();
}
