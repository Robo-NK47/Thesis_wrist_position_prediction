// Call of libraries
#include <Wire.h>
#include <SparkFunLSM9DS1.h>
#include <SoftwareSerial.h>

const byte rxPin = 0;
const byte txPin = 1;
SoftwareSerial HC06 (rxPin, txPin);


// defining module addresses PMOD NAV
#define LSM9DS1_M_1  0x1E
#define LSM9DS1_AG_1 0x6B

#define LSM9DS1_M_2  0x1C
#define LSM9DS1_AG_2 0x6A

LSM9DS1 imu1; // Creation of the wrist IMU object
LSM9DS1 imu2; // Creation of the arm IMU object

const int buttonPin = 13; 
int button = false;


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void setup()
{
  pinMode(buttonPin, INPUT);
  attachInterrupt(digitalPinToInterrupt(buttonPin), raise_flag, RISING);
  
  HC06.begin(9600);
  Wire.begin();     //initialization of the I2C communication
  Wire.setClock(9600);
  
  imu1.settings.device.commInterface = IMU_MODE_I2C;
  imu2.settings.device.commInterface = IMU_MODE_I2C;
  
  imu1.settings.device.mAddress = LSM9DS1_M_1;
  imu1.settings.device.agAddress = LSM9DS1_AG_1;

  imu2.settings.device.mAddress = LSM9DS1_M_2;
  imu2.settings.device.agAddress = LSM9DS1_AG_2;
  
  if (!imu1.begin() || !imu2.begin()) //display error message if that's the case
  {
    HC06.println("Communication problem.");
    while (1);
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void loop()
{
  if (button == false)
    {
    if (imu1.gyroAvailable() && imu2.gyroAvailable())
    {
      imu1.readGyro(); //measure with the gyroscope
      imu2.readGyro();
    }
    if (imu1.accelAvailable() && imu2.accelAvailable())
    {
      imu1.readAccel(); //measure with the accelerometer
      imu2.readAccel();
    }
    if ( imu1.magAvailable() && imu2.magAvailable())
    {
      imu1.readMag(); //measure with the magnetometer
      imu2.readMag();
    }

  //display data
  printGyro1(imu1);
  printAccel1(imu1);
  printMag1(imu1);

  printGyro2(imu2);
  printAccel2(imu2);
  printMag2(imu2);
 
  }
 else
 {
  button = false;
  HC06.write(11111.0);
 }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void raise_flag()
{
  button = true;
}

void printGyro1(LSM9DS1 imu)
{
float array1[] = {1000001.0, imu.calcGyro(imu.gx), imu.calcGyro(imu.gy), imu.calcGyro(imu.gz)};
byte *p = (byte*)array1;
for(byte i = 0; i < sizeof(array1); i++)
{
 HC06.write(p[i]);
}
}

void printAccel1(LSM9DS1 imu)
{
float array2[] = {1000002.0, imu.calcAccel(imu.ax), imu.calcAccel(imu.ay), imu.calcAccel(imu.az)};
byte *p = (byte*)array2;
for(byte i = 0; i < sizeof(array2); i++)
{
 HC06.write(p[i]);
}
}

void printMag1(LSM9DS1 imu)
{
float array3[] = {1000003.0, imu.calcMag(imu.mx), imu.calcMag(imu.my), imu.calcMag(imu.mz)};
byte *p = (byte*)array3;
for(byte i = 0; i < sizeof(array3); i++)
{
 HC06.write(p[i]);
}
}

void printGyro2(LSM9DS1 imu)
{
float array1[] = {2000001.0, imu.calcGyro(imu.gx), imu.calcGyro(imu.gy), imu.calcGyro(imu.gz)};
byte *p = (byte*)array1;
for(byte i = 0; i < sizeof(array1); i++)
{
 HC06.write(p[i]);
}
}

void printAccel2(LSM9DS1 imu)
{
float array2[] = {2000002.0, imu.calcAccel(imu.ax), imu.calcAccel(imu.ay), imu.calcAccel(imu.az)};
byte *p = (byte*)array2;
for(byte i = 0; i < sizeof(array2); i++)
{
 HC06.write(p[i]);
}
}

void printMag2(LSM9DS1 imu)
{
float array3[] = {2000003.0, imu.calcMag(imu.mx), imu.calcMag(imu.my), imu.calcMag(imu.mz)};
byte *p = (byte*)array3;
for(byte i = 0; i < sizeof(array3); i++)
{
 HC06.write(p[i]);
}

}
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
