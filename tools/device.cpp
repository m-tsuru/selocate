#include "MeOrion.h"
#include <Wire.h>

MeGyro gyro;
MeUltrasonicSensor ultraSensor(PORT_7);

unsigned long lastUltraTime = 0;

void setup()
{
  Serial.begin(9600);
  gyro.begin();
}

void loop()
{
  // Read Gyro Sensors
  gyro.update();
  Serial.read();
  Serial.print(gyro.getAngleX());
  Serial.print(" ");
  Serial.print(gyro.getAngleY());
  Serial.print(" ");
  Serial.print(gyro.getAngleZ());
  Serial.print(" ");

  // 超音波は 60ms 以上あけて読む
  if (millis() - lastUltraTime >= 60) {
    lastUltraTime = millis();
    Serial.println(ultraSensor.distanceCm());
  } else {
    Serial.println();  // 表示ずれ防止
  }

  delay(100);
}
