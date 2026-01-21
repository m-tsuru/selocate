#include "MeMegaPi.h"
#include <Wire.h>

MeGyro gyro;
MeUltrasonicSensor ultraSensor(PORT_7);
MeMegaPiDCMotor motor1(PORT1B);
MeMegaPiDCMotor motor2(PORT2B);
MeMegaPiDCMotor motor3(PORT3B);
MeMegaPiDCMotor motor4(PORT4B);

// 受信した値を格納する変数
int val[4] = { 0, 0, 0, 0 };
String inputBuffer = "";  // シリアルデータを一時保存するバッファ

void setup() {
  Serial.begin(9600);
  gyro.begin();
}

void loop() {
  // --- 1. シリアル受信処理 ---
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '/') {
      // '/' が来たら、それまでのバッファは「不完全」として破棄し、リセット
      inputBuffer = "";
    } else if (c == '\n' || c == '\r') {
      // 改行が来たら、溜まったデータが「4つの整数」かチェック
      parseSerialData(inputBuffer);
      inputBuffer = "";  // 解析が終わったので空にする
    } else {
      // それ以外の文字（数字やスペース）はバッファに溜める
      inputBuffer += c;
    }
  }

  // Serial.print("Data: ");
  // for (int i = 0; i < 4; i++) {
  //   Serial.print(val[i]);
  //   if (i < 3) Serial.print(", ");
  // }

  if (val[0] == 0) {
    motor1.stop();
  } else {
    motor1.run(val[0]);
  }

  if (val[1] == 0) {
    motor2.stop();
  } else {
    motor2.run(val[1]);
  }

  if (val[2] == 0) {
    motor3.stop();
  } else {
    motor3.run(val[2]);
  }

  if (val[3] == 0) {
    motor4.stop();
  } else {
    motor4.run(val[3]);
  }

  // --- 2. ジャイロ処理と出力 ---
  gyro.update();

  // 10msごとに現在のステータスを表示
  Serial.print("/");
  Serial.print(" ");
  Serial.print(millis());
  Serial.print(" ");
  Serial.print(gyro.getAccX());
  Serial.print(" ");
  Serial.print(gyro.getAccY());
  Serial.print(" ");
  Serial.print(gyro.getAccZ());
  Serial.print(" ");
  Serial.print(gyro.getAngleX());
  Serial.print(" ");
  Serial.print(gyro.getAngleY());
  Serial.print(" ");
  Serial.println(gyro.getAngleZ());

  delay(10);
}

// バッファ内の文字列から4つの整数を抽出する関数
void parseSerialData(String data) {
  int tempVal[4];
  int foundCount = 0;

  // 文字列を解析して整数を抽出
  // sscanfは成功した抽出数を返すので、それが4の時だけ本採用する
  foundCount = sscanf(data.c_str(), "%d %d %d %d", &tempVal[0], &tempVal[1], &tempVal[2], &tempVal[3]);

  if (foundCount == 4) {
    // 4つ揃っている場合のみ、メインの変数に代入
    for (int i = 0; i < 4; i++) {
      val[i] = tempVal[i];
    }
  }
  // 4つ揃っていない場合は、何もしない（スキップ）
}
