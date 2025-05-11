#include <ESPAsyncWebServer.h>
#include <ESP32Servo.h>

const int servo1Pin = 25;  // GPIO for Servo 1
const int servo2Pin = 26;  // GPIO for Servo 2
const int servo3Pin = 33;  // GPIO for Servo 2
const int servo4Pin = 32;  // GPIO for Servo 2

const int potPin = 34;     // Potentiometer connected to GPIO 34

bool potControlEnabled = true;  // true = potentiometer mode, false = web/app control
int lastPotAngle = 0;           // store last potentiometer angle


Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;

AsyncWebServer server(80);

bool isRecording = false;
bool isReplaying = false;
unsigned long previousMillis = 0;
const int replayInterval = 100;  // Delay between replay movements (in ms)

// Arrays to store recorded angles
const int maxSteps = 500;  // Max steps that can be recorded
int servo1Angles[maxSteps];
int servo2Angles[maxSteps];
int servo3Angles[maxSteps];
int servo4Angles[maxSteps];
int stepCount = 0;
int replayIndex = 0;

// HTML for web interface
String getHTML() {
  return R"rawliteral(
    <!DOCTYPE html>
    <html>
    <head>
      <title>ESP32 Servo Control with Record & Replay</title>
      <style>
        body { text-align: center; font-family: Arial; margin-top: 50px; }
        input[type="range"] { width: 80%; }
        #angleValue1, #angleValue2, #angleValue3, #angleValue4 { font-size: 24px; }
        button { padding: 12px 24px; font-size: 18px; margin: 10px; cursor: pointer; }
        #recordBtn { background-color: red; color: white; border: none; }
        #replayBtn { background-color: green; color: white; border: none; }
      </style>
    </head>
    <body>
      <h2>ESP32 Servo Control with Record & Replay</h2>
      
      <h3>Servo 1 Control</h3>
      <input type="range" id="servoSlider1" min="0" max="180" value="90" onchange="updateServo1(this.value)">
      <p>Angle 1: <span id="angleValue1">90</span>°</p>
      
      <h3>Servo 2 Control</h3>
      <input type="range" id="servoSlider2" min="0" max="180" value="90" onchange="updateServo2(this.value)">
      <p>Angle 2: <span id="angleValue2">90</span>°</p>

      <h3>Servo 3 Control</h3>
      <input type="range" id="servoSlider3" min="0" max="180" value="90" onchange="updateServo3(this.value)">
      <p>Angle 3: <span id="angleValue3">90</span>°</p>

      <h3>Servo 4 Control</h3>
      <input type="range" id="servoSlider4" min="0" max="180" value="90" onchange="updateServo4(this.value)">
      <p>Angle 4: <span id="angleValue4">90</span>°</p>
      
      <button id="recordBtn" onclick="toggleRecord()">Record</button>
      <button id="replayBtn" onclick="replayMovements()">Replay</button>
      <button onclick="togglePot()">Toggle Potentiometer Control</button>
      
      <script>
        let isRecording = false;

        function updateServo1(angle) {
          document.getElementById("angleValue1").innerText = angle;
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/servo1?angle=" + angle, true);
          xhr.send();
        }

        function updateServo2(angle) {
          document.getElementById("angleValue2").innerText = angle;
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/servo2?angle=" + angle, true);
          xhr.send();
        }

        function updateServo3(angle) {
          document.getElementById("angleValue3").innerText = angle;
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/servo3?angle=" + angle, true);
          xhr.send();
        }

        function updateServo4(angle) {
          document.getElementById("angleValue4").innerText = angle;
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/servo4?angle=" + angle, true);
          xhr.send();
        }



        function toggleRecord() {
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/toggleRecord", true);
          xhr.send();
          
          let btn = document.getElementById("recordBtn");
          if (!isRecording) {
            btn.innerText = "Stop";
            btn.style.backgroundColor = "gray";
          } else {
            btn.innerText = "Record";
            btn.style.backgroundColor = "red";
          }
          isRecording = !isRecording;
        }

        function replayMovements() {
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/replay", true);
          xhr.send();
        }

        function togglePot() {
          var xhr = new XMLHttpRequest();
          xhr.open("GET", "/togglePotControl", true);
          xhr.send();
        }
      </script>
    </body>
    </html>
  )rawliteral";
}

void smoothMove(Servo &servo, int startAngle, int targetAngle, int steps, int delayMs) {
  float stepSize = (targetAngle - startAngle) / (float)steps;
  for (int i = 0; i <= steps; i++) {
    int newAngle = startAngle + (stepSize * i);
    servo.write(newAngle);
    delay(delayMs);
  }
}

int getStablePotValue(int samples = 10) {
  long total = 0;
  for (int i = 0; i < samples; i++) {
    total += analogRead(potPin);
    delayMicroseconds(500);  // Small delay between reads
  }
  return total / samples;
}



void setup() {
  Serial.begin(115200);

  // Attach both servos
  servo1.attach(servo1Pin);
  servo2.attach(servo2Pin);
  servo3.attach(servo3Pin);
  servo4.attach(servo4Pin);

  //Set initial angles
  smoothMove(servo1, 110, 120, 50, 10);
  delay(500);
  smoothMove(servo2, 110, 120, 50, 10);
  delay(500);
  smoothMove(servo3, 60, 70, 50, 10);
  delay(500);
  smoothMove(servo4, 70, 80, 10, 10);
  delay(500);


  // Connect to Wi-Fi
  WiFi.begin("MainAuditorium2", "hacktonix2025");
  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected to Wi-Fi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  // ✅ Print IP address           

  // Serve web page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    request->send(200, "text/html", getHTML());
  });

  // Control Servo 1
  server.on("/servo1", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("angle")) {
      String angleParam = request->getParam("angle")->value();
      int angle = angleParam.toInt();

      // Automatically disable pot control when command is received via web
      potControlEnabled = false;
      int currentAngle1 = servo1.read();
      // smoothMove(servo1, currentAngle1, angle, 50, 10);
      servo1.write(angle);
      Serial.printf("Servo 1 angle: %d\n", angle);
      
      // Record angle if recording is active
      if (isRecording && stepCount < maxSteps) {
        servo1Angles[stepCount] = angle;
      }
    }
    request->send(200, "text/plain", "OK");
  });

  // Control Servo 2
  server.on("/servo2", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("angle")) {
      String angleParam = request->getParam("angle")->value();
      int angle = angleParam.toInt();
      int currentAngle2 = servo2.read();
      // smoothMove(servo2, currentAngle2, angle, 50, 10);
      servo2.write(angle);
      Serial.printf("Servo 2 angle: %d\n", angle);
      
      // Record angle if recording is active
      if (isRecording && stepCount < maxSteps) {
        servo2Angles[stepCount] = angle;
        stepCount++;
      }
    }
    request->send(200, "text/plain", "OK");
  });

  server.on("/servo3", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("angle")) {
      String angleParam = request->getParam("angle")->value();
      int angle = angleParam.toInt();
      int currentAngle = servo3.read();
      // smoothMove(servo3, currentAngle, angle, 10, 10);
      servo3.write(angle);
      Serial.printf("Servo 3 angle: %d\n", angle);
      
      // Record angle if recording is active
      if (isRecording && stepCount < maxSteps) {
        servo3Angles[stepCount] = angle;
        stepCount++;
      }
    }
    request->send(200, "text/plain", "OK");
  });

server.on("/servo4", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("angle")) {
      String angleParam = request->getParam("angle")->value();
      int angle = angleParam.toInt();
      int currentAngle = servo4.read();
      // smoothMove(servo4, currentAngle, angle, 10, 10);
      servo4.write(angle);
      Serial.printf("Servo 4 angle: %d\n", angle);
      
      // Record angle if recording is active
      if (isRecording && stepCount < maxSteps) {
        servo4Angles[stepCount] = angle;
        stepCount++;
      }
    }
    request->send(200, "text/plain", "OK");
  });
  // Toggle recording
  server.on("/toggleRecord", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!isRecording) {
      stepCount = 0;
      Serial.println("Recording started...");
    } else {
      Serial.printf("Recording stopped. Total steps: %d\n", stepCount);
    }
    isRecording = !isRecording;
    request->send(200, "text/plain", "OK");
  });

  // Replay recorded movements
  server.on("/replay", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (!isReplaying && stepCount > 0) {
      Serial.println("Replaying recorded movements...");
      isReplaying = true;
      replayIndex = 0;
    }
    request->send(200, "text/plain", "OK");
  });

  server.on("/python", HTTP_GET, [](AsyncWebServerRequest *request) {
    if (request->hasParam("data")) {
      String color = request->getParam("data")->value();
      if(color == "green") {

        servo4.write(90);
        delay(1000);
        servo3.write(90);
        delay(1000);
        servo1.write(80);
        delay(1000);
        servo2.write(140);
        delay(1000);
        

        servo4.write(0);
        delay(1000);
        servo3.write(70);
        delay(1000);
        servo1.write(124);
        delay(1000);
        servo2.write(37);
        delay(1000);
        servo3.write(155);
        delay(1000);

        servo1.write(69);
        delay(1000);
        servo2.write(124);
        delay(1000);
        
        servo4.write(90);
        delay(1000); 
      }
      // if(color == "green"){
      //     servo3.write(20);
      //     delay(200);
      //     servo3.write(60);
      // }
    }
    request->send(200, "text/plain", "OK");
  });

  server.on("/togglePotControl", HTTP_GET, [](AsyncWebServerRequest *request) {
  potControlEnabled = !potControlEnabled;
  Serial.printf("Potentiometer control %s\n", potControlEnabled ? "ENABLED" : "DISABLED");
  request->send(200, "text/plain", potControlEnabled ? "Potentiometer mode ON" : "Web control mode ON");


});


  server.begin();
}

void loop() {
  // if (isReplaying) {
  //   unsigned long currentMillis = millis();
  //   if (currentMillis - previousMillis >= replayInterval) {
  //     previousMillis = currentMillis;

  //     if (replayIndex < stepCount) {
  //       // Get current angles
  //       int currentAngle1 = servo1.read();
  //       int currentAngle2 = servo2.read();
  //       int currentAngle3 = servo3.read();
  //       int currentAngle4 = servo4.read();

  //       // Smoothly move to the recorded angles
  //       smoothMove(servo1, currentAngle1, servo1Angles[replayIndex], 50, 10);
  //       delay(1000);
  //       smoothMove(servo2, currentAngle2, servo2Angles[replayIndex], 50, 10);
  //       delay(1000);
  //       smoothMove(servo3, currentAngle3, servo3Angles[replayIndex], 50, 10);
  //       delay(1000);
  //       smoothMove(servo4, currentAngle4, servo4Angles[replayIndex], 10, 10);
  //       delay(1000);

  //       Serial.printf("Replaying step %d: Servo1=%d, Servo2=%d\n, Servo3=%d\n, Servo4=%d\n", 
  //                     replayIndex, servo1Angles[replayIndex], servo2Angles[replayIndex], servo3Angles[replayIndex], servo4Angles[replayIndex]);
  //       replayIndex++;
  //     } else {
  //       isReplaying = false;
  //       Serial.println("Replay finished.");
  //     }
  //   }
  // }
}
