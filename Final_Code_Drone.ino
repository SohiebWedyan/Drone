#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>

// Bluetooth RX/TX on pins 0 and 1 (hardware serial)
#define TX_PIN 0
#define RX_PIN 1

// Define the motors
Servo esc1, esc2, esc3, esc4;

// MPU6050 object
MPU6050 mpu;

// PID variables
float rollSetpoint = 0.0, pitchSetpoint = 0.0, yawSetpoint = 0.0; // Added yaw
float rollInput, pitchInput, yawInput;
float rollOutput = 0.0, pitchOutput = 0.0, yawOutput = 0.0; // Added yaw output
float kp_roll = 3.0, ki_roll = 1.5, kd_roll = 1;   // PID gains for roll
float kp_pitch = 3.0, ki_pitch = 1.5, kd_pitch = 1; // PID gains for pitch
float kp_yaw = 1.0, ki_yaw = 0.5, kd_yaw = 0.2;      // PID gains for yaw

// Base speed and throttle control
int baseSpeed1400 = 1200;
int baseSpeed2200 = 1200;
int throttle = 1200; // Initial throttle value

int motorSpeed[4];
bool motorsArmed = false;
char lastCommand = 'S';

// Ultrasonic sensor pins
#define TRIG_PIN 8
#define ECHO_PIN 7

// Variables for PID tuning
bool tuningMode = false;
char tuningAxis = ' '; // 'R' for roll, 'P' for pitch, 'Y' for yaw
float tuningStep = 0.1; // Step for increasing PID values
String inputString = "";

void setup() {
  // Initialize the motors
  esc1.attach(3);
  esc2.attach(5);
  esc3.attach(6);
  esc4.attach(9);

  // Initialize the serial communication for Bluetooth
  Serial.begin(9600);

  // Initialize the MPU6050
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  // Initialize the ultrasonic sensor pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Serial.println("System ready. Waiting for Bluetooth commands...");
}

void loop() {
    // Handle Bluetooth communication
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    if (inChar == '\n' || inChar == '\r') {  // Process the input
      if (tuningMode) {
        processTuningCommand(inputString);
      } else {
        processCommand(inputString.charAt(0));
      }
      inputString = "";
    } else if (inChar != ' ') { // Ignore spaces in the command
      inputString += inChar;
    }
  }

  // Read accelerometer and gyro data
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);


  //Offsets ensure that the sensor outputs 0 for acceleration due to gravity when there's no movement or tilt.
  int16_t ax_offset = 0, ay_offset = 0, az_offset = 0;
  int16_t gx_offset = 0, gy_offset = 0, gz_offset = 0;

  // Calculate roll and pitch inputs with offsets
  rollInput = (float)(ax - ax_offset) / 16384.0;
  pitchInput = (float)(ay - ay_offset) / 16384.0;
  yawInput = (float)(gz - gz_offset) / 131.0 ;  // Gyro to degrees per second


  // Read the distance from the ultrasonic sensor
  long distance = readUltrasonicDistance();

  // Print sensor data and distance (for debugging)
  Serial.print("Ultrasonic Distance: ");
  Serial.print(distance);
  Serial.print(" cm\t");

  Serial.print("MPU6050 - Roll: ");
  Serial.print(rollInput);
  Serial.print(" Pitch: ");
  Serial.print(pitchInput);
  Serial.print(" Yaw: ");
  Serial.println(yawInput);


  // Update motors if armed
  if (motorsArmed) {
      calculatePID();
      updateMotors();
  }


  delay(50);
}

// Process Bluetooth commands
void processCommand(char command) {
  Serial.print("Command received: ");
  Serial.println(command);

  if (command == 'A') {
    armMotors();
    motorsArmed = true;
    lastCommand = 'S';
    Serial.println("Motors armed.");
  } else if (command == 'S') {
    disarmMotors();
    motorsArmed = false;
    lastCommand = 'S';
    Serial.println("Motors disarmed.");
  } else if (command == 'T') { // Enter tuning mode
    tuningMode = true;
    Serial.println("Entered tuning mode. Use 'R', 'P', 'Y' and '+', '-' to adjust PID values.");
  } else if (command == 'E') { // Exit tuning mode
      tuningMode = false;
      Serial.println("Exited tuning mode.");
   }else if (motorsArmed) {
      lastCommand = command;
      executeCommand(lastCommand);
    Serial.println("Command accepted.");
  } else {
    Serial.println("Motors are not armed. Please send 'A' to arm.");
  }
}

void processTuningCommand(String command) {
    Serial.print("Tuning command received: ");
    Serial.println(command);

  if (command.length() == 2) {
    tuningAxis = command.charAt(0);
    char operation = command.charAt(1);

      if(tuningAxis == 'R' || tuningAxis == 'P' || tuningAxis == 'Y'){
         if (operation == '+') {
            if(tuningAxis == 'R'){
                kp_roll += tuningStep;
            } else if (tuningAxis == 'P'){
                kp_pitch += tuningStep;
            } else if(tuningAxis == 'Y'){
                kp_yaw += tuningStep;
            }
            Serial.print("Increased "); Serial.print(tuningAxis); Serial.print(" gain: ");
        } else if (operation == '-') {
            if(tuningAxis == 'R'){
                kp_roll -= tuningStep;
            } else if (tuningAxis == 'P'){
                kp_pitch -= tuningStep;
            } else if(tuningAxis == 'Y'){
                kp_yaw -= tuningStep;
            }
            Serial.print("Decreased "); Serial.print(tuningAxis); Serial.print(" gain: ");
        }
    
    printPIDValues();
    }
  }
}

void printPIDValues(){
  Serial.print("Roll (kp, ki, kd): "); Serial.print(kp_roll); Serial.print(", "); Serial.print(ki_roll); Serial.print(", "); Serial.println(kd_roll);
  Serial.print("Pitch (kp, ki, kd): "); Serial.print(kp_pitch); Serial.print(", "); Serial.print(ki_pitch); Serial.print(", "); Serial.println(kd_pitch);
  Serial.print("Yaw (kp, ki, kd): "); Serial.print(kp_yaw); Serial.print(", "); Serial.print(ki_yaw); Serial.print(", "); Serial.println(kd_yaw);
}



// Arm motors
void armMotors() {
    throttle = 1200;
    motorSpeed[0] = throttle; // Motor 1
    motorSpeed[1] = throttle; // Motor 2
    motorSpeed[2] = throttle; // Motor 3
    motorSpeed[3] = throttle; // Motor 4

    esc1.writeMicroseconds(motorSpeed[0]);
    esc2.writeMicroseconds(motorSpeed[1]);
    esc3.writeMicroseconds(motorSpeed[2]);
    esc4.writeMicroseconds(motorSpeed[3]);
  
    delay(3000);
    Serial.println("Motors armed.");
}

// Disarm motors
void disarmMotors() {
    throttle = 1000;
  motorSpeed[0] = throttle; // Motor 1
  motorSpeed[1] = throttle; // Motor 2
  motorSpeed[2] = throttle; // Motor 3
  motorSpeed[3] = throttle; // Motor 4

    esc1.writeMicroseconds(motorSpeed[0]);
    esc2.writeMicroseconds(motorSpeed[1]);
    esc3.writeMicroseconds(motorSpeed[2]);
    esc4.writeMicroseconds(motorSpeed[3]);
    Serial.println("Motors disarmed.");
}

// Execute commands based on Bluetooth input
void executeCommand(char command) {
    switch (command) {
        case 'F': // Forward
            baseSpeed1400 = 1500;
            baseSpeed2200 = 1400;
            pitchSetpoint = 5.0;
            rollSetpoint = 0.0;
            Serial.println("Moving Forward");
            break;
        case 'B': // Backward
            baseSpeed1400 = 1300;
            baseSpeed2200 = 1300;
            pitchSetpoint = -5.0;
            rollSetpoint = 0.0;
            Serial.println("Moving Backward");
            break;
        case 'L': // Left
            baseSpeed1400 = 1350;
            baseSpeed2200 = 1350;
            rollSetpoint = -5.0;
             pitchSetpoint = 0.0;
            Serial.println("Turning Left");
            break;
        case 'R': // Right
            baseSpeed1400 = 1350;
            baseSpeed2200 = 1350;
            rollSetpoint = 5.0;
            pitchSetpoint = 0.0;
            Serial.println("Turning Right");
            break;
        case 'U': // Up
          throttle += 50;  // Increase throttle (you can modify the step)
          throttle = constrain(throttle, 1000, 2000);
          Serial.print("Increasing throttle, current throttle: "); Serial.println(throttle);
          break;
        case 'D': // Down
          throttle -= 50;  // Decrease throttle
          throttle = constrain(throttle, 1000, 2000);
          Serial.print("Decreasing throttle, current throttle: "); Serial.println(throttle);
          break;
        case 'S':
            baseSpeed1400 = 1200;
            baseSpeed2200 = 1200;
            rollSetpoint = 0.0;
            pitchSetpoint = 0.0;
          Serial.println("Stopping");
          break;
        default:
            break;
        }
}

// Calculate PID for stabilization
void calculatePID() {
    float rollError = rollSetpoint - rollInput;
    float pitchError = pitchSetpoint - pitchInput;
    float yawError = yawSetpoint - yawInput;
  
    // Calculate the PID outputs for each axis
    rollOutput = kp_roll * rollError + ki_roll * rollError + kd_roll * (rollError - rollOutput);
    pitchOutput = kp_pitch * pitchError + ki_pitch * pitchError + kd_pitch * (pitchError - pitchOutput);
    yawOutput = kp_yaw * yawError + ki_yaw * yawError + kd_yaw * (yawError - yawOutput);
}


// Update motor speeds based on PID output
void updateMotors() {
  // Adjust motor speeds based on the calculated PID output
  motorSpeed[0] = constrain(throttle + rollOutput + pitchOutput - yawOutput, 1000, 2000); // Motor 1 (assume 1400kv)
  motorSpeed[1] = constrain(throttle - rollOutput + pitchOutput + yawOutput, 1000, 2000); // Motor 2  (assume 1400kv)
  motorSpeed[2] = constrain(throttle + rollOutput - pitchOutput + yawOutput, 1000, 2000); // Motor 3 (assume 2200kv)
  motorSpeed[3] = constrain(throttle - rollOutput - pitchOutput - yawOutput, 1000, 2000); // Motor 4 (assume 2200kv)

  // Send speeds to motors
  esc1.writeMicroseconds(motorSpeed[0]);
  esc2.writeMicroseconds(motorSpeed[1]);
  esc3.writeMicroseconds(motorSpeed[2]);
  esc4.writeMicroseconds(motorSpeed[3]);

  // Print motor speeds to serial monitor for debugging
  Serial.print("Motor speeds: ");
  Serial.print(motorSpeed[0]); Serial.print(", ");
  Serial.print(motorSpeed[1]); Serial.print(", ");
  Serial.print(motorSpeed[2]); Serial.print(", ");
  Serial.println(motorSpeed[3]);

}


// Function to read the ultrasonic sensor
long readUltrasonicDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH);
  long distance = duration / 58.2;
  return distance;
}