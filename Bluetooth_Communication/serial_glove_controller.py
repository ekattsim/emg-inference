import serial
import time


class RoboticGloveSerialController:
    def __init__(self, port='/dev/ttyTHS1', baudrate=9600):
        """
        Initializes the serial connection to the Arduino.
        :param port: The serial port name (e.g., '/dev/ttyTHS1' on Jetson).
        :param baudrate: must match the Arduino's Serial.begin().
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = None  # Will hold the serial connection object
        self.is_connected = False

    def connect(self):
        """
        Connects to the Arduino via the specified serial port.
        """
        if self.is_connected:
            print("Already connected.")
            return True

        try:
            print(f"Attempting to connect to {
                  self.port} at {self.baudrate} baud...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            # Wait for the connection to establish and for the Arduino to reset
            time.sleep(2)
            self.is_connected = True
            print(f"Successfully connected to Arduino on {self.port}.")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to serial device {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """
        Closes the serial connection.
        """
        if self.is_connected and self.ser:
            try:
                self.ser.close()
                self.is_connected = False
                print("Disconnected from serial device.")
            except Exception as e:
                print(f"Error disconnecting: {e}")
        else:
            print("Not connected.")

    def read_data(self, lines=1):
        """
        Reads data from the serial port.
        """
        if not self.is_connected:
            print("Cannot read: Not connected.")
            return None

        try:
            data_lines = []
            for _ in range(lines):
                line = self.ser.readline()
                if line:
                    # Decode from bytes to string, stripping newline characters
                    decoded_line = line.decode('utf-8').strip()
                    data_lines.append(decoded_line)
            return "\n".join(data_lines) if data_lines else None
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            return None

    def send_command(self, command: str):
        """
        Sends a command string to the Arduino over the serial port.
        """
        if not self.is_connected:
            print("Not connected to device. Cannot send command.")
            return

        # The Arduino code reads until a '$', so we ensure it's there.
        if not command.endswith("$"):
            command += "$"

        try:
            # Encode the string command to bytes and send it
            self.ser.write(command.encode("utf-8"))
            print(f"Sent command: '{command}'")
        except Exception as e:
            print(f"Error sending command '{command}' over serial: {e}")

    def set_all_servos_batch(self, angles: list[int]):
        """
        Sets all servos in a single, efficient batch command.
        Expects a list of 6 angles.
        Constructs a command like "A90$B30$C0$D45$E180$F90$"
        """
        if len(angles) != 6:
            print(f"ERROR: set_all_servos_batch requires a list of 6 angles. Got {
                  len(angles)}.")
            return

        servo_chars = ['A', 'B', 'C', 'D', 'E', 'F']
        command_parts = []
        for i, angle in enumerate(angles):
            clamped_angle = max(0, min(180, angle))
            command_parts.append(f"{servo_chars[i]}{clamped_angle}$")

        final_command = "".join(command_parts)
        self.send_command(final_command)

    def set_servo_angle(self, servo_index: int, angle: int):
        """
        Sets the angle for a specific servo.
        :param servo_index: The index of the servo (0-5).
        :param angle: The desired angle (0-180 degrees).
        """
        if not 0 <= servo_index <= 5:
            print(f"ERROR: Servo index {servo_index} must be between 0 and 5.")
            return
        if not 0 <= angle <= 180:
            print(f"WARNING: Angle {angle} is outside 0-180. Clamping.")
            angle = max(0, min(180, angle))

        # Command character 'A' is servo 0, 'B' is 1, etc.
        command_char = chr(ord('A') + servo_index)
        command_string = f"{command_char}{angle}"
        self.send_command(command_string)
