from machine import PWM

class Servo:
    def __init__(self, pin):
        self.pwm = PWM(pin, freq=50)
        self.min_us = 500
        self.max_us = 2500

    def write(self, angle):
        angle = max(0, min(180, angle))

        us = self.min_us + (self.max_us - self.min_us) * (angle / 180)
        duty = int(us * 1023 / 20000)  # 20ms period

        self.pwm.duty(duty)
