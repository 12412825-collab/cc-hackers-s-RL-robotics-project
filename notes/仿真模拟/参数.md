![[Pasted image 20260718190248.png]]

# For reinforcement learning, what other parameters need to be measured?

Online parameters can only serve as initial values. When doing real simulation, I recommend **measuring these in reality**:

| Parameter | Acquisition Method |
| ----------- | ---------- |
| Total vehicle weight | Weigh with an electronic scale |
| Wheelbase (front-to-rear center distance) | Measure with a ruler |
| Track width (left-to-right) | Measure with a ruler |
| Camera installation height | Measure after installation |
| Camera pitch angle | Measure with a protractor or phone |
| Maximum speed | Time a 1-meter straight run |
| Turning radius | Test by drawing circles on the ground |
| PWM to speed relation | Write a test script and record gear by gear |