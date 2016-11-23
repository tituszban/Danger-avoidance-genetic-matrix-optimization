# Danger-avoidance-genetic-matrix-optimization

I have built a small two+one wheeled robot, with four small angle ultrasonic distance sensors, to avoid two random moving "drone" robots (built by one of my mentee) that moved around in a 1x1 meter box.

The software was based on a preveous genetic matrix optimization software, that I used for smaller project like function fitting and matrix inversion.

The optimizer was a non-regionised genetic algorithm, that used elitism, and multiple different cross recombinations, point and area mutations, to create matrices that were selected by the evaluator, on a one-vs-all method.

Since the physical evaluation took a lot of time and each generation contained a large number of speciments, two layers of preevaluation was used that check idle situation reaction and extreme situation reaction to filter the speciments to a low number, that were physically tested against the "drones".

Usually after about three generations, the robot started avoiding the "drones", and a few more generations it survived for minutes in a relatively small place.

The "drones" were programmed in the basic NXTG language, the robot was running a modified linux kernel, and ran the community made operating system EV3-DEV (the development of which I slightly got involved for a short period of time)
