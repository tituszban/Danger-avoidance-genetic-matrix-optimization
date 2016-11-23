#!/usr/bin/python
from ev3dev.ev3 import *

import time
import random
import math
import sys

us1 = UltrasonicSensor (INPUT_1)
us2 = UltrasonicSensor (INPUT_2)
us3 = UltrasonicSensor (INPUT_3)
us4 = UltrasonicSensor (INPUT_4)
btn = Button()

m_A = LargeMotor(OUTPUT_A)
m_D = LargeMotor(OUTPUT_D)

sensors = [us1, us2, us3, us4]
max_sensor_values = [2560.0, 255.0, 255.0, 2560.0]

sensor_near_tresholds = [0.985, 0.975, 0.973, 0.975]
sensor_over_tresholds = [0.01, 0.01, 0.01, 0.01]
running_average_val = 2.0

init_range = 200

def Clamp(v, mi, ma):
	if v < mi:
		return mi
	elif v > ma:
		return ma
	else:
		return v

def Quit():
	m_A.stop()
	m_D.stop()
	sys.exit()
		
def MatrixMultiplication(mx_A, mx_B):
	if not type(mx_A) == list or not type(mx_B) == list:
		return
	if not type(mx_A[0]) == list or not type (mx_B[0]) == list:
		return
	dim_A = [len(mx_A), len(mx_A[0])]
	dim_B = [len(mx_B), len(mx_B[0])]
	
	if not dim_A[1] == dim_B[0]:
		return
	dim_res = [dim_A[0], dim_B[1]]
	res = []
	for i in range(0, dim_res[0]):
		line = []
		for j in range(0, dim_res[1]):
			s = 0
			for k in range(0, dim_A[1]):
				s += mx_A[i][k] * mx_B[k][j]
			line.append(s)   
		res.append(line)
	return res

def Transpose(m):
	res = []
	if not type(m) == list:
		return m
	if not type(m[0]) == list:
		for i in range(0, len(m)):
			res.append([m[i]])
		return res
	
	dim = [len(m), len(m[0])]
	for i in range(0, dim[1]):
		line = []
		for j in range(0, dim[0]):
			line.append(m[j][i])
		res.append(line)
	return res

def WRandom(n, p):
	#n: max value, p: steepness of random function
	x = random.random()
	return int(math.floor(- n * x ** p + n))

def RandomPair(prev, rand, randparam_n, randparam_p):
	#prev: previous value, rand: random generating method, randparam_n, randparam_p: rand parameters
	r = rand(randparam_n, randparam_p)
	if prev == r:
		return RandomPair(prev, rand, randparam_n, randparam_p)
	else:
		return r

def RandomPairs(prev, rand, randPair, randparam_n, randparam_p):
	#prev: previous pair, rand: random generating method, randPair: pair generator, randparam_n, randparam_p: rand parameters
	r1 = WRandom(randparam_n, randparam_p)
	r2 = randPair(r1, rand, randparam_n, randparam_p)
	if [r1, r2] == prev:
		return RandomPairs(prev, rand, randPair, randparam_n, randparam_p)
	else:
		return [r1, r2]

def RandomList(n, m, p, rand = WRandom, randomPair = RandomPair, randomPairs = RandomPairs):
	#n: max number in array; m: number of pairs, p: steepness of random, rand: random generating method, randomPair: pair generator, randomPairs: next pair generator
	rnds = []
	r = rand(n, p)
	rnds.append([r, randomPair(r, rand, n, p)])
	for i in range(1, m):
		rnds.append(randomPairs(rnds[i - 1], rand, randomPair, n, p))
	return rnds

def RandomMatrix(size, borders):
	res = []
	for x in range(0, size[0]):
		line = []
		for y in range(0, size[1]):
			line.append((random.random() * abs(borders[0] - borders[1]) + borders[0]))
		res.append(line)
	return res

def PointRecombination(mx_A, mx_B):
	p = [random.randrange(0, len(mx_A)), random.randrange(0, len(mx_A[0]))]
	res = []
	for x in range(0, len(mx_A)):
		line = []
		for y in range(0, len(mx_A[0])):
			if x == p[0] and y == p[1]:
				line.append(mx_B[x][y])
			else:
				line.append(mx_A[x][y])
		res.append(line)
	return res

def CrossRecombination(mx_A, mx_B):
	p_A = [random.randrange(0, len(mx_A)), random.randrange(0, len(mx_A[0]))]
	p_B = [random.randrange(0, len(mx_B)), random.randrange(0, len(mx_B[0]))]
	res = []
	for x in range(0, len(mx_A)):
		line = []
		for y in range(0, len(mx_A[0])):
			if x == p_A[0] and y == p_A[1]:
				line.append(mx_B[p_B[0]][p_B[1]])
			else:
				line.append(mx_A[x][y])
		res.append(line)
	return res

def AreaMoveRecombination(mx_A, mx_B):
	m = [random.randint(-(len(mx_A) - 1), len(mx_A) - 1), random.randint(-(len(mx_A[0]) - 1), len(mx_A[0]) - 1)]
	res = []
	for x in range(0, len(mx_A)):
		line = []
		for y in range(0, len(mx_A[0])):
			if x >= m[0] and x < m[0] + len(mx_A) and y >= m[1] and y < m[1] + len(mx_A[0]):
				line.append(mx_B[x - m[0]][y - m[1]])
			else:
				line.append(mx_A[x][y])
		res.append(line)
	return res

def AreaRecombination(mx_A, mx_B):
	m = [random.randint(-(len(mx_A) - 1), len(mx_A) - 1), random.randint(-(len(mx_A[0]) - 1), len(mx_A[0]) - 1)]
	res = []
	for x in range(0, len(mx_A)):
		line = []
		for y in range(0, len(mx_A[0])):
			if x >= m[0] and x < m[0] + len(mx_A) and y >= m[1] and y < m[1] + len(mx_A[0]):
				line.append(mx_B[x][y])
			else:
				line.append(mx_A[x][y])
		res.append(line)
	return res

def AreaWRandom(areas):
	ars = []
	s = sum(areas)
	for i in range(0, len(areas)):
		ars.append(float(areas[i]) / float(s))
	r = random.random()
	i = 0
	while i < len(ars) and r > ars[i]:
		r -= ars[i]
		i += 1
	return i       

def Recombine(mx_A, mx_B, p = 1, actions = [PointRecombination, CrossRecombination, AreaRecombination, AreaMoveRecombination], costs = [0.7, 0.55, 0.45, 0.25]):
	if random.random() > p:
		#print p
		return mx_A
	ac = AreaWRandom(costs)
	#print p, actions[ac]
	m = actions[ac](mx_A, mx_B)
	diff = 1
	if not Difference(mx_A, mx_B) == 0:
		diff = (1 - Difference(m, mx_A) / Difference(mx_A, mx_B)) ** 2
	#print 'd:', diff
	return Recombine(m, mx_B, p * costs[ac] * diff)

def Difference(mx_A, mx_B):
	d = 0
	for i in range(0, len(mx_A)):
		for j in range(0, len(mx_A[0])):
			d += abs(mx_A[i][j] - mx_B[i][j])
	return d

def PointAdditiveMutate(mx, r = init_range * 0.1):
	p = [random.randrange(0, len(mx)), random.randrange(0, len(mx[0]))]
	res = []
	for x in range(0, len(mx)):
		line = []
		for y in range(0, len(mx[0])):
			if x == p[0] and y == p[1]:
				line.append(mx[x][y] + random.random() * 2 * r - r)
			else:
				line.append(mx[x][y])
		res.append(line)
	return res

def PointMultiplicativeMutate(mx, r = 3):
	p = [random.randrange(0, len(mx)), random.randrange(0, len(mx[0]))]
	res = []
	for x in range(0, len(mx)):
		line = []
		for y in range(0, len(mx[0])):
			if x == p[0] and y == p[1]:
				line.append(mx[x][y] * random.random() * 2 * r - r)
			else:
				line.append(mx[x][y])
		res.append(line)
	return res

def PointChangeMutate(mx, r = init_range):
	p = [random.randrange(0, len(mx)), random.randrange(0, len(mx[0]))]
	res = []
	for x in range(0, len(mx)):
		line = []
		for y in range(0, len(mx[0])):
			if x == p[0] and y == p[1]:
				line.append(random.random() * 2 * r - r)
			else:
				line.append(mx[x][y])
		res.append(line)
	return res

def Mutate(mx, p = 1, actions = [PointAdditiveMutate, PointMultiplicativeMutate, PointChangeMutate], costs = [0.8, 0.1, 0.2]):
	if random.random() > p:
		#print p
		return mx
	ac = AreaWRandom(costs)
	#print p, actions[ac]
	m = actions[ac](mx)
	diff = Clamp(1 - Difference(m, mx) / 120, 0, 1) ** 2
	#print 'd:', diff
	return Mutate(m, p * costs[ac] * diff)

def Evaluate(m):
	pass

def SortByEvaluation(s, ev):
	change = True
	while change:
		change = False
		for i in range(0, len(ev) - 1):
			if ev[i] < ev[i + 1]:
				ev_np = ev[i]
				ev[i] = ev[i + 1]
				ev[i + 1] = ev_np
				s_np = s[i]
				s[i] = s[i + 1]
				s[i + 1] = s_np
				change = True
	return s

	
def Run_Evaluation(matrix):
	item_close = False
	start = time.clock()
	#print start
	control_matrix = [[0.5, 0.5, 0.5, 0.5]]
	normalised_sensor_values = [0, 0, 0, 0]
	m_A.run_direct()
	m_D.run_direct()
	over_error = 0
	lst_5_avg = [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]]
	while not item_close:
		if btn.backspace:
			Quit()
		if btn.enter:
			item_close = True;
		for i in range(0, 4):
			normalised_sensor_values[i] = 1 - sensors[i].value() / max_sensor_values[i]
			for j in range(0, 4):
				lst_5_avg[i][j] = lst_5_avg[i][j+1]
			lst_5_avg[i][4] = normalised_sensor_values[i]

		for i in range(0, 4):
			control_matrix[0][i] = (control_matrix[0][i] * running_average_val + (normalised_sensor_values[i]) ** 3) / (running_average_val + 1)
		#control_matrix[0][5] = control_matrix[0][1] * control_matrix[0][4]
		#control_matrix[0][6] = control_matrix[0][2] * control_matrix[0][3]
		
		#print normalised_sensor_values
		#print lst_5_avg
		#print control_matrix
		
		#for i in range(0, 4):
		#	if sum(lst_5_avg[i]) / 5.0 >= sensor_near_tresholds[i] or sum(lst_5_avg[i]) / 5.0 <= sensor_over_tresholds[i]:
		#		item_close = True
		
		
		#print sensor_values
		move = MatrixMultiplication(control_matrix, matrix)
		for i in range(0, 2):
			if abs(move[0][i]) > 100:
				over_error += abs(move[0][i]) - 100.0
		
		m_A.duty_cycle_sp = Clamp(move[0][0], -100, 100)
		m_D.duty_cycle_sp = Clamp(move[0][1], -100, 100)
		
		time.sleep(0.01)
	end = time.clock()
	m_A.stop()
	m_D.stop()
	while btn.any():
		time.sleep(0.1)
	#print end
	return end - start

generation = []
#generation_count = 3200
#speciment_count = 0
first_round_speciment_count = 200
second_round_speciment_count = 50
third_round_speciment_count = 8
mutation_chance = 0.8
random_steepnes = 0.2
target_matrix = [[-400, -600], [300, -100], [-300, 100], [400, 600]]
for i in range(0, first_round_speciment_count):
    generation.append([-1, RandomMatrix([4, 2], [-init_range, init_range])])

def Base_value_Evaluation(matrix):
	test_matrix = [[0.5, 0.5, 0.5, 0.5]]
	test_motors = MatrixMultiplication(test_matrix, matrix)
	base_difference = abs(test_motors[0][0] - test_motors[0][1])
	#base_offset = abs(test_motors[0][0]) + abs(test_motors[0][0])
	return -(base_difference)
	
def Extreme_value_Evaluation(matrix):
	test_matrixs = [[[0.9, 0.1, 0.1, 0.1]],  [[0.1, 0.9, 0.9, 0.9]], [[0.1, 0.1, 0.9, 0.1]], [[0.1, 0.1, 0.1, 0.9]]]
	extreme_offset = 0
	for i in range (0, 4):
		test_motors = MatrixMultiplication(test_matrixs[i], matrix)
		target_motors = MatrixMultiplication(test_matrixs[i], target_matrix)
		extreme_offset -= abs(test_motors[0][0] - target_motors[0][0]) + abs(test_motors[0][1] - target_motors[0][1])
	return extreme_offset

def Squared_difference_Evaluation(mx_A, mx_B):
	d = 0
	for i in range(0, len(mx_A)):
		for j in range(0, len(mx_A[0])):
			d += (mx_A[i][j] - mx_B[i][j]) ** 2
	return d

def Simplify(matrix):
	smp = []
	for i in range(0, len(matrix)):
		smp.append([])
		for j in range(0, len(matrix[0])):
			smp[i].append(round(matrix[i][j], 3))
			
	return smp
	
def Run_Generation(generation, gen_count):    
	ev = []
	log = {}
	for i in range(0, len(generation)):
		generation[i][0] = i
		log[i] = [i, -1, -1, -1, -1, -1, -1]
	
	
	
	first_round = generation
	#print generation[0]
	elite = generation[0]
	ev_1st_rnd = []
	for i in range(0, len(first_round)):
		ev_1st_rnd.append(Base_value_Evaluation(first_round[i][1]) + Extreme_value_Evaluation(first_round[i][1]) * 0.5)
	first_round = SortByEvaluation(first_round, ev_1st_rnd)
	
	for i in range(0, len(first_round)):
		log[first_round[i][0]][1] = ev_1st_rnd[i]
		log[first_round[i][0]][2] = i
	
	print 'First round evaluated'
	
	second_round = first_round[0:second_round_speciment_count]
	ev_2nd_rnd = []
	for i in range(0, len(second_round)):
		ev_2nd_rnd.append(-Squared_difference_Evaluation(second_round[i][1], target_matrix))
	second_round = SortByEvaluation(second_round, ev_1st_rnd)
	
	for i in range(0, len(ev_2nd_rnd)):
		log[second_round[i][0]][3] = ev_2nd_rnd[i]
		log[second_round[i][0]][4] = i
	
	third_round = second_round[0:third_round_speciment_count]
	#third_round[-1] = elite
	#print generation[-1]
	#print third_round[-1]
	
	print 'Second round evaluated'
	
	for i in range(0, len(third_round)):
		print 'Speciment_' + str(i)
		Sound.tone([(1000, 100, 100), (500, 500, 500)])
		ev.append(Run_Evaluation(third_round[i][1])) # Run
		Sound.tone([(200, 500, 500)])
		print 'Speciment_' + str(i) + ' scr: ' + str(round(ev[-1], 2))
		print 'base_value: ' + str(round(Base_value_Evaluation(third_round[i][1]), 2))
		print 'extreme_value: ' + str(round(Extreme_value_Evaluation(third_round[i][1]), 2))
		while not btn.any():
			time.sleep(0.1)
			if btn.backspace:
				Quit()
		while btn.any():
			time.sleep(0.1)
	third_round = SortByEvaluation(third_round, ev)
	
	for i in range(0, len(third_round)):
		log[second_round[i][0]][5] = ev[i]
		log[second_round[i][0]][6] = i
		
	f = open('generation%d.txt' % gen_count, 'w')
	output = ''
	for i in range(0, len(generation)):
		outputL = ''
		for j in range(0, 7):
			outputL += str(log[generation[i][0]][j]) + '\t'
		
		output = str(Simplify(generation[i][1])) + '\t' + outputL + " \n"
		f.write(output)
	f.close()
	print 'Saved'
	
	
	
	#print ev
	
	print 'Third round evaluated'
	
	n_generation = []
	n_generation.append(third_round[0][1])
	r_list = RandomList(third_round_speciment_count, first_round_speciment_count - 1, random_steepnes)
	for r in r_list:
		n_spec = Recombine(third_round[r[0]][1], third_round[r[1]][1])
		if random.random() < mutation_chance:
			n_spec = Mutate(n_spec)
		n_generation.append(n_spec)
	#print n_generation
	n_idd_generation = []
	for i in range(0, len(n_generation)):
		n_idd_generation.append([-1, n_generation[i]])
	return n_idd_generation


#print generation



#target_matrix = [[-400, -600], [300, -100], [-300, 100], [400, 600]]
#print Extreme_value_Evaluation(target_matrix)
#print Base_value_Evaluation(target_matrix)
#Run_Evaluation(target_matrix)
#Quit()

generation_count = 0

while True:
	print '\n Generations ' + str(generation_count) + '\n'
	Sound.tone([(1000, 200, 200), (1000, 200, 200), (1000, 200, 200)])
	while btn.any():
		time.sleep(0.1)
	generation_count += 1
	while not (btn.backspace or btn.enter):
		time.sleep(0.1)
		if btn.backspace:
			Quit()
	while btn.any():
		time.sleep(0.1)
	print 'Generation started'
	generation = Run_Generation(generation, generation_count)
