import argparse
import sys
import os
import pprint
import random
from collections import Counter, defaultdict
import datetime
from itertools import product
import scipy.stats as stats
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, beta, norm, pointbiserialr
from sklearn.metrics import accuracy_score
from collections import defaultdict
from pynini import Weight, shortestdistance
# from plotnine import *
import scipy.stats as stats
import math
import functools

from pynini import Weight, shortestdistance, Fst, Arc
from learner_wfst import *

from data_analysis import *

pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x



def memoize(func):
	cache = func.cache = {}
	@functools.wraps(func)
	def memoized_func(*args, **kwargs):
		# Convert any sets in args to frozensets, lists to tuples, and dicts to tuple of tuples
		args = tuple(frozenset(arg) if isinstance(arg, set) 
					 else tuple(arg) if isinstance(arg, list) 
					 else tuple(arg.items()) if isinstance(arg, dict) 
					 else arg for arg in args)
		if args in cache:
			return cache[args]
		else:
			result = func(*args, **kwargs)
			cache[args] = result
			return result
	return memoized_func


class Phonotactics:

	def __init__(self, N=2):
		self.language = language
		self.N = N
		self.hypothesized_grammar = {}
		self.previous_grammar = {}
		self.updated_training_sample = []
		self.O = {}
		self.E = {}
		self.counter = 0
		self.con = set()
		self.tier = []
		self.phone2ix = {}
		self.parameters = {}
		self.threshold = 0.5
		self.confidence = 0.975
		self.penalty_weight = 3.0
		self.memo = {}
		self.max_length = 0
		self.model = 'filtering'
		self.sample_size = 0
		self.use_cache = True
		self.observed_smooth = 0
		self.filter = True
		# self.alpha = 0.00625 # Danis (2019)
		self.padding = False
		self.structure = "local"

	def process_features(self, file_path):
		alphabet = []
		feature_dict = {}
		file = open(file_path, 'r', encoding='utf-8')
		header = file.readline()
		for line in file:
			line = line.rstrip("\n").split("\t")
			alphabet += [line[0]]
			# line = line.split(',')
			feature_dict[line[0]] = [x for x in line[1:]]
			
			feature_dict[line[0]] += [0, 0]

		num_feats = len(feature_dict[line[0]])

		feature_dict['<s>'] = [0  for x in range(num_feats-2)] + ['-', '+']
		feature_dict['<e>'] = [0 for x in range(num_feats-2)] + ['+', '-']

		feat = [feat for feat in header.rstrip("\n").split("\t")]
		feat.pop(0)
		feat.extend([ '<s>','<e>'])

		feat2ix = {f: ix for (ix, f) in enumerate(feat)}
		ix2feat = {ix: f for (ix, f) in enumerate(feat)}
		

		# feature_table = np.chararray((len(alphabet), num_feats))
		# for i in range(inv_size):
		# 	feature_table[i] = feature_dict[ix2phone[i]]
		return feature_dict, feat2ix


	def get_corpus_data(self,filename):
		with open(filename, 'r', encoding='utf-8') as file:
			raw_data = [line.rstrip().split(' ') for line in file]

		processed_data = []
		for line in raw_data:
			# line = ['<s>'] + line + ['<e>']
			
			# if len(line) > 1:
			processed_data.append(line)

		random.shuffle(processed_data)
		# pp.pprint(processed_data)

		return processed_data


	def vectorize_length(self, data):
		'''write the number to the list every time you see a word with certain length'''
		m = 0
		for w in data:
			if len(w) >= m:
				m = len(w)
		l =  [0]*m
		for w in data:
			l[len(w)-1] += 1
		
		# for i in range(m):
			# length of the string is divided by the possible combination in expected sample
			# l[i] = l[i]/2**(i+1)

		# alphabet = list(set(phone for w in raw_data for phone in w))
		# max_chars = max([len(x) for x in raw_data])
		return m, l


	def make_count_table(self, grammar):
		'''
		Print how often each segment in a pair was observed and how often it was expected
		'''
		# Ensure boundary symbols are in the tier list
		tier = self.tier
		if '<s>' not in tier:
			tier.append('<s>')
		if '<e>' not in tier:
			tier.append('<e>')


		sl2_possible = [tuple(comb) for comb in product(tier, repeat=2) if '<p>' not in comb if '<e>' not in comb]
		header = '\t'+'\t'.join(tier)
		rows = [header]
		pairsdic = {}
		for bigram in sl2_possible:
			if bigram in grammar:
				# if self.bigram_freqs[bigram] != 0:
				# 	print(bigram)
				pairsdic[bigram] = 0 
			else: 
				pairsdic[bigram] = 1

		for seg in tier:
			row = [seg]
			for otherseg in tier:
				pair = (str(seg),str(otherseg))
				row.append(str(pairsdic.get(pair, 'NaN')))
			outrow = '\t'.join(row)
			rows.append(outrow)
		
		# Convert ARPAbet to IPA symbols
		ARPAbet_to_IPA = {
			'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', '<s>': '<s>', '<e>': '<e>', '<p>': '<p>',
		}
		if self.language == 'english':
			rows = [[ARPAbet_to_IPA.get(cell, cell) for cell in row.split('\t')] for row in rows]

		return (rows)


	def match(self, ngrams, grammar):
		# if any(ngram in grammar.keys() for ngram in ngrams):
		# 	print(ngrams, grammar, any(ngram in grammar for ngram in ngrams))
		# 	# breakpoint()
		return any(ngram in grammar.keys() for ngram in ngrams)


	def ngramize_item(self, string):
		N = self.N
		return [tuple(string[i:i+N]) for i in range(len(string) - (N - 1))] if len(string) >= N else []


	def count_bigram_frequencies(self, data):
		bigram_freqs = defaultdict(int)
		for sequence in data:
			for i in range(len(sequence) - 1):
				bigram = (sequence[i], sequence[i + 1])
				bigram_freqs[bigram] += 1
		return dict(bigram_freqs)


	def penalty(self, wfst, src, arc, grammar):
			symbol = (wfst.state_label(src)[0], wfst.ilabel(arc))
			if symbol in grammar:
				return Weight('log', self.penalty_weight) # Add a penalty term based on the number of constraints in the grammar
			# the more constraints in the grammar, the lower the expected frequency, and the 
			# lesser constraints to be added
				# return Weight('log', 3.0 + 0.1 * len(grammar)) # Add a penalty term based on the number of constraints in the grammar
			else:
				return Weight('log', 0.0)


	def Z(self, wfst, use_cache):
		if use_cache and wfst in self.memo:
			return self.memo[wfst]

		beta = shortestdistance(wfst, reverse=True)
		beta = np.array([float(w) for w in beta])
		result = np.exp(-beta[0])

		if use_cache:
			self.memo[wfst] = result

		return result
	
	
	def observed_dictionary(self):
		observed = {i: 0 for i in self.con}
		for string in self.updated_training_sample:
			unique_ngrams = set(self.ngramize_item(string))
			for ngram in unique_ngrams:
				if ngram in observed:
					observed[ngram] += 1
		return observed


	def expected_dictionary(self):

		use_cache = self.use_cache
		con = self.con
		hypothesized_grammar = list(self.hypothesized_grammar.keys())
		self.tier = [symbol for symbol in self.tier if symbol not in ['<s>', '<e>']]

		if self.padding == True:

			max_length = self.max_length # - 2
			E = {constraint: 0 for constraint in con}
			M_previous = ngram(context='left', length=1, arc_type='log')
			M_previous.assign_weights(hypothesized_grammar, self.penalty)
			M_updated = ngram(context='left', length=1, arc_type='log')

			A = braid(max_length, arc_type='log')
			S_previous = compose(A, M_previous)	# Z_S_previous = Z(S_previous)  # Store the value of Z(S_previous) to avoid recomputing it inside the loop
			Z_S_previous = self.Z(S_previous, use_cache)  # or False to disable caching
		
			# os.system('dot -Tpdf plot/S_previous.dot -o plot/S_previous.pdf')
			for constraint in con:
				if constraint not in hypothesized_grammar:
					hypothesized_grammar.append(constraint)
					M_updated.assign_weights(hypothesized_grammar, self.penalty)
					S_updated = compose(A, M_updated)
					E[constraint] += (1.0 - (self.Z(S_updated, use_cache) / Z_S_previous)) * self.sample_size
					hypothesized_grammar.remove(constraint)
					# max_length = self.max_length - 2
		else:
			# len_vector = self.len_vector[2:]
			len_vector = self.len_vector
			print(len_vector)
			E = {constraint: 0 for constraint in con}
			M_previous = ngram(context='left', length=1, arc_type='log')
			M_previous.assign_weights(hypothesized_grammar, self.penalty)

			M_updated = ngram(context='left', length=1, arc_type='log')
			for n in range(len(len_vector)):
				A = braid(n+1, arc_type='log') # length 10
				S_previous = compose(A, M_previous)	# Z_S_previous = Z(S_previous)  # Store the value of Z(S_previous) to avoid recomputing it inside the loop
				Z_S_previous = self.Z(S_previous, use_cache)  # or False to disable caching
			
				# os.system('dot -Tpdf plot/S_previous.dot -o plot/S_previous.pdf')
				for constraint in con:
					if constraint not in hypothesized_grammar:
						hypothesized_grammar.append(constraint)
						M_updated.assign_weights(hypothesized_grammar, self.penalty)
						S_updated = compose(A, M_updated)
						E[constraint] += (1.0 - (self.Z(S_updated, use_cache) / Z_S_previous)) * len_vector[n]
						hypothesized_grammar.remove(constraint)
		return E


	def upper_confidence_limit_wald(self, O, E):
		confidence = self.confidence

		if E == 0:
			print(f"Upper confidence limit is infinity because E = {E}")
			return float('inf')
		else:
			p = O / E
			z = stats.norm.ppf(confidence)
			denom = 1 + z**2 / E
			center = p + z**2 / (2 * E)
			radius = z * ((p * (1 - p) / E + z**2 / (4 * E**2))**0.5)
			upper_limit = (center + radius) / denom
			if np.isnan(upper_limit) and O <= E:
				print(f"NaN value encountered: O = {O}, E = {E}, confidence = {confidence}, p = {p}, z = {z}, denom = {denom}, center = {center}, radius = {radius}")
			return upper_limit
		
		
	def calculate_OE_and_upper_confidence(self, observed, expected):
		alpha = self.confidence
		n = expected
		t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)  # t-value for two-tailed test
		O = observed + self.observed_smooth
		E = expected

		# Calculate OE ratio
		OE_ratio = float('inf') if E == 0 else O / E

		# Calculate upper confidence limit
		upper_limit = 1.0
		if O < E and E != 0.0:
			p = OE_ratio
			std_err = (p * (1 - p)) / n
			pi_U = p + math.sqrt(std_err) * t_value
			upper_limit = pi_U if not np.isnan(pi_U) else upper_limit

			if np.isnan(pi_U):
				print(f"NaN value encountered: O = {O}, E = {E}, confidence = {alpha}, p = {p}, t_value = {t_value}, std_err = {std_err}, lower_limit = {pi_U}")

		return OE_ratio, upper_limit
	
	def iterate_and_update(self):
		'''HW algo'''
		# self.tier.remove('<s>')
		# self.tier.remove('<e>')
		# config_wfst = {'sigma': self.tier}
		# config.init(config_wfst)

		# Define a list of O/E ratio thresholds for the stepwise rising accuracy scale
		max_threshold = self.threshold
		thresholds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		filtered_thresholds = [threshold for threshold in thresholds if threshold <= max_threshold]
		for threshold in filtered_thresholds:
			if self.filter == True:
				self.updated_training_sample = [
				string for string in self.updated_training_sample if not self.match(self.ngramize_item(string), self.hypothesized_grammar)]
				self.con = set(self.con) - set(self.hypothesized_grammar)
				print(len(self.updated_training_sample))
			self.max_length, self.len_vector = self.vectorize_length(self.updated_training_sample)
			self.sample_size = len(self.updated_training_sample)
			updated = False
			self.counter += 1
			print(f"\nIteration {self.counter}: ")
			self.O = self.observed_dictionary()
			self.E = self.expected_dictionary()
			print(self.E)
			# update E here and try new threshold
			for constraint in self.con:
				O = self.O.get(constraint, 0)
				E = self.E.get(constraint, 0)


				OE_ratio, upper_limit = self.calculate_OE_and_upper_confidence(O, E)
				# If the upper confidence limit is less than the threshold, add the constraint		
				if upper_limit <= threshold:
					print("constraint "+ str(constraint) + " OE_ratio "+ str(OE_ratio) + " Upper limit "+str(upper_limit))
					self.hypothesized_grammar[constraint] = (O, E, upper_limit)
					updated = True

		# If no new constraints are being added, break out of the loop
		if not updated:
			return self.hypothesized_grammar
		
		return self.hypothesized_grammar
		

	def scan_and_judge_categorical(self, input_filename, out_filename, neg_grammar, pos_grammar):
		tier = self.tier
		if '<s>' not in tier:
			tier.append('<s>')
		if '<e>' not in tier:
			tier.append('<e>')
		inp_file = open(input_filename, 'r', encoding='UTF-8')
		out_file = open(out_filename, 'w', encoding='UTF-8')

		data = []
		as_strings = []

		for line in inp_file:
			line = line.rstrip()
			as_strings.append(line)
			line = line.split()
			data.append([i for i in line if i in self.tier])

		# # Compute the maximum count
		# max_count = 0
		# min_count = float('inf')  # start with infinity so any count will be less
		# all_counts = []

		# for string in data:
		# 	ng = self.ngramize_item(string)
		# 	count = sum(neg_grammar[ngram] for ngram in ng if ngram in neg_grammar)
		# 	all_counts.append(count)
		# 	max_count = max(max_count, count)
		# 	min_count = min(min_count, count)

		# # Now, calculate mu (mean) and sigma (standard deviation)
		# mu = np.mean(all_counts)
		# sigma = np.std(all_counts)

		# # scale_factor = 1 / np.log(max_count + 1)  # to ensure that the probability stays between 0 and 1
		
		# def z_to_prob(z):
		# 	return stats.norm.cdf(z)
		
		# Scale and shift the probability to center it around 1.0
	
		for i, string in enumerate(data):
			curr_string = as_strings[i]
			ngrams = self.ngramize_item(string)

			if all(ngram in pos_grammar for ngram in ngrams):
				# breakpoint()
				prob = 1.0 # very good
			else:
				prob = 0.0
				# print(neg_grammar) #(bigrarm):(O, E, O/E)
				# counts = [neg_grammar[ngram][0] for ngram in ng if ngram in neg_grammar]
				# if any(count == 0 for count in counts):
				# 	prob = 0.0
				# else:
				# 	count = sum(counts)
				# 	z = (count - mu) / sigma if sigma > 0 else 0.0
				# 	prob = z_to_prob(z)

			# scaling techniques 1
			# prob = (count - 0/ max_count - 0) if max_count > 0 else 0.0
			# scaling techniques 2: MinMax Scaling
			# prob = (count - min_count) / (max_count - min_count) if max_count > min_count else 0.0
			# scaling techniques 3: Standard Scaling (Z-score normalization)
				# z = (count - mu) / sigma if sigma > 0 else 0.0
				# prob = z_to_prob(z)
			# scaling techniques 4: 

			# prob = np.exp(scale_factor * count) if max_count > 0 else 0.0
			out_file.write(curr_string.rstrip() + '\t' + str(prob) + "\n")

		inp_file.close()
		out_file.close()


	def scan_and_judge(self, input_filename, out_filename, pos_grammar,neg_grammar):
		tier = self.tier
		if '<s>' not in tier:
			tier.append('<s>')
		if '<e>' not in tier:
			tier.append('<e>')
		inp_file = open(input_filename, 'r', encoding='UTF-8')
		out_file = open(out_filename, 'w', encoding='UTF-8')

		data = []
		as_strings = []

		for line in inp_file:
			line = line.rstrip()
			as_strings.append(line)
			line = line.split()
			data.append([i for i in line if i in self.tier])

		# for i, string in enumerate(data):
		# 	curr_string = as_strings[i]
		# 	ngrams = self.ngramize_item(string)
		# 	probability = 1
		# 	for ngram in ngrams:
		# 	# If the bigram is in the grammar, multiply the probability by its relative frequency
		# 		if ngram in grammar:
		# 			probability *= grammar[ngram]
		# 		# If the bigram is not in the grammar, return 0 because the sequence is not valid
		# 		else:
		# 			probability = 0
		for i, string in enumerate(data):
			curr_string = as_strings[i]
			ngrams = self.ngramize_item(string)
			probability = 1
			for ngram in ngrams:
				# If the bigram is in the grammar, multiply the probability by its relative frequency
				# Then also multiply by the high prior probability
				if ngram in pos_grammar:
					probability *= pos_grammar[ngram] * 0.9
				# If the bigram is not in the grammar, multiply the current probability by the low prior probability
				elif ngram in neg_grammar:
					probability *= neg_grammar[ngram] * 0.1

			out_file.write(curr_string + '\t' + str(probability) + "\n")

		inp_file.close()
		out_file.close()


	def goodman_kruskals_gamma(self, x, y):
		concordant = 0
		discordant = 0
		ties = 0

		for i in range(len(x)):
			for j in range(i+1, len(x)):
				if (x[i] > x[j] and y[i] > y[j]) or (x[i] < x[j] and y[i] < y[j]):
					concordant += 1

				elif (x[i] > x[j] and y[i] < y[j]) or (x[i] < x[j] and y[i] > y[j]):
					discordant += 1
					# print(f'discordant at pairs ({x[i]}, {y[i]}) and ({x[j]}, {y[j]})')
				else:  # This now accounts for all other possibilities
					ties += 1
					# print(f'Tie at pairs ({x[i]}, {y[i]}) and ({x[j]}, {y[j]})')

		print("Number of concordant:", concordant)
		print("Number of discordant:", discordant)
		print("Number of ties:", ties)

		try:
			gamma = (concordant - discordant) / (concordant + discordant)

		except ZeroDivisionError:
			gamma = "NaN"		
		tau_a = (concordant - discordant) / (concordant + discordant + ties)

		# print("Number of possible pairs:", concordant + discordant + ties)
		return gamma, tau_a


	def kendalls_tau(self, x, y):
		n = len(x)
		n0 = n * (n - 1) / 2  # total number of pairs
		n1 = n2 = n0
		concordant = discordant = 0
		for i in range(n):
			for j in range(i+1, n):
				xdiff = x[i] - x[j]
				ydiff = y[i] - y[j]
				if xdiff * ydiff < 0:
					discordant += 1
				elif xdiff * ydiff > 0:
					concordant += 1
				else:
					if xdiff != 0:
						n1 -= 1
					if ydiff != 0:
						n2 -= 1
		tau = (concordant - discordant) / ((n1 * n2) ** 0.5)
		# print("Number of possible pairs in tau:", n0)

		return tau
	

	def visual_judgefile(self, humanjudgefile, machine_judgment):

		# Setup plot properties
		title_text = fm.FontProperties(family="Times New Roman")
		axis_text = fm.FontProperties(family="Times New Roman")
		body_text = fm.FontProperties(family="Times New Roman")
		title_text.set_size(16)
		axis_text.set_size(12)
		body_text.set_size(12)

		# Read data
		data = pd.read_csv(humanjudgefile, sep=",", header=0, encoding="utf-8")
		machine_data = pd.read_csv(machine_judgment, sep="\t", names=["onset","grammaticality","score"])
		machine_data["onset"] = machine_data["onset"].str.strip()

		# Add machine judgment to data
		data["machine_judgment"] = machine_data["score"]
		data["form"] = machine_data["onset"]
		data['zscore'] = data['likert_rating'].transform(zscore)
		# Calculate standard deviation and print
		std_dev = np.std(data['likert_rating'])
		print("Standard Deviation: ", std_dev)

		mean_likert_rating = data['likert_rating'].mean()
		print("Likert rating corresponding to z-score=0: ", mean_likert_rating)
		data = data.groupby("form").agg({
		# "score": "mean",
		"likert_rating":"mean",
		"zscore": "mean",
		"machine_judgment": "mean",
		"attestedness": "first" # take the first 'attestedness' value encountered for each form
		}).reset_index()

		# data['likert_rating'] = data['zscore']
		# data['boolean'] = np.where(data['likert_rating']>=2.4, 1, 0)
		data['boolean'] = np.where(data['zscore']>=0, 1, 0)

		print(data)
		# Calculate precision, recall, f1 score
		precision = precision_score(data['boolean'], data['machine_judgment'])
		recall = recall_score(data['boolean'], data['machine_judgment'])
		f1 = f1_score(data['boolean'], data['machine_judgment'])

		# print('Precision: ', precision)
		# print('Recall: ', recall)
		# print('F1 Score: ', f1)

		pearsoncorr, p = pearsonr(data['likert_rating'], data['machine_judgment'])
		print('Pearson correlation: %.3f' % pearsoncorr)
		# pointbiserialrcorr, p = pointbiserialr(data['likert_rating'], data['machine_judgment'])
		# print('Point-biserial correlation: %.3f' % pointbiserialrcorr)
		spearmancorr, s = spearmanr(data['likert_rating'], data['machine_judgment'])
		print('Spearman correlation: %.3f' % spearmancorr)

		gamma, tau_a = self.goodman_kruskals_gamma(data['likert_rating'], data['machine_judgment'])
		# kendalltau
		if gamma == "NaN":
			print('Goodman-Kruskal gamma: NaN')
		else:
			print('Goodman-Kruskal gamma: %.3f' % gamma)

		print('Kendall type a correlation: %.3f' % tau_a)
		kendalltaucorr = self.kendalls_tau(data['likert_rating'], data['machine_judgment'])
		# print('Kendall type b correlation: ' + str(kendalltaucorr))
		print('Kendall type b correlation: %.3f' % kendalltaucorr)

		# Plot the scatter plot
		# annotation_spearman = 'Spearman: %.3f' % round(spearmancorr, 3)
		# a = (
		# 	ggplot(data, aes(x='machine_judgment', y='likert_rating')) + 
		# 	geom_point(aes(color='attestedness', shape='attestedness')) + 
		# 	scale_color_brewer(type="qual", palette="Set1") +
		# 	geom_smooth(method='lm', mapping = aes(x='machine_judgment', y='likert_rating'), color = 'black', inherit_aes=False) +
		# 	labs(x='Predicted judgment', y='Likert rating') + 
		# 	theme(legend_position=(0.2, 0.9), legend_direction='vertical', legend_title=element_blank(),
		# 	figure_size=(3,5),
		# 	axis_line_x=element_line(size=0.6, color="black"),
		# 	axis_line_y=element_line(size=0.6, color="black"),
		# 	panel_grid_major=element_blank(),
		# 	panel_grid_minor=element_blank(),
		# 	panel_border=element_blank(),
		# 	panel_background=element_blank(),
		# 	plot_title=element_text(fontproperties=title_text),
		# 	text=element_text(fontproperties=body_text),
		# 	axis_text_x=element_text(color="black"),
		# 	axis_text_y=element_text(color="black"),
		# 		) + 
		# 	scale_x_continuous(breaks=np.arange(0, 1.005, 0.5), 
		# 					limits=[0, 1.005]) +
		# 	scale_y_continuous(breaks=np.arange(1, 5.01, 1), 
		# 					limits=[1, 5.1]) +
		# 	geom_text(aes(x=0.5, y = 1.9), family = "Times New Roman", label = annotation_spearman
		# 	)
		# )

		# # Save the scatter plot
		# a.save('plot/scatterplot_likert_NT.pdf', dpi=400)
		
		# Return results
		return pearsoncorr, spearmancorr, kendalltaucorr, f1
	
	def evaluate_fscore(self, filepath):
		# Read the data
		data = pd.read_csv(filepath, sep='\t', header=None, names=['word', 'judgment', 'score'])

		# Convert 'grammatical' to 1 and 'ungrammatical' to 0
		data['judgment'] = data['judgment'].map({'grammatical': 1, 'ungrammatical': 0})

		# Calculate F1 score

		precision = precision_score(data['judgment'], data['score'])
		recall = recall_score(data['judgment'], data['score'])
		f1 = f1_score(data['judgment'], data['score'])

		print('Precision: ', precision)
		print('Recall: ', recall)
		print('F1 Score: ', f1)

		# Calculate overall accuracy
		data['correct_prediction'] = (data['judgment'] == data['score']).astype(int)
		overall_accuracy = data['correct_prediction'].mean()
		print(f"Overall accuracy: {overall_accuracy}")

		# Calculate accuracy grouped by 'likert_rating_binary'
		grouped_accuracy = data.groupby('judgment')['correct_prediction'].mean()
		print(f"Grouped accuracy:\n{grouped_accuracy}")

		# Filter and print incorrect predictions
		incorrect_predictions = data[data['correct_prediction'] == 0]
		print(f"Incorrect predictions:\n{incorrect_predictions}")

		pearsoncorr, p = pearsonr(data['judgment'], data['score'])
		print('Pearson correlation: %.3f' % pearsoncorr)
		spearmancorr, s = spearmanr(data['judgment'], data['score'])
		print('Spearman correlation: %.3f' % spearmancorr)
		kendalltaucorr = self.kendalls_tau(data['judgment'], data['score'])
		# print('Kendall type b correlation: ' + str(kendalltaucorr))
		print('Kendall type b correlation: %.3f' % kendalltaucorr)

		gamma, tau_a = self.goodman_kruskals_gamma(data['judgment'], data['score'])
		# kendalltau
		if gamma == "NaN":
			print('Goodman-Kruskal gamma: NaN')
		else:
			print('Goodman-Kruskal gamma: %.3f' % gamma)

		print('Kendall type a correlation: %.3f' % tau_a)


		return f1
	
	def main(self,TrainingFile,FeatureFile,JudgmentFile,TestingFile,MatrixFile):
		raw_training_sample = self.get_corpus_data(TrainingFile)
		alphabet = list(set(phone for w in raw_training_sample for phone in w))
		boundary_list = ['<e>', '<s>', '<p>']
		
		# print(alphabet)
		if self.structure == "nonlocal":
			feature_dict, feat2ix = self.process_features(FeatureFile)	
			vowel = [x for x in feature_dict if feature_dict[x][feat2ix['syll']] == "+" if feature_dict[x][feat2ix['long']] != "+"] #
			self.tier = vowel
		else:
			# feature_dict, feat2ix = self.process_features(FeatureFile)	
			# breakpoint()
			self.tier = alphabet 
		
		self.tier = [item for item in self.tier if item not in boundary_list]

		print(self.tier)

		self.phoneme_to_idx = {p: ix for (ix, p) in enumerate(self.tier)}

		con = [tuple(comb) for comb in product(self.tier, repeat=2)]
		con = [constraint for constraint in con if constraint not in product(boundary_list, repeat=2)]
		self.con = [constraint for constraint in con if not (constraint[1] == '<e>' and constraint[0] in alphabet or constraint[0] == '<p>' and constraint[1] in alphabet or constraint[0] == '<e>' and constraint[1] in alphabet or constraint[0] in alphabet and constraint[1] == '<s>' or constraint[1] in alphabet and constraint[0] == '<s>' or constraint[0] in alphabet and constraint[1] == '<p>')]
		self.updated_training_sample = [[i for i in string if i in self.tier] for string in raw_training_sample]


		if self.padding == True:
			self.max_length, _ = self.vectorize_length(self.updated_training_sample)

			processed_data = []
			seen = set()  # Set for keeping track of seen lines
			for line in self.updated_training_sample:
				str_line = str(line)  # Convert list to string to make it hashable for set
				if str_line not in seen:
					# seen.add(str_line)
					if len(line) < self.max_length:
						line = ['<s>']  + line + ['<p>'] * (self.max_length - len(line)) + ['<e>']
					else:
						line = ['<s>'] + line + ['<e>']
					processed_data.append(line)
			self.updated_training_sample = processed_data

		# print(self.updated_training_sample)
		# print(self.con)
		# breakpoint()
		
		self.bigram_freqs = self.observed_dictionary()
		bigram_freqs = self.bigram_freqs

		self.tier = [symbol for symbol in self.tier if symbol not in ['<s>', '<e>']]

		config_wfst = {'sigma': self.tier}
		config.init(config_wfst)

		if self.model == 'gross':
			converged_neg_grammar = {c:0 for c in self.bigram_freqs if self.bigram_freqs[c]==0}
		else:
			converged_neg_grammar = self.iterate_and_update()
		pp.pprint(converged_neg_grammar)

		# exceptions = 0
		# for k in converged_neg_grammar:
		# 	exceptions += self.bigram_freqs[k]
		# print(exceptions)

		total_freq = sum(self.bigram_freqs.values())
		penalty_factor = 0.5  # adjust this value as needed

		neg_grammar = {c:(self.bigram_freqs[c]/total_freq)*penalty_factor for c in converged_neg_grammar}
		pos_grammar = {c:self.bigram_freqs[c]/total_freq for c in self.con if c not in converged_neg_grammar}

		# converged_grammar = {**pos_grammar, **neg_grammar}
		# self.scan_and_judge(TestingFile, JudgmentFile, pos_grammar,neg_grammar)

		# def tp_exploration():
		# 	def calculate_tolerance(bigram, bigram_freqs):
		# 		factor1, factor2 = bigram
		# 		total_frequency = sum(value for key, value in bigram_freqs.items() if key[0] == factor1 or key[1] == factor1 or key[0] == factor2 or key[1] == factor2)
		# 		return total_frequency / np.log(total_frequency)

			# some test of Tolerance Principle, does not work
			# tolerances = {bigram: calculate_tolerance(bigram, bigram_freqs) for bigram in bigram_freqs}
			# tp_pos = {}
			# tp_neg = {}
			# for bigram, tolerance in tolerances.items():
			# 	# neg_grammar & tp give you the frequency of exceptions

			# 	if bigram in pos_grammar and pos_grammar[bigram] != 0:
			# 		tp_pos[bigram] = (tolerance, bigram_freqs[bigram])
			# 	elif bigram in neg_grammar and neg_grammar[bigram] != 0:
			# 		tp_neg[bigram] = (tolerance, bigram_freqs[bigram])

			# E_neg = 0
			# for k in tp_neg:
			# 	E_neg += tp_neg[k][1]

			# E_pos = 0
			# for k in tp_pos:
			# 	E_pos += tp_pos[k][1]
			
			# breakpoint()

		self.scan_and_judge_categorical(TestingFile, JudgmentFile, neg_grammar, pos_grammar)

		# put constraints in a matrix
		table = self.make_count_table(converged_neg_grammar)
		with open(MatrixFile, 'w') as f:
			f.writelines('\t'.join(row) + '\n' for row in table)

def parse_args():
	# Define the parser and the custom usage message
	parser = argparse.ArgumentParser(
		description="Run phonotactics modeling with customizable settings.",
		usage="""learner_segment_based.py <language> <structure> <max_threshold> <feature_file> 
				<training_file> <testing_file> <judgment_file> <matrix_file> [--weight WEIGHT] [--model MODEL]
			\n\nWhere:
			language        : Set the language for the model.
			structure        : Set structure type (local or nonlocal).
			max_threshold   : Set maximum threshold for the model.
			feature_file    : Path to the feature file.
			training_file   : Path to the training data file.
			testing_file    : Path to the testing data file.
			judgment_file   : Path to the judgment file.
			matrix_file     : Path to the output matrix file.
			WEIGHT          : Weight for WFA transition (default: 10).
			MODEL           : Model type (default: filtering)."""
	)

	# Required positional arguments
	parser.add_argument("language", type=str, help="Language for the model.")
	parser.add_argument("structure", type=str, help="Structure type (local or nonlocal).")
	parser.add_argument("max_threshold", type=float, help="Maximum threshold for the model.")
	# parser.add_argument("feature_file", type=str, help="Path to the feature file.")
	# parser.add_argument("training_file", type=str, help="Path to the training data file.")
	# parser.add_argument("testing_file", type=str, help="Path to the testing data file.")
	# parser.add_argument("judgment_file", type=str, help="Path to the judgment file.")
	# parser.add_argument("matrix_file", type=str, help="Path to the output matrix file.")

	# Optional arguments with defaults
	parser.add_argument("--weight", type=int, default=10, help="Weight for WFA transition (default: 10).")
	parser.add_argument("--model", type=str, default="filtering", help="Model type (filtering or gross, default: filtering).")

	return parser.parse_args()

if __name__ == '__main__':

	# structure = "nonlocal" # or "nonlocal"
	# max_threshold = 0.5 # 0.1 < max_threshold < 1	
	# wfa_transition_weight = 3 # or 3 or 10; small differences
	# model = 'filtering' # 'gross' (Gorman 2013; adapted) or 'filtering' (Dai 2023）
	# Example usage of parsed args

	args = parse_args()
	language = args.language  

	print("Language:", args.language)
	print("Structure:", args.structure)
	print("Max Threshold:", args.max_threshold)
	print("Weight:", args.weight)
	print("Model:", args.model)

	if args.language == 'toy':
		# TrainingFile = 'data/toy/ToyLearningData_CV_3_noCC_except_CCV.txt'
		TrainingFile = 'data/toy/ToyLearningData.txt'
		TestingFile = "data/toy/ToyTestingData.txt"
		FeatureFile = "data/toy/ToyFeatures.txt"
		JudgmentFile = "result/toy/ToyJudgment_%s.txt" % str(datetime.datetime.now()).split(".")[0].replace(" ", "-").replace(":","-")

		MatrixFile = f"result/toy/matrix_{str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '-')}.txt"

	if args.language == 'english':
		FeatureFile = "data/english/EnglishFeatures.txt"
		TrainingFile = 'data/english/EnglishLearningData.txt'
		TestingFile = 'data/english/EnglishTestingData.txt'
		humanJudgment = "data/english/EnglishJudgement.csv"
		MatrixFile = f"result/english/matrix_{str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '-')}.txt"

	if args.language == 'polish':
		FeatureFile = "data/polish/PolishFeatures.txt"
		TrainingFile = 'data/polish/PolishLearningData.txt'
		TestingFile = 'data/polish/PolishTestingData.txt'
		humanJudgment = "NaN"
		MatrixFile = f"result/polish/matrix_{str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '-')}.txt"
	
	if args.language == 'turkish':
		FeatureFile = 'data/turkish/TurkishFeatures.txt'
		TrainingFile = 'data/turkish/TurkishLearningData.txt'
		TestingFile = 'data/turkish/TurkishTestingData.txt'
		humanJudgment = "NaN"
		MatrixFile = f"result/turkish/matrix_{str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '-')}.txt"


	# Assuming Phonotactics class and other necessary imports are defined elsewhere in this script
	phonotactics = Phonotactics()
	phonotactics.language = args.language
	phonotactics.structure = args.structure
	phonotactics.threshold = args.max_threshold
	phonotactics.penalty_weight = args.weight
	phonotactics.model = args.model
	phonotactics.filter = True
	phonotactics.padding = False

	# File paths
	feature_file = f"data/{args.language}/{args.language}Features.txt"
	training_file =  f"data/{args.language}/{args.language}LearningData.txt"
	testing_file = f"data/{args.language}/{args.language}TestingData.txt"
	judgment_file =  f"result/{args.language}/Judgment.txt"
	matrix_file =  f"result/{args.language}/Matrix.txt"

	# Here you can integrate the file paths into the rest of your script
	print("Using feature file:", feature_file)
	# Additional implementation details here

	JudgmentFile = ("result/%s/judgment_struc-%s_thr-%s_pen-%s_model-%s.txt" % 
		(
		phonotactics.language,
		phonotactics.structure, 
		# 'T' if phonotactics.filter else 'F', 
		# 'T' if phonotactics.padding else 'F', 
		# str(phonotactics.confidence), 
		str(phonotactics.threshold),
		str(phonotactics.penalty_weight), 
		phonotactics.model
		)
	)
	


	MatrixFile = ("result/%s/matrix_struc-%s_thr-%s_pen-%s_model-%s.txt" % 
		(
		phonotactics.language,
		phonotactics.structure, 
		# 'T' if phonotactics.filter else 'F', 
		# 'T' if phonotactics.padding else 'F', 
		# str(phonotactics.confidence), 
		str(phonotactics.threshold),
		str(phonotactics.penalty_weight), 
		phonotactics.model
		)
	)

	def run():
		# we also removed line boundary

		phonotactics.main(TrainingFile, FeatureFile, JudgmentFile, TestingFile, MatrixFile)

		# if language == 'english':
		# 	pearsoncorr, spearmancorr, kendalltaucorr, fscore = phonotactics.visual_judgefile(humanJudgment, JudgmentFile)

		# if language == 'polish':
		# 	# Best Spearman correlation:  0.5989787903872147
		# 	# Best hyperparameters:  [[0.1, 0.99]]
		# 	input_path = JudgmentFile
		# 	output_path = "result/polish/correlation_plot.png"
		# 	output_path_ucla = "result/polish/correlation_plot_ucla.png"
		# 	pearsoncorr, spearmancorr, kendalltaucorr, F1 = process_and_plot(input_path, output_path, output_path_ucla)
		
		# if language == 'turkish':
		# 	if "zimmer" in TestingFile.lower():
		# 		_, _, kendalltaucorr = phonotactics.process_and_plot_zimmer(JudgmentFile)
		# 	else:
		# 		fscore = phonotactics.evaluate_fscore(JudgmentFile)
		print("Done! Open the 'result' folder to find your output judgment files!")
	run()


	# code for fitting max_threshold using the testing data
	def hyperparameter_tuning(phonotactics, TrainingFile, JudgmentFile, TestingFile, MatrixFile, humanJudgment, language):
		# Define ranges for hyperparameters
		thresholds = np.linspace(0.001, 1, num=10)
		confidences = np.linspace(0.975, 0.995, num=5)
		penalties = np.linspace(10, 20, num=5)

		best_objective = -1
		best_params = []

		all_objectives = []
		all_thresholds = []

		# Loop over all combinations of hyperparameters
		for threshold in thresholds:
			phonotactics.threshold = threshold
			# phonotactics.confidence = 0.975
			# phonotactics.penalty_weight = 10
			phonotactics.main(TrainingFile, FeatureFile, JudgmentFile, TestingFile, MatrixFile)

			if language == 'english':
				_, spearmancorr, kendalltaucorr, fscore = phonotactics.visual_judgefile(humanJudgment, JudgmentFile)
				all_objectives.append(spearmancorr)  # Store fscore in the list

				# If the F-score is better than the best so far, update best_params and best_objective
				if spearmancorr > best_objective:
					best_objective = spearmancorr
					best_params = [threshold]

			elif language == 'turkish':
				fscore = phonotactics.evaluate_fscore(JudgmentFile)
				all_objectives.append(fscore)  # Store fscore in the list

				# If the F-score is better than the best so far, update best_params and best_objective
				if fscore > best_objective:
					best_objective = fscore
					best_params = [threshold]

			elif language == 'polish':
				output_path = "result/polish/correlation_plot.png"
				output_path_ucla = "result/polish/correlation_plot_ucla.png"
				_, _, kendalltaucorr, _ = process_and_plot(JudgmentFile, output_path, output_path_ucla)
				all_objectives.append(kendalltaucorr)  # Store kendalltaucorr in the list

				# If the Spearman correlation is better than the best so far, update best_params and best_spearman
				if kendalltaucorr > best_objective:
					best_objective = kendalltaucorr
					best_params = [threshold]

			all_thresholds.append(threshold)  # Append threshold for each iteration

		# After the grid search, print the best parameters
		print("Best objectives: ", best_objective)
		print("Best parameters - Threshold: {}".format(*best_params))

		mpl.rc('font',family='Times New Roman')

		# Plot objectives vs thresholds
		fig, ax = plt.subplots()
		ax.plot(all_thresholds, all_objectives, marker='o')  # Plot all objectives and thresholds
		ax.set_xlabel('Threshold', fontname='Times New Roman', fontsize=12)
		ax.set_ylabel('Objective', fontname='Times New Roman', fontsize=12)
		plt.grid(True)

		# Save the figure
		plt.savefig('plot/tuning.png', dpi=400)

	# hyperparameter_tuning(phonotactics, TrainingFile, JudgmentFile, TestingFile, MatrixFile, humanJudgment, language)


