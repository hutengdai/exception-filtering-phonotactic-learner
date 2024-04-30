from platform import machine
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
# from statannot import add_stat_annotation
# import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
import pprint
import numpy as np
import matplotlib.font_manager as fm
import os
os.system("color")

from plotnine import *
import matplotlib.font_manager as fm
from sklearn.metrics import precision_score, recall_score, f1_score

pp = pprint.PrettyPrinter(indent=4)
pprint.sorted = lambda x, key=None: x
  
# Constants
MEAN = {True: 1, False: 0}
STD_DEV = 0.1
FLOAT_RANGE = (0, 1)

# Function mapping
BOOLEAN_TO_LIKERT = np.vectorize(lambda value: np.clip(np.random.normal(MEAN[value], STD_DEV), *FLOAT_RANGE))



def boolean_to_logistic(value, n, noise_mean=0, noise_std=0.1):
	ratings_sum = 0

	for _ in range(n):
		if value:
			log_odds = 1.3862943611198906  # Log-odds for the True value
		else:
			log_odds = -1.3862943611198906  # Log-odds for the False value

		# Calculate the inverse sigmoid function to map log-odds to a probability
		probability = 1 / (1 + np.exp(-log_odds))

		# Map the probability to the valid Likert rating range (1 to 5)
		rating = round(probability * 4 + 1)
		ratings_sum += rating

	average_rating = ratings_sum / n

	# Add random noise to the average rating
	noisy_average_rating = average_rating + np.random.normal(noise_mean, noise_std)

	# Ensure the noisy rating is within the valid Likert rating range (1 to 5)
	noisy_average_rating = max(1, min(noisy_average_rating, 5))

	return noisy_average_rating

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def logit(p):
	return np.log(p / (1 - p))


def tanh(x):
	return np.tanh(x)


def eharmony(x):
	return np.exp(-x)


def laplace_correction(counter, k=0):
	return counter + k


def log_corrected_counts(counter):
	return sigmoid(counter)


def acceptability(score):
	corrected_counts = laplace_correction(score)
	log_counts = log_corrected_counts(corrected_counts)
	return log_counts


def normalize(series):
	return (series - series.min()) / (series.max() - series.min())



def visual_judgefile(humanjudgefile, machine_judgment):
	# Read data
	data = pd.read_csv(humanjudgefile, sep=",", header=0, encoding="utf-8")
	machine_data = pd.read_csv(machine_judgment, sep="\t", names=["onset","grammaticality","machine_judgment"], encoding="utf-8")

	# Add machine judgement to data
	data["machine_judgment"] = machine_data["machine_judgment"]
	data["form"] = machine_data["onset"]

	# Transform machine judgement to Likert scale
	# data['normalized_machine_judgement'] = (data['machine_judgment'])

	# data['likert_rating'] = data['likert_rating'].apply(log)
	# data['machine_judgement_normal'] = data['normalized_machine_judgement']
	# data['form'] = machine_data['onset']
	# selected_columns = data[["machine_judgment",'likert_rating','attestedness', "normalized_machine_judgement", "machine_judgement_normal", "form"]]
	# data = selected_columns.groupby("form").agg({
	# "machine_judgment": "mean",
	# "normalized_machine_judgement": "mean",
	# "machine_judgement_normal": "mean",
	# "likert_rating":"mean",
	# "attestedness": "first" # take the first 'attestedness' value encountered for each form
	# }).reset_index()

	data = data.groupby("form").agg({
	"score": "mean",
	"likert_rating":"mean",
	"machine_judgment": "mean",
	"attestedness": "first" # take the first 'attestedness' value encountered for each form
	}).reset_index()
	data['likert_rating_binary'] = np.where(data['likert_rating']>3, 1, 0)
	data['normalized_machine_judgement'] = normalize(data['machine_judgment'])
	data['machine_judgement_normal'] = data['normalized_machine_judgement']

	precision = precision_score(data['likert_rating_binary'], data['machine_judgement_normal'])
	recall = recall_score(data['likert_rating_binary'], data['machine_judgement_normal'])
	f1 = f1_score(data['likert_rating_binary'], data['machine_judgement_normal'])

	print('Precision: ', precision)
	print('Recall: ', recall)
	print('F1 Score: ', f1)

	print(data)
		
	# Define the bins
	# bins = [1, 2, 3, 4, 5]

	# # Create a new column 'likert_rating_binned' in the DataFrame that contains the binned likert ratings
	# data['likert_rating_binned'] = pd.cut(data['likert_rating'], bins)

	# # Get the value counts for each bin
	# likert_rating_counts = data['likert_rating_binned'].value_counts().sort_index()

	# # Plot the histogram
	# plt.figure(figsize=(10, 6))
	# likert_rating_counts.plot(kind='bar', edgecolor='black')

	# # Set the title and labels
	# plt.title('Distribution of Likert Ratings')
	# plt.xlabel('Likert Rating')
	# plt.ylabel('Number of Forms')

	# Show the plot
	# plt.show()
	
	pearsoncorr, p = pearsonr(data['likert_rating_binary'], data['machine_judgement_normal'])
	print('Pearson correlation: %.3f' % pearsoncorr)
	print('Pearson  pvalue: %.3f' % p)

	# 
	spearmancorr, s = spearmanr(data['likert_rating_binary'], data['machine_judgement_normal'])
	print('Spearman correlation: %.3f' % spearmancorr)
	print('Spearman pvalue: %.3f' % s)

	kendalltaucorr, k = kendalltau(data['likert_rating_binary'], data['machine_judgement_normal'])
	print('Kendall correlation: %.3f' % kendalltaucorr)
	print('Kendall pvalue: %.3f' % k)

	# fig = ggplot.scatterplot(data=data, x="machine_judgment", y="likert_rating")
	title_text = fm.FontProperties(family="Times New Roman")
	axis_text = fm.FontProperties(family="Times New Roman")
	body_text = fm.FontProperties(family="Times New Roman")

	# Alter size and weight of font objects
	title_text.set_size(16)
	axis_text.set_size(12)
	body_text.set_size(12)

	annotation_pearson = 'Pearson: %.3f' % round(pearsoncorr, 3)
	annotation_spearman = 'Spearman: %.3f' % round(spearmancorr, 3)

	
	
	a = (
		ggplot(data, aes(x='machine_judgement_normal', y='likert_rating_binary')) + 
		geom_point(aes(color='attestedness', shape='attestedness')) + 
		scale_color_brewer(type="qual", palette="Set1") +
		geom_smooth(method='lm', mapping = aes(x='machine_judgement_normal', y='likert_rating_binary'), color = 'black', inherit_aes=False) +
		labs(x='predicted judgment', y='Likert rating') + 
		theme(legend_position=(0.34, 0.8), legend_direction='vertical', legend_title=element_blank(),
		figure_size=(3,5),
		axis_line_x=element_line(size=0.6, color="black"),
		axis_line_y=element_line(size=0.6, color="black"),
		panel_grid_major=element_blank(),
		panel_grid_minor=element_blank(),
		panel_border=element_blank(),
		panel_background=element_blank(),
		plot_title=element_text(fontproperties=title_text),
		text=element_text(fontproperties=body_text),
		axis_text_x=element_text(color="black"),
		axis_text_y=element_text(color="black"),
		) + 
		scale_y_continuous(breaks=np.arange(1, 5.01, 1), 
				limits=[1, 5.1]) +
		scale_x_continuous(breaks=np.arange(0, 1.005, 0.2), 
				limits=[0, 1.005]) +
		geom_text(aes(x=0.5, y = 1.9), family = "Times New Roman", label = annotation_spearman
		)
	)
	a.save('scatterplot_likert_NT.pdf', dpi=400)	

	# data.loc[data['attestedness'] == 'attested', 'bool_judgment'] = 1.0
	# data.loc[data['attestedness'] == 'marginal', 'bool_judgment'] = 0.0
	# data.loc[data['attestedness'] == 'unattested', 'bool_judgment'] = 0.0

	# pearsoncorr, _ = pearsonr(newdata['bool_judgment'], newdata['machine_judgment'])
	# print('Pearsons correlation: %.3f' % pearsoncorr)
	# spearmancorr, _ = spearmanr(newdata['bool_judgment'], newdata['machine_judgment'])
	# print('Spearman correlation: %.3f' % spearmancorr)
	# kendalltaucorr, _ = kendalltau(newdata['bool_judgment'], newdata['machine_judgment'])
	# print('Kentall correlation: %.3f' % kendalltaucorr)

	# b = (
	# 		ggplot(newdata, aes(x='machine_judgment', y='bool_judgment', color='bool_judgment')) + 
	# 		geom_point() 
	# )


	# b.save('scatterplot_binary.pdf', dpi=400)


def visual_judgefile_MaxEnt(humanjudgefile, machine_judgment, MaxEnt=True):
	data = pd.read_csv(humanjudgefile, sep=",", header=0, encoding="utf-8")
	machine_data = pd.read_csv(machine_judgment, sep="\t", header=0, skiprows=2, encoding="utf-8")

	if MaxEnt:
		# data['machine_judgment'] = machine_data['score']
		data['machine_judgment'] = np.exp(-machine_data['score']) * 10e13
		data['form'] = machine_data['word']
		data['machine_judgment'] = normalize(data['machine_judgment'])
		print(data)

	# selected_columns = data[["machine_judgment",'likert_rating','attestedness', "form"]]
	# data = selected_columns.groupby("form").agg({
	# "machine_judgment": "mean",
	# "likert_rating":"mean",
	# "attestedness": "first" # take the first 'attestedness' value encountered for each form
	# }).reset_index()


	pearsoncorr, p = pearsonr(data['likert_rating'], data['machine_judgment'])
	print('Pearson correlation: %.3f' % pearsoncorr)
	print('Pearson  pvalue: %.3f' % p)

	# 
	spearmancorr, s = spearmanr(data['likert_rating'], data['machine_judgment'])
	print('Spearman correlation: %.3f' % spearmancorr)
	print('Spearman pvalue: %.3f' % s)

	kendalltaucorr, k = kendalltau(data['likert_rating'], data['machine_judgment'])
	print('Kendall correlation: %.3f' % kendalltaucorr)
	print('Kendall pvalue: %.3f' % k)

	# Font settings
	title_text = fm.FontProperties(weight='bold', size=10, family="Times New Roman")
	body_text = fm.FontProperties(weight='normal', size=10, family="Times New Roman")


	# Annotation for Spearman correlation
	annotation_spearman = "Spearman correlation: {:.3f}".format(spearmanr(data['machine_judgment'], data['likert_rating']).correlation)

	# ggplot
	plot = (
		ggplot(data, aes(x='machine_judgment', y='likert_rating')) +
		geom_point(aes(color='attestedness', shape='attestedness')) +
		scale_color_brewer(type="qual", palette="Set1") +
		geom_smooth(method='lm', mapping=aes(x='machine_judgment', y='likert_rating'), color='black', inherit_aes=False) +
		labs(title='', x='Judgment', y='Likert rating') +
		theme_bw() +
		theme(
			legend_position='none',
			figure_size=(3, 5),
			axis_line_x=element_line(size=0.6, color="black"),
			axis_line_y=element_line(size=0.6, color="black"),
			panel_grid_major=element_blank(),
			panel_grid_minor=element_blank(),
			panel_border=element_blank(),
			panel_background=element_blank(),
			plot_title=element_text(fontproperties=title_text),
			text=element_text(fontproperties=body_text),
			axis_text_x=element_text(color="black", family="Times New Roman"),
			axis_text_y=element_text(color="black", family="Times New Roman"),
		) + 
		scale_y_continuous(breaks=np.arange(1, 5.01, 1), limits=[1, 5.1]) +
		scale_x_continuous(breaks=np.arange(0, 1.005, 0.2), limits=[0, 1.005]) +
		geom_text(aes(x=0.5, y=1.9), family="Times New Roman", label=annotation_spearman)
	)
	plot.save('scatterplot_likert_MaxEnt.pdf', dpi=400)
	
def visual_judgefile_turkish(machine_judgment):

	data = pd.read_csv(
		machine_judgment,
		sep="\t",
		# header=0,
		names=["word","grammaticality","score"],
		encoding="utf-8")



	data.loc[data['grammaticality'] == 'grammatical', 'bool_judgment'] = 1
	data.loc[data['grammaticality'] == 'ungrammatical', 'bool_judgment'] = 0
	print(data)

	pearsoncorr, _ = pearsonr(data['bool_judgment'], data['score'])
	print('Pearsons correlation: %.3f' % pearsoncorr)
	spearmancorr, _ = spearmanr(data['bool_judgment'], data['score'])
	print('Spearman correlation: %.3f' % spearmancorr)
	kendalltaucorr, _ = kendalltau(data['bool_judgment'], data['score'])
	print('Kentall correlation: %.3f' % kendalltaucorr)

	# fig = ggplot.scatterplot(data=data, x="machine_judgment", y="likert_rating")
	title_text = fm.FontProperties(family="Times New Roman")
	axis_text = fm.FontProperties(family="Times New Roman")
	body_text = fm.FontProperties(family="Times New Roman")

	# Alter size and weight of font objects
	title_text.set_size(16)
	axis_text.set_size(12)
	body_text.set_size(12)

	annotation_pearson = 'Pearson: %.3f' % round(pearsoncorr, 3)
	
	a = (
			ggplot(data, aes(x='score', y='bool_judgment')) + 
			geom_point(aes(color='bool_judgment', shape='bool_judgment')) 
			# + 
			# scale_color_brewer(type="qual", palette="Set1") 
			#+geom_smooth(method='lm', mapping = aes(x='score', y='bool_judgment'), color = 'black', inherit_aes=False) +
			# labs(x='predicted judgment', y='grammaticality') + 
			# theme(legend_position=(0.34, 0.8), legend_direction='vertical', legend_title=element_blank(),
			# figure_size=(3,5),
			# axis_line_x=element_line(size=0.6, color="black"),
			# axis_line_y=element_line(size=0.6, color="black"),
			# panel_grid_major=element_blank(),
			# panel_grid_minor=element_blank(),
			# panel_border=element_blank(),
			# panel_background=element_blank(),
			# plot_title=element_text(fontproperties=title_text),
			# text=element_text(fontproperties=body_text),
			# axis_text_x=element_text(color="black"),
			# axis_text_y=element_text(color="black"),
			# ) + 
		# scale_y_discrete(breaks=np.arange(0, 1.005, 1), 
		#                  limits=[0,1.005]) +
		# scale_x_discrete(breaks=np.arange(0, 1.005, 1), 
		#                  limits=[0, 1.005]) +
		# geom_text(aes(x=0.5, y = 0.5), family = "Times New Roman", label = annotation_pearson)
	)
	a.save('turk_scatterplot_NT.pdf', dpi=400)
	# print(newdata)



if __name__ == '__main__':


	humanJudgement = "data/english/Daland_etal_2011_AverageScores.csv"
	# bilinear_judgment = "data/bilinear_judgment.txt"
	# nelson_judgment = "data/Nelson_model_onset_judgment.txt"
	# categorical_judgment = "result/EnglishJudgement_2023-02-04-18-17-13.txt"
	# categorical_judgment = "result/EnglishJudgement_2023-05-08-08-13-45.txt"
	categorical_judgment = "result/english/EnglishJudgement_2023-06-03-11-01-05.txt"
	# categorical_judgment = 'result/EnglishJudgement_LEXICALGAP_TRUE.txt'
	maxent_judgment = "result/english/EnglishJudgement_MaxEnt_300con-OE1_exceptionful.txt"
	# maxent_judgment = "result/EnglishJudgement_MaxEnt_OE1.txt"

	visual_judgefile(humanJudgement,categorical_judgment)
	# visual_judgefile_MaxEnt(humanJudgement,maxent_judgment)


	# categorical_judgment = 'result/TurkishJudgement_2022-11-21-09-22-39.txt'
	# visual_judgefile_turkish(categorical_judgment)
	# earson correlation: 0.764
	# Spearman correlation: 0.914
	# Kendall correlation: 0.763




	# take the average of two rows?
	# Assign 0 to O 0
	# 1 to O non-zero
	# 2 to everything else