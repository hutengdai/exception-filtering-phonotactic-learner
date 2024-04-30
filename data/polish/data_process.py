import csv

def convert_onset(onset):
	ortho2_to_ortho1 = {"ń":"ni","dż":"drz", "ż":"rz", "ź":"zi",  "w":"v", "ł": "w", "ch":"x", "ś":"si" }

	for key in ortho2_to_ortho1:
		onset = onset.replace(key, ortho2_to_ortho1[key])
	return onset

def split_string(string):
	trigrams = ["dzi", "drz"]
	bigrams = ["dz", "cz", "ch", "sz", "rz", "ni", "ci", "si", "zi"]
	split_indices = []
	i = 0
	while i < len(string):
		if i < len(string) - 2 and string[i:i+3] in trigrams:
			split_indices.append(i+2)
			i += 3
		elif i < len(string) - 1 and string[i:i+2] in bigrams:
			split_indices.append(i+1)
			i += 2
		else:
			split_indices.append(i)
			i += 1
	if len(split_indices) == 1:
		split_indices = [0, split_indices[0], len(string)-1]
	else:
		split_indices = [-1] + split_indices + [len(string)-1]
	substrings = [string[split_indices[i]+1:split_indices[i+1]+1] for i in range(len(split_indices)-1)]
	substrings.pop()
	return " ".join(substrings)

def generate_testing(file_path, output_path0, output_path1, tokcount_path, novel_index_path):

	with open(file_path, "r", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		with open(output_path0, "w", encoding="utf-8") as new_file0, \
			 open(output_path1, "w", encoding="utf-8") as new_file1, \
			 open(tokcount_path, "w", encoding="utf-8", newline='') as csv_out,\
			 open(novel_index_path, "w", encoding="utf-8", newline='') as csv_novel:
				
			writer = csv.writer(csv_out)
			writer.writerow(['tokcount', 'typc'])
		
			novel_writer = csv.writer(csv_novel)
			novel_writer.writerow(['index', 'novel'])
		
			for i, row in enumerate(reader):
				onset = row["ihead"]
				onset_converted = convert_onset(onset)
				onset_preprocessed = split_string(onset_converted)
				new_line = f"{onset_preprocessed}\t{row['tail']}\t{row['ucla']}\t{row['subj']}\t{row['response']}\t{row['tokcount']}\t{row['typc']}\n"

				tokcount = row['tokcount']
				typc = row['typc']
				writer.writerow([tokcount, typc])

				if row["novel"] == '0':
					new_file0.write(new_line)
				else:
					new_file1.write(new_line)
				
				novel_writer.writerow([i, row["novel"]])



def generate_training(training_path, training_out):
	# Open the input file and read its contents
	with open(training_path, "r") as a:
		lines = a.readlines()

	# Process each line and write the repeated strings to the output file
	with open(training_out, "w") as f:
		for line in lines:
			parts = line.strip().split("\t")
			string = parts[0]
			num_repeats = int(parts[1])
			repeated_string = (string + "\n") * num_repeats
			f.write(repeated_string)

if __name__ == "__main__":
	file_path = "data/polish/all_legit_data.csv"
	output_path_filler = "data/polish/TestingData_filler.txt"
	output_path_novel = "data/polish/TestingData_novel.txt"

	tokcount_path = "data/polish/tokcount_typc.csv"
	novel_index_path = "data/polish/novel_index.csv"

	training_path = "data/polish/LearningData_type.txt"
	training_out = "data/polish/LearningData.txt"
	# with open(file_path, "r", encoding="utf-8") as csv_file:
	# 	reader = csv.DictReader(csv_file)
	# 	with open(output_path, "w", encoding="utf-8") as new_file:
	# 		for row in reader:
	# 			onset = row["ihead"]
	# 			onset_preprocessed = split_string(onset)
	# 			new_line = f"{onset_preprocessed}\t{row['response']}\n"
	# 			new_file.write(new_line)

	generate_testing(file_path, output_path_filler, output_path_novel, tokcount_path, novel_index_path)
	
	# ortho1 (training, feature)
	# 'm', 'w', 'p', 'b', 'f', 'v', 'r', 'l', 'n', 't', 'd', 'c', 'dz', 's', 'z', 'cz', 'drz', 'sz', 'rz', 'ni', 'j', 'ci', 'dzi', 'si', 'zi', 'k', 'g', 'x', 'a', 'e', 'i', 'o', 'u', 'y', 'A', 'E'
	# ortho2 (alllegitdata, IbexHead)
	# lots of errors
	# ipa:
	# replace_dict = {'rz': 'ʐ', 'sz': 'ʂ', 'cz': 'tʂ', 'dzi': 'dʐ', 'si': 'ɕ', 'zi': 'ʑ', 'ni': 'ɲ', 'dz': 'dz',  'w': 'ł', 'v': 'w', 'x':'ch'}


	a = {
		"r rz": "rż",
		"w rz": "łż",
		"m rz": "mż",
		"l ni": "lni",
		"sz p": "szp",
		"sz f": "szw",
		"m n": "mn",
		"g dzi": "gdzi",
		"rz v": "żw",
		"p sz": "psz",
		"g v": "gw",
		"cz k": "czk",
		"g n": "gn",
		"d ni": "dni",
		"s n": "sn",
		"s m": "sm",
		"l j": "lj",
		"x m": "chm",
		"rz m": "żm",
		"m r": "mr",
		"rz r": "żr",
		"rz w": "żł",
		"zi l": "źl",
		"p w": "pł",
		"x r": "chr",
		"m w": "mł",
		"zi w": "źł",
		"cz w": "czł",
		"g l": "gl",
		"j f": "jf",
		"j dz": "jdz",
		"l zi": "lzi",
		"w m": "łm",
		"l cz": "lcz",
		"w r": "łr",
		"m zi": "mzi",
		"m dzi": "mdzi",
		"ni v": "ńw",
		"n p": "np",
		"x si": "chsi",
		"k cz": "kcz",
		"n m": "nm",
		"b g": "bg",
		"si x": "śch",
		"drz m": "dżm",
		"zi m": "źm",
		"f n": "fn",
		"dz ni": "dzni",
		"r w": "rł",
		"dz j": "dzj",
		"cz l": "czl",
		"n w": "nł",
		"rz j": "żj"
	}

	polish_conversion_dict = {a[k]:k for k in a}

	modified_polish_conversion_dict = {}

	for key, value in polish_conversion_dict.items():
		split_key = key.split()
		first_symbol_key = split_key[0]  # first symbol from key
		remaining_key = ' '.join(split_key[1:])  # remaining symbols to form the new key
# = {'rz': 'ż', 'sz': 'sz', 'cz': 'cz', 'dzi': 'dzi', 'si': 'si', 'zi': 'zi', 'ni': 'ɲ', 'dz': 'dz',  'w': 'ł', 'v': 'w', 'x':'ch'}.