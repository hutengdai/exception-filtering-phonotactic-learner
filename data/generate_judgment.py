vowels = {
	'i': {'high': True, 'back': False, 'round': False},
	'y': {'high': True, 'back': False, 'round': True},
	'ɯ': {'high': True, 'back': True, 'round': False},
	'u': {'high': True, 'back': True, 'round': True},
	'e': {'high': False, 'back': False, 'round': False},
	'ø': {'high': False, 'back': False, 'round': True},
	'ɑ': {'high': False, 'back': True, 'round': False},
	'o': {'high': False, 'back': True, 'round': True},
}

def check_back_harmony(v1, v2):
	return vowels[v1]['back'] == vowels[v2]['back']

def check_round_harmony(v1, v2):
	return not (vowels[v2]['high'] and vowels[v1]['round'] != vowels[v2]['round'])

def check_mid_round_vowel(v2):
	return not (vowels[v2]['high'] == False and vowels[v2]['round'])

def is_harmonic(v1, v2):
	return check_back_harmony(v1, v2) and check_round_harmony(v1, v2)#and check_mid_round_vowel(v2)

def generate_words(tier):
	words = [('t' + ' ' + v1 + ' ' + 'p' + ' ' + v2) for v1 in tier for v2 in tier]
	harmonic_words = [word + "\tgrammatical" for word in words if is_harmonic(word[2], word[6])]
	non_harmonic_words = [word + "\tungrammatical" for word in words if not is_harmonic(word[2], word[6])]
	return harmonic_words, non_harmonic_words

harmonic_words, non_harmonic_words = generate_words(vowels.keys())

write_nonce = open("nonce.txt", "w", encoding='utf8')
for x in harmonic_words:
	write_nonce.write(x + '\n')
for y in non_harmonic_words:
	write_nonce.write(y + '\n')

print(harmonic_words)
print(harmonic_words)