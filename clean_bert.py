def main():

	with open('datasets_507677_938093_vocab.txt') as f:
		lines = f.readlines()

	tokens = []

	for i in lines:
		if '#' in i:
			continue
		if i.startswith('['):
			continue
		tokens.append(i)

	print(tokens, len(tokens))
	with open('bert_vocab.txt', 'w') as f:
		f.write(''.join(tokens))


main()