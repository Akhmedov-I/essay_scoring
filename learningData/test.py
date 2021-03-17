import docx2txt
import nltk

nltk.download()

result = docx2txt.process('/Users/islombekakhmedov/Documents/Programming training/data_science/text_identifier/essay.docx')



tokens = nltk.word_tokenize(result)

text = nltk.Text(tokens)

tags = nltk.pos_tag(text)

print(tags)