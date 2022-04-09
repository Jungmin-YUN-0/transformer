# transformer-update-
transformer (updated version)


	
[설치필요]

 python -m spacy download en_core_web_sm


 python -m spacy download de_core_news_sm


[실행]
1. preprocess.py
 
	 python preprocess.py -data_dir wmt16 -data_ext csv -data_pkl data.pickle

2. train.py

	python main.py -gpu 1 -option transformer -task train

3. test.py

	python main.py -gpu 1 -option transformer -task test
