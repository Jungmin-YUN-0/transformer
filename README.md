# transformer-update-
transformer (updated version)


	
[설치필요]

 python -m spacy download en_core_web_sm


 python -m spacy download de_core_news_sm


[실행]
1. DATA PREPROCESSING
 
	 python preprocess.py -data_dir wmt16 -data_ext csv -data_pkl data.pickle

2. MAIN (TASK: TRAIN, TEST)
	python main.py -gpu 1 -option transformer -task train
	python main.py -gpu 1 -option transformer -task test
