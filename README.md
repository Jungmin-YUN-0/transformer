# transformer-update-
transformer (updated version)


	
[설치필요]

 python -m spacy download en_core_web_sm


 python -m spacy download de_core_news_sm


[실행]
1. DATA PREPROCESSING
 
	 python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb] -data_ext csv -data_pkl [data_wmt16.pickle / data_imdb.pickle]
	 
	 (MT is machine translation, CF is classification)

2. MAIN (TASK: TRAIN, TEST)

	python main.py -gpu 1 -option [BASE / LR / CT 중에서 선택] -task [TRAIN / TEST 중에서 선택]	
	
	(LR is for low-rank attention, CT is for core-token attention)
