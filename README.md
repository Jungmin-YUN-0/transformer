# transformer-update-
transformer (updated version 2022.04.21)


	
[설치필요]

 python -m spacy download en_core_web_sm


 python -m spacy download de_core_news_sm


[실행]
1. DATA PREPROCESSING
 
	 python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb] -data_ext csv -data_pkl [pickleName.pickle]
	 
	 (MT is machine translation, CF is classification)

2. MAIN 

	python main.py -gpu 1 -option [BASE / LR / CT] -task [TRAIN / TEST] -data_pkl [pickleName.pickle] -model_save [modelName.pt]  
	-pred_name [predictionName.txt]
	
	(BASE is for vanilla transformer, LR is for low-rank attention(linformer), CT is for core-token attention(my model))
	
	
![figure](https://user-images.githubusercontent.com/76892989/164385253-f8a784bd-df55-4ac1-a3d8-d34c80c321f2.png)
