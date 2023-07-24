# transformer-update-
transformer (updated version 2022.04.21)


	
## Dependencies

 python -m spacy download en_core_web_sm


 python -m spacy download de_core_news_sm


## Usage


1. DATA PREPROCESSING
 
	 python preprocess.py -data_task [MT / CF] -data_dir [wmt16 / imdb / yelp5 / sst2 / sst5] -data_ext csv -data_pkl [pickleName.pickle]
	 
	 (MT: machine translation, CF: classification)

2. MAIN 

	python main.py -gpu 1 -task [TRAIN / TEST] -data_pkl [pickleName.pickle] -model_save [modelName.pt] -pred_save [predictionName.txt] -data_task [MT / CF]
