Introduction:
	This project aims to preform Question and Answering from Context using Decoder only Transformer (GPT2) pretrained weights loaded from huggingface finetune on Stanford QA dataset with LoRA.


Dataset:
	https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/data


Model Weights:
	https://huggingface.co/openai-community/gpt2


Build: 
	System:
		CPU: Intel i9-13900H (14 cores)
		GPU: NIVIDIA RTX 4060 (VRAM 8 GB)
		RAM: 32 GB

	Configuration:
		CUDA 12.1
		Anaconda 3
		Python = 3.11
		Spyder = 5.4.1


Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	gpt.py
	model_converter.py
	qt_main.py
	summary.py


Credits:
	GPT code is reference from "https://github.com/karpathy/nanoGPT"