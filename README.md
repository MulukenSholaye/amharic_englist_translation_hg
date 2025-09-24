# Amharic-English Machine Translation Comparator Project 
![Blogger](https://img.shields.io/badge/Blogger-FF5722?style=for-the-badge&logo=blogger&logoColor=white)
![Dev.to blog](https://img.shields.io/badge/dev.to-0A0A0A?style=for-the-badge&logo=dev.to&logoColor=white)
![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)
![Firefox](https://img.shields.io/badge/Firefox-FF7139?style=for-the-badge&logo=Firefox-Browser&logoColor=white)
![Google Chrome](https://img.shields.io/badge/Google%20Chrome-4285F4?style=for-the-badge&logo=GoogleChrome&logoColor=white)
![GitLab CI](https://img.shields.io/badge/gitlab%20ci-%23181717.svg?style=for-the-badge&logo=gitlab&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)

## Overview
This project is a machine translation application designed to compare the performance of various transformer-based models for translating text from Amharic to English. The application provides a user-friendly interface to input Amharic text, receive translations from multiple models simultaneously, and visualize their performance metrics. 
			The goal is to identify the most effective and efficient transformer architecture for this specific language pair.

![dsfd](https://github.com/MulukenSholaye/amharic_englist_translation_hg/blob/c059ec3a47896bfe205583539712b9f537a4c4d4/Screenshot%20from%202025-09-24%2019-51-40.png)

# Key Features
* Multi-Model Translation: Translate a single Amharic text snippet using several different transformer models.
* Performance Comparison: View and compare key metrics for each translation, such as BLEU score, model latency, and other qualitative assessments.
* Intuitive UI: A clean and responsive user interface for easy text input, model selection, and result viewing.
* Extensible Architecture: Designed to easily integrate new transformer models for future comparisons.ModelsThe application is built to support a variety of transformer models. Initial models included in this project are:Base 
*  Transformer Model: A standard, vanilla transformer architecture trained from scratch on the Amharic-English dataset.Pre-trained Transformer Model (e.g., mBART-50): 
* A multilingual pre-trained model fine-tuned for the Amharic-English language pair.
* Distilled Transformer Model: A smaller, more efficient version of a larger transformer model, optimized for faster inference with minimal loss in accuracy.DatasetThe models are trained and evaluated on a custom-curated Amharic-English parallel corpus. The dataset consists of parallel sentences sourced from various domains to ensure a broad coverage of vocabulary and grammar. The dataset is split into training, validation, and test sets to facilitate robust model training and evaluation.Getting StartedFollow these instructions to get a copy of the project up and running on your local machine.
# Prerequisites
* Python 3.8 or laterpip package managerInstallationClone the repository:git clone [https://github.com/your-username/amharic-english-translation.git](https://github.com/your-username/amharic-english-translation.git)
* 		cd amharic-english-translation
* Create a virtual environment (recommended):python -m venv venv
* 		source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
* Install the required Python packages:pip install -r requirements.txt
* 		Download the pre-trained models and datasets (instructions to be provided in a separate file or script).UsageStart the application:python main.py
		* Open your web browser and navigate to http://127.0.0.1:5000 (or the address shown in the terminal).Enter the Amharic text you wish to translate into the input box.Click "Translate" to see the output from each of the configured models. The results page will display the translated text and performance metrics for each model.Project Structure.
		*			├── python_files/                     # Trained model checkpoints and tokenizer python files
		*			├── htmlfiles/                       # Dataset files (e.g., amharic-en-corpus.txt)
					├--
  		```python
			def run_models(self, model):
  			self.model = model
		```

# We welcome contributions!
* Please feel free to open an issue or submit a pull request. For major changes, please open an issue first to discuss the proposed changes.LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
* ![](https://github.com/MulukenSholaye/amharic_englist_translation_hg/blob/793ac1c73145ba16ac25862ce406057719f98e2d/Screenshot%20from%202025-09-24%2019-38-18.png)
* ![](https://github.com/MulukenSholaye/amharic_englist_translation_hg/blob/19c067289de13a1f88370f2ee709151c5ab53270/Screenshot%20from%202025-09-24%2019-51-40.png)
* ![](https://github.com/MulukenSholaye/amharic_englist_translation_hg/blob/b114654b40fb8836f3bee75d625ffabb6c2c6839/Screenshot%20from%202025-09-24%2019-43-05.png)
