
# General
Notebook files can be run on a local machine where all the files are located in a specific folder and required packages are already preinstalled. It is also possible, and recommnded, to run them in Google Colab. The notebooks include pip install commands to load the required packages.

- Set the local_mode = True to run in a local machine. The default is False.

# Pre-processing
The [behrt/data_prep.ipynb](/behrt/data_prep.ipynb) notebook takes all the source data (csv files) and generates pickle files that are used for pre-training and disease prediction. The input files are big, even after compression, and we could not add them to github. However, the pickle files are available in the data folder.

# Pre-training
[behrt/berth.ipynb](/behrt/behrth.ipynb): Implements the pre-training and MLM tasks. The notebook can run in Google Colab or a local machine and parameters (hyper-parameters, sample size, file locations, etc.). The number of epochs is defaulted to 5, but the saved model was trained on 50 epochs.

To load the saved model and view the 2D embeddings or the top similar conditons, set the train_model_flag = False.

 # Pre-trained model
 The pre-trained model used for next disease predictions is located under the saved models folder (mlm128.pt).

- Google Colab Link: [https://colab.research.google.com/drive/1wTRr-lAdkPh28xLXNAwv1xD0fQjh8hLL?usp=sharing](https://colab.research.google.com/drive/1wTRr-lAdkPh28xLXNAwv1xD0fQjh8hLL?usp=sharing)

# Next visit prediction
[behrt/behrt_NextVisit.ipynb](/behrt/behrt_NextVisit.ipynb): Implements the next visit disease prediction.

- Google Colab Link: [https://colab.research.google.com/drive/1bhiPx4IfwwHj6sonWzpUcU4MbBNJ-Tiv?usp=sharing](https://colab.research.google.com/drive/1bhiPx4IfwwHj6sonWzpUcU4MbBNJ-Tiv?usp=sharing)

# Next X visit prediction
[behrt/behrt_NextXMonths.ipynb](/behrt/behrt_NextXMonths.ipynb): Implements disease prediction after "x" months from previous visit. A configuration parameter is used to run for 6 or 12 months.

- Google Colab Link: [https://colab.research.google.com/drive/1g0mNgg1SLiLkRoM-8SqQhC-WGbbzXz4H?usp=sharing](https://colab.research.google.com/drive/1g0mNgg1SLiLkRoM-8SqQhC-WGbbzXz4H?usp=sharing)

# Training results
The training results by epoch, and even by group of batches, is available in the /results folder.

# Embedding visualization
The 2D disease map is located here.\

![](/images/embeddings-2d.png)


# Top 20 similar disease codes
**Subacute and chronic vulvitis**\
	 Other specified noninflammatory disorders of vagina, Similarity=0.624294638633728\

**Obstructive and reflux uropathy** \
	 Calculus of kidney and ureter, Similarity=0.6137647032737732  

**Benign neoplasm of colon, rectum, anus and anal canal** \ 
	 Diverticular disease of intestine, Similarity=0.6117430329322815  

**Other diseases of anus and rectum**\
	 Hemorrhoids and perianal venous thrombosis, Similarity=0.6104586720466614

**Superficial injury of head** \
	 Other and unspecified injuries of head, Similarity=0.6079360246658325

**Benign neoplasm of colon, rectum, anus and anal canal** \
	 Hemorrhoids and perianal venous thrombosis, Similarity=0.603598415851593
	 
**Alcohol abuse** \
	 Alcohol dependence, Similarity=0.6028408408164978

**Diverticular disease of large intestine without perforation or abscess** \
	 Polyp of colon, Similarity=0.5936653017997742

**Other and unspecified ovarian cysts** \
	 Leiomyoma of uterus, unspecified, Similarity=0.5918169021606445

**Diverticular disease of large intestine without perforation or abscess**\
	 Other hemorrhoids, Similarity=0.5874903798103333

**Open wound of head** \
	 Other and unspecified injuries of head, Similarity=0.5814423561096191

**Benign neoplasm of colon, rectum, anus and anal canal** \
	 Other diseases of intestine, Similarity=0.5748730897903442

**Other specified disorders of nose and nasal sinuses** \
	 Chronic sinusitis, unspecified, Similarity=0.5707678198814392

**Chronic sinusitis** \
	 Other and unspecified disorders of nose and nasal sinuses, Similarity=0.567510187625885

**Benign prostatic hyperplasia without lower urinary tract symptoms** \
	 Benign prostatic hyperplasia with lower urinary tract symptoms, Similarity=0.5647885203361511


