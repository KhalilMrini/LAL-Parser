pip install torch==1.1.0
pip install cython==0.29.13 
pip install numpy==1.17.2
sudo apt-get install libhdf5-serial-dev=1.8.16+docs-4ubuntu1.1
pip install benepar[gpu]==0.1.2 --ignore-installed
pip install pytorch_pretrained_bert==0.6.2
pip install sentencepiece==0.1.83
pip install tensorflow==2.0.0
pip install tensorboard==2.0.0
pip install nltk==3.5
python3 -m nltk.downloader punkt averaged_perceptron_tagger
pip install git+https://github.com/lanpa/tensorboardX
pip install transformers==2.8.0

# Download pre-trained model
pip install gdown
gdown https://drive.google.com/uc?id=1LC5iVcvgksQhNVJ-CbMigqXnPAaquiA2

# If you get a "Too many users have viewed or downloaded this file recently." error, 
# run the following 3 lines to download pre-trained model from a mirror:
# pip install internetarchive
# ia download neuraladobe-ucsdparser
# mv neuraladobe-ucsdparser/best_parser.pt .
