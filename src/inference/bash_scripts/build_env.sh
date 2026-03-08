conda create -n semanticVocoder python==3.11
conda activate semanticVocoder

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt 

pip install lhotse
pip install dasheng
pip install SentencePiece
pip install six
pip install librosa
# for eval
pip install scikit-image
pip install torchlibrosa
pip install ssr_eval --no-deps
pip install laion_clap
pip install resampy
