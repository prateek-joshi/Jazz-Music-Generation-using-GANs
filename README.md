# Jazz Music Generation using GAN

Dataset used: Jazz songs from GTZAN Dataset (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)<br><br>
preprocess.py generates a spectrogram for each split of a song and stores it in a JSON file.<br>
train.py trains the GAN model on the spectrogram values stored on the JSON file.<br>
```
python .\preprocess.py --data_path=<path\to\songsfolder> --json_path=<path\to\json\save>\filename.json [--num_segments=<total_song_divisions>]
python .\train.py --json_path=<path\to\json\file> [--epochs=<number_of_epochs> --save_weights=<True/False>]
```
