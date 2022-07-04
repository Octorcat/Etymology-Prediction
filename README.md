# Etymology prediction

## Live website

See [Etymology prediction live website](https://etymology-classifier.herokuapp.com/).

## Character-sequence based English word etymology prediction

This project aims to predict the etymology of an English word while showcasing the whole process end-to-end.

- Data collection:
  + The training data was scraped from Wiktionary. 
- Machine Learning:
  + It uses a character-level many-to-one RNN.
- Server:
  + It is REST API that runs the predicts a word's etymology through the endpoin GET /etymology/{word}
- Client:
  + It is performant lightweight reactive UI that connects with the etymolyg prediction server.
  
 ## Tech stack used in this project (all is in this repo)

- Date collection:
    + wiktionaryparser (Python)
- Machine Learning :
    + Pytorch
- Server-side:
    + Flask
- Client-side:
    + Svelte (ts)
 
## Data collection
The training data was scraped from the etymology section of Wiktionary using wiktionaryparser. 

Since this etymology section is presented in plain text, the actual etymology labels for training must be extracted. For simplicity sake, I only consider two possible etymologies: germanic and latin. This is, of course, a big oversimplication of the etymology of English words; but I thought that it could yield useful results nonetheless. I scraped the etymology of the words contained in the CMU dictionary.

The raw data collected is under [/collected_etymology_dict.json](https://github.com/FrederickRoman/etymology-predictor/blob/main/machine_learning/preprocessing/CMU_source_dict.json)

If you want to rerun the data collection process (which may yield different results since wiktionary may have changed), run:

```
pip install -r requirements.txt
python machine_learning/preprocessing/web_scrape.py

```
## Machine Learning
For etymology prediction, I used a many-to-one RNN based on the Pytorch example found in the official website. All of the training, can be found under [/train.ipynb](https://github.com/FrederickRoman/etymology-predictor/blob/main/machine_learning/train.ipynb)

Loss over iterations

<div style="display:flex; justify-content:center; align-items:center;">
    <img src="https://github.com/FrederickRoman/etymology-predictor/blob/main/docs/ml/loss_over_iterations.png" height="300" alt="Loss over iterations"/>
</div>

Confusion matrix

<div style="display:flex; justify-content:center; align-items:center;">
    <img src="https://github.com/FrederickRoman/etymology-predictor/blob/main/docs/ml/confusion_matrix.png" height="300" alt="Loss over iterations"/>
</div>

## Server
The prediction of the etymology of a word is offered through a REST API.

To run the API (with cmd) on http://localhost:5000/etymology/{word}

```
pip install -r requirements.txt
cd server
set FLASK_APP=server
flask run
```
To see the API swagger documentation go to http://localhost:5000/doc

<div style="display:flex; justify-content:center; align-items:center;">
    <img src="https://github.com/FrederickRoman/etymology-predictor/blob/main/docs/server/api_swagger.png" height="900" alt="Loss over iterations"/>
</div>

## Client
The prediction of the etymology of a word can also be done through an interactive UI. To run it, start the server then go to http://localhost:5000

<div style="display:flex; justify-content:center; align-items:center;">
    <img src="https://github.com/FrederickRoman/etymology-predictor/blob/main/docs/client/client_UI.png" height="600" alt="Loss over iterations"/>
</div>

### Project's client setup

```
cd client
npm install
```

#### Compiles and hot-reloads

```
npm run dev
```

#### Builds for production

```
npm run build
```
## Acknowledgements
#### The etymology prediction model was adapted from NLP From Scratch: Classifying Names with a Character-Level RNN 
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 


