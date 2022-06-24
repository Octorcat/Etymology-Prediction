# Etymology prediction


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
