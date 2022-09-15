# Project 6 Visualization

Purpose of this project:
- Gain a basic appreciation of data science programming and data visualization
- Implement a data science algorithm with given pseudocode
- Be aware of efficiency of the implemented code


In this project we implemented three different methods of visualizing data; PCA, Isomap and t-SNE.


## How to Run
- requires Python 3+
- install dependencies 

```properties
pip install -r requirements.txt
```

### Our datasets
- Swiss roll
- OptDigits 

## PCA

#### PCA with swiss roll
![pca_swiss](https://user-images.githubusercontent.com/113537402/190419082-b2a98b9a-904f-4ddc-8d55-b14e8f15e8db.PNG)

Swiss Roll is reduced in 2 dimensions and we can clearly see the Spiral.

#### PCA with OptDigits 
![pca_optidigits](https://user-images.githubusercontent.com/113537402/190419264-19547e93-6116-4a65-b93a-06f5db714a98.PNG)

The OptDigits dataset contains a lot of data that cant be reduced on 2 dimensions. That explains our poor results. 

## Isomap

#### Isomap swiss
![isomap_swiss](https://user-images.githubusercontent.com/113537402/190419350-235aa999-4a43-444c-a946-1c3a2050a874.PNG)

Using Isomap we see that the swiss roll dataset becomes flat. 

#### Optdigits with Isomap
![isomap_optidigits](https://user-images.githubusercontent.com/113537402/190419520-3bf130a0-2157-4680-8f4f-c57be03acb3f.PNG)

Applying Isomap to Optdigits still does not result in something visible. 

#### Optidigts with t-sne

![tsne_optidigits](https://user-images.githubusercontent.com/113537402/190419612-64be4bba-80c1-40ae-9d55-650ad28fde87.PNG)

When we apply t-sne to the optidigits dataset we can clearly see the different clusters. 




