# CS_assignment
The main focus of this assignment is on producing a scalable solution for product duplicate detection. 
This  means  that  coming  up  with  a  very  complex  duplicate  detection  algorithm  and  a  sophisticated 
similarity measure is less important here than coming up with an approach that scales well given the 
large number of products and pairs of these that need to be considered. By scaling well it is meant to 
have a slightly less effective method while improving greatly on its efficiency. 
Using LSH as a pre-selection of candidate pairs before applying the Multi-component Similarity Method (MSM). 
The code is divided into 6 parts:
Part 1 deals with loading the data and feature extraction of product codes, brands, screen sizes, refresh rates and resolutions. Creates binary vector representation of products.<br />
Part 2 deals with the LSH step: minhashing, hashing and identifying candidate pairs<br />
Part 3 deals with the Multi-component Similarity Method<br />
Part 4 calculates the pair completeness. pair quality and $F_{1}^{*}$-measure after LSH.<br />
Part 5 performs hierarchical clustering by means of the python package SciPy<br />
Part 6 calculates the $F_{1}$-measure after MSM
