#!/usr/bin/env python
# coding: utf-8

# In[209]:


import json 
import pandas as pd
import numpy as np
from random import shuffle
import shlex
import re
import itertools
from collections import Counter
from math import sqrt
import textdistance
import random
import regex as re
from collections import Counter

#################################################################################################
#Part1

with open(r"C:\Users\Vanja\Downloads\TVs-all-merged\TVs-all-merged.json") as f:
    data = json.load(f)

data_dict = {} #empty dict to fill with data from products where product codes that have duplicates are split in seperate keys

for product_code in data: #loop through the entire dictionary
    if len(data[product_code]) >1:
        data_dict[product_code] = data[product_code][0] #check which keys have a list with length greater than 1. Only split these
        for i in range(1, len(data[product_code])):
            data_dict[product_code + f"_duplicate_{i}"] = data[product_code][i] #add "unique" key to product duplicate by adding the string "_duplicate"
    else:
        data_dict[product_code] = data[product_code][0]
        


#Used for bootstrapping 63% of the data
empty_list = []
for key in data_dict:
    empty_list.append(key)

fraction_data = int(63 / 100 * len(data_dict))

rand = random.sample(empty_list,fraction_data)

data_dict2 = {}
for key in rand:
    data_dict2[key] = data_dict[key]
        
data_dict = data_dict2




#feature extraction of product codes, brands, screen sizes, refresh rates and resolutions 

extracted_codes = []

for product in data_dict:
    title = [data_dict[product]["title"]]
    for i in title:
        for x in i.split(' '):
            if x.isalnum() and not x.isalpha() and not x.isdigit():
                extracted_codes.append(x)

brands = []
for product_code in data: #loop through the entire dictionary
    
    all_product_features = data[product_code][0]['featuresMap'] #access "featuresMap" of every product
    
    if "Brand" in all_product_features: #see if the feature "Brand" exists in the featuresMap dict
        brands.append(all_product_features.get("Brand")) 
        all_product_features.update({'Brand': all_product_features.get('Brand') })
    elif "Brand Name" in all_product_features: #see if the feature "Brand Name" exists in the featuresMap dict
        brands.append(all_product_features.get("Brand Name"))


screen_sizes = []

for product_code in data: #loop through the entire dictionary
    
    all_product_features = data[product_code][0]['featuresMap'] #access "featuresMap" of every product
    
    if "Screen Size" in all_product_features: #see if the feature "Screen Size" exists in the featuresMap dict
        
        value = shlex.split(all_product_features.get("Screen Size"), posix= False)[0]
        #value = re.findall(r'\d+(?:\.\d+)?', value)[0]
        screen_sizes.append(value)
        all_product_features.update({'Screen Size': f"{value}"})
        
    elif "Screen Size Class" in all_product_features: #see if the feature "Screen Size Class" exists in the featuresMap dict
        value = shlex.split(all_product_features.get("Screen Size Class"), posix= False)[0]
        #value = re.findall(r'\d+(?:\.\d+)?', value)[0]
        screen_sizes.append(value)
        all_product_features.update({'Screen Size': f"{value}"})


        
refresh_rates = []
for product_code in data: #loop through the entire dictionary
    
    all_product_features = data[product_code][0]['featuresMap'] #access "featuresMap" of every product
    
    if "Refresh Rate" in all_product_features: #see if the feature "Refresh Rate" exists in the featuresMap dict
        refresh_rates.append(all_product_features.get("Refresh Rate")) 
        all_product_features.update({'Refresh Rate': all_product_features.get('Refresh Rate') })
    elif "Screen Refresh Rate" in all_product_features: #see if the feature "Brand Name" exists in the featuresMap dict
        refresh_rates.append(all_product_features.get("Screen Refresh Rate"))
        #all_product_features.update({'Brand': all_product_features.get('Brand')})
    elif "Enhanced Refresh Rate" in all_product_features:
        refresh_rates.append(all_product_features.get("Enhanced Refresh Rate"))
    elif "Standard Refresh Rate" in all_product_features:
        refresh_rates.append(all_product_features.get("Standard Refresh Rate"))


resolutions = []
for product_code in data: #loop through the entire dictionary
    
    all_product_features = data[product_code][0]['featuresMap'] #access "featuresMap" of every product
    
    if "Vertical Resolution" in all_product_features: #see if the feature "Brand" exists in the featuresMap dict
        resolutions.append(all_product_features.get("Vertical Resolution")) 
        all_product_features.update({'Vertical Resolution': all_product_features.get('Vertical Resolution') })
    elif "Recommended Resolution" in all_product_features: #see if the feature "Brand Name" exists in the featuresMap dict
        resolutions.append(all_product_features.get("Recommended Resolution"))



def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

a = Diff(extracted_codes, refresh_rates)
b = Diff(a,resolutions)

        
codes = list(filter(('3D').__ne__, b))
codes = b[0:1139]

to_delete = ['40in','120hz', '2K','2160p', '120HZ',  '1080P','720P', 'cdm2', '70inch', '120CMR', '240hz', '230Hz']
for i in to_delete:
    if i in codes:
        codes.remove(i)

Brands = list(dict.fromkeys(brands))
aces = [word + "," for word in Brands]
Brands = Brands+aces
Screen_Sizes = list(dict.fromkeys(screen_sizes))
Refresh_Rates = list(dict.fromkeys(refresh_rates))
Resolutions = list(dict.fromkeys(resolutions))
Codes = list(set(codes))


#create binary vector representation of products

obj_brands = {}

for product_code in data_dict: #loop through the entire dictionary
    
    title = data_dict[product_code]["title"]
    
    obj_brands[product_code] = [] #create list for every product 
        
    for i in Brands: #loop through all brands in Brands list
        if i in title.split(" "):
            obj_brands[product_code].append(1)   
        else:
            obj_brands[product_code].append(0)

obj_screens = {}

for product_code in data_dict: #loop through the entire dictionary
    
    title = data_dict[product_code]["title"]
    
    obj_screens[product_code] = [] #create list for every product 
        
    for i in Screen_Sizes: #loop through all brands in Brands list
        if i in title.split(" "):
            obj_screens[product_code].append(1)   
        else:
            obj_screens[product_code].append(0)


obj_refresh_rates = {}

for product_code in data_dict: #loop through the entire dictionary
    
    title = data_dict[product_code]["title"]
    
    obj_refresh_rates[product_code] = [] #create list for every product 
        
    for i in Refresh_Rates: #loop through all brands in Brands list
        if i in title.split(" "):
            obj_refresh_rates[product_code].append(1)   
        else:
            obj_refresh_rates[product_code].append(0)

obj_resolutions = {}

for product_code in data_dict: #loop through the entire dictionary
    
    title = data_dict[product_code]["title"]
    
    obj_resolutions[product_code] = [] #create list for every product 
        
    for i in Resolutions: #loop through all brands in Brands list
        if i in title.split(" "):
            obj_resolutions[product_code].append(1)   
        else:
            obj_resolutions[product_code].append(0)
            

obj_codes = {}
for product_code in data_dict: #loop through the entire dictionary
    
    title = data_dict[product_code]["title"]
    obj_codes[product_code] = []
    
    for i in Codes:
        if i in title.split(" "):
            obj_codes[product_code].append(1)   
        else:
            obj_codes[product_code].append(0)

def merge_dicts(dict1, dict2):
    #function merges the binary vectors for all products
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[key] = value
    merged_dict = dict1
    return merged_dict

dict_brand_screen = merge_dicts(obj_brands, obj_screens)
dict_refresh_resolutions = merge_dicts(obj_refresh_rates, obj_resolutions)

dict_all = merge_dicts(dict_brand_screen, dict_refresh_resolutions)
dict_all = merge_dicts(dict_all, obj_codes)

all_lists = Brands + Screen_Sizes + Refresh_Rates + Resolutions + Codes


# In[136]:





# In[137]:





# In[138]:





# In[139]:





# In[210]:


#End Part 1
#################################################################################################

#################################################################################################
#Part 2

def create_hash_function(size:int):
    # function that creates hash vector
    hash_ex = list(range(1, len(all_lists)+1))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_function(brand_size:int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for x in range(nbits): #we don't care about the iterator value, just that it should run some specific number of times"
        hashes.append(create_hash_function(brand_size))
    return hashes

minhash_function = build_minhash_function(len(all_lists), 100) #Create 200 minhash vectors for example


def create_hash(binary_vector:list):
    #function that creates signatures
    signature = []
    for func in minhash_function:
        for i in range(1, len(all_lists)+1):
            element = func.index(i)
            sign_value = binary_vector[element]
            if sign_value == 1:
                signature.append(element)
                break
    return signature


obj_sign = {}

for product_code in data_dict: #loop through the entire dictionary
    obj_sign[product_code] = create_hash(dict_all[product_code]) #assign to each product code the corresponding signature values

    
to_delete = []
for key in obj_sign:       # check which product code gives empty lists
    if len(obj_sign[key]) == 0:
         to_delete.append(key)

for i in to_delete: #loop through products to delete and delete them from dictionary obj_sign
    del obj_sign[i]
    
    
def split_vector(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature)/b)
    
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i : i+r])
    return subvecs


a = list(itertools.combinations(obj_sign, 2)) # make pairs of all product codes once
list_code_pairs = [] 
for i in a:
    list_code_pairs.append(list(i)) #convert the pairs into lists 


similar_pairs = []

for code_pair in list_code_pairs:
    #print(len(obj_sign[code_pair[0]]))

    band = split_vector(obj_sign[code_pair[0]], 50)
    #print(band)

    band_1 = split_vector(obj_sign[code_pair[1]], 50)
    #print(band_1)
    
    counter = 0
    for i in band:
        for j in band_1:
            if i == j:
                counter += 1
                
    if counter >= 1:
        similar_pairs.append([code_pair[0], code_pair[1]])
        
#End Part 2
#################################################################################################

#################################################################################################
#Part 3 

def qgram(string1, string2):
    num = 3
    split_string1_list = [string1[i:i+num] for i in range(0, len(string1), num)] 
    split_string2_list = [string2[i:i+num] for i in range(0, len(string2), num)]
    common = [] 
    cnt = 0 
    for i in split_string1_list:
        for j in split_string2_list:
            if i == j:
                cnt += 1
                common.append(i)
    return cnt/len(split_string1_list)

Brands = list(dict.fromkeys(brands))
model_words = Brands + Screen_Sizes + Refresh_Rates + Resolutions


def cosine_similarity(product1, product2, ndigits):
    
    #count characters in every product name
    count_characters1 = Counter(product1)
    count_characters2 = Counter(product2)
    
    #Get the characters in every product name
    set_characters1 = set(count_characters1)
    set_characters2 = set(count_characters2)
    
    length1 = sqrt(sum(c*c for c in count_characters1.values()))
    length2 = sqrt(sum(c*c for c in count_characters2.values()))
    
    #Get the common characters between the two character sets
    common_characters = set_characters1.intersection(set_characters2)

    # Sum of the product of each intersection character
    product_summation = sum(count_characters1[character] * count_characters2[character] for character in common_characters)

    # Gets the length of each vector
    length = length1 * length2
    
    similarity = round(product_summation/length, ndigits)

    return similarity


def jaccard(x,y):
    return len(x.intersection(y))/len(x.union(y))

def TMWMsim(product1, product2, alpha, beta):
    #cosine similarity of 2 product names
    
    #cosine similarity of 2 product names
    cos_sim = cosine_similarity(product1, product2,2)
    title_sim = 1
    
    product1_name = product1.split('_')
    product2_name = product2.split('_')
    
    title1 = data_dict[product1].get("title").split()
    title2 = data_dict[product2].get("title").split()
    
    if product1_name[0] == product2_name[0]:
        title_sim = 1
    #else:
        #title_sim = -1
    
    #if product names are not similar enough
    elif cos_sim < alpha:
        mwPerc = jaccard(set(title1),set(title2))
        if mwPerc > beta:
            title_sim = mwPerc
        else:
            title_sim = -1
    else:
        title_sim = 1 
            
    return title_sim


gamma = 0.756
alpha = 0.90 
beta = 0.0
mu = 0.650

similar_pairs_and_distance = []

for cand_pair in similar_pairs:
        
    if data_dict[cand_pair[0]]['shop'] == data_dict[cand_pair[1]]['shop']:
        distance = np.inf
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
    elif data_dict[cand_pair[0]]['featuresMap'].get("Brand") != None and data_dict[cand_pair[1]]['featuresMap'].get("Brand") != None and data_dict[cand_pair[0]]['featuresMap'].get("Brand") != data_dict[cand_pair[1]]['featuresMap'].get("Brand"):
        distance = np.inf
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
    elif data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != None and data_dict[cand_pair[1]]['featuresMap'].get("Brand Name") != None and data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != data_dict[cand_pair[1]]['featuresMap'].get("Brand Name"):
    #data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != data_dict[cand_pair[1]]['featuresMap'].get("Brand Name"):
        distance = np.inf
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
    elif data_dict[cand_pair[0]]['featuresMap'].get("Brand") != None and data_dict[cand_pair[1]]['featuresMap'].get("Brand Name") != None and data_dict[cand_pair[0]]['featuresMap'].get("Brand") != data_dict[cand_pair[1]]['featuresMap'].get("Brand Name"):

    #data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != data_dict[cand_pair[1]]['featuresMap'].get("Brand"):
        distance = np.inf
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
    elif data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != None and data_dict[cand_pair[1]]['featuresMap'].get("Brand") != None and data_dict[cand_pair[0]]['featuresMap'].get("Brand Name") != data_dict[cand_pair[1]]['featuresMap'].get("Brand"):
        distance = np.inf
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
    else:
        sim = 0
        avgsim = 0
        number_matches = 0
        weight_matches = 0
        
        keys_of_product1 = set(data_dict[cand_pair[0]]['featuresMap'].keys())
        keys_of_product2 = set(data_dict[cand_pair[1]]['featuresMap'].keys())
        #print(keys_of_product1)
        
        to_remove1 = []
        to_remove2 = []
        
        for key1 in keys_of_product1:
            for key2 in  keys_of_product2:
                
                keysim = qgram(key1,key2)
                
                if keysim > gamma: #gamma
                    valueSim = qgram(data_dict[cand_pair[0]]['featuresMap'][key1], data_dict[cand_pair[1]]['featuresMap'][key2])
                    
                    weight = keysim
                    
                    sim = sim + weight*valueSim
                    
                    number_matches =number_matches + 1
                    
                    weight_matches = weight_matches + weight
                    
                    #to_remove1 = []
                    to_remove1.append(key1)
                    #keys_of_product1.remove(key1)
                    #to_remove2 = []
                    to_remove2.append(key2)
                    #keys_of_product2.remove(key2)
                    
        
        for i in to_remove1:
            if i in keys_of_product1:
                keys_of_product1.remove(i)
                
        for i in to_remove2:
            if i in keys_of_product2:
                keys_of_product2.remove(i)
                        
        if weight_matches > 0:
            avgSim = sim/weight_matches
        
        
        mw1 = []
        for i in keys_of_product1:
            mw1.append(data_dict[cand_pair[0]]['featuresMap'].get(i).split())
            
        
        mw2 = []
        for i in keys_of_product2:
            mw2.append(data_dict[cand_pair[1]]['featuresMap'].get(i).split())
                
        flat_list1 = [item for sublist in mw1 for item in sublist]
        flat_list2 = [item for sublist in mw2 for item in sublist]
        
        if len(flat_list1) != 0 and len(flat_list2) != 0: 
            mwPerc = (len(set(flat_list1).intersection(set(flat_list2))))/max(len(flat_list1),len(flat_list2))
            
        titleSim = TMWMsim(cand_pair[0], cand_pair[1], alpha, beta)
        
        if titleSim == -1:
            theta_1 = number_matches/min(len((data_dict[cand_pair[0]]['featuresMap'].keys())), len((data_dict[cand_pair[1]]['featuresMap'].keys())))
            theta_2 = 1 - theta_1
            
            hsim = theta_1*avgSim + theta_2*mwPerc
            
        else:
            theta_1 = (1 - mu)*number_matches/min(len((data_dict[cand_pair[0]]['featuresMap'])), len((data_dict[cand_pair[1]]['featuresMap'])))
            theta_2 = 1 - mu - theta_1
            
            hSim = theta_1*avgSim + theta_2*mwPerc + mu*titleSim
        
        distance = 1 - hSim
        cand_pair.append(distance)
        similar_pairs_and_distance.append(cand_pair)
        
#End Part 3
#################################################################################################


# In[ ]:





# In[203]:





# In[204]:


#################################################################################################

#################################################################################################
#Part 4

all_modelid = []
for product in data_dict:
    all_modelid.append(data_dict[product]["modelID"])

counts = Counter(all_modelid)
print(counts)
unique_modelID = list(set(all_modelid))

total_duplicates = 0    
for i in unique_modelID:
    if counts[i] >= 2:
        total_duplicates += counts[i]
print(total_duplicates) #total number of duplicates in data_dict 

duplicates_found = 0
for i in similar_pairs:
    if data_dict[i[0]]["modelID"] == data_dict[i[1]]["modelID"]:
        if "duplicate" in i[0].split('_') and 'duplicate' in i[1].split('_'):
            duplicates_found += 2
        else:
            duplicates_found += 1
                
print(len(similar_pairs)) #total number of comparisons made
print(duplicates_found) #total number of duplicates found

pq = duplicates_found/len(similar_pairs)
print(f"pq: {pq}")


pc = duplicates_found/total_duplicates
print(f"pc: {pc}")

f1_star = 2*pc*pq/(pc + pq)
print(f"f1_star: {f1_star}")

#End Part 4
#################################################################################################
print(total_duplicates)
print(duplicates_found)


# In[ ]:





# In[205]:


#################################################################################################
#Part 5

from scipy.cluster import hierarchy
from scipy.spatial import distance
import urllib.request



for i in similar_pairs_and_distance:
    if i[2] == np.inf:
        i[2] = 1000

empty1= []

empty2 = []

for i in similar_pairs:
    empty1.append(i[0])
    empty2.append(i[1])
    
empty3 = empty1 + empty2

empty3 = list(set(empty3)) #lijst met alle producten

df = pd.DataFrame(np.full((len(empty3),len(empty3)), 1000), columns=empty3, index=empty3)

df = df.astype(float)

for j in empty3:
    df.at[j, j] = 0

for i in similar_pairs_and_distance:
    df.at[i[0], i[1]] = i[2]
    df.at[i[1], i[0]] = i[2]
    

df.replace(np.inf, 1000)

qw = distance.squareform(df)
print(qw)

threshold = 0.30
linkage = hierarchy.linkage(qw, method="single")
clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")


# In[206]:



final = []

#remove clusters of size 1 as those certainly don't contain duplicates
for cluster in range(1, len(set(clusters)) + 1):
    duplicates = []
    L = [i for i,val in enumerate(clusters) if val==cluster]

    for i in L:
        duplicates.append(empty3[i])
        
    final.append(duplicates)
        

final_final = []
for i in final:
    if len(i) != 1:
        final_final.append(i)

#End Part 5       
#################################################################################################


# In[207]:


#length = []
#pp = []

#for i in final_final:
#    length.append(len(i))
    
#print(length)


# In[208]:


from collections import Counter


#################################################################################################
#Part 6

all_modelid = []
for product in data_dict:
    all_modelid.append(data_dict[product]["modelID"])

counts = Counter(all_modelid)

unique_modelID = list(set(all_modelid))
    
total_duplicates = 0    
for i in unique_modelID:
    if counts[i] > 1:
        total_duplicates += counts[i]

duplicates_found_cluster = 0

for cluster_i in final_final:
    modelID_from_cluster = []
    for product_i in cluster_i:
        modelID_from_cluster.append(data_dict[product_i]["modelID"])
        
    counts_cluster = Counter(modelID_from_cluster)
    unique_modelID_cluster = list(set(modelID_from_cluster))
    
    for i in unique_modelID_cluster:
        if counts_cluster[i] > 1:
            duplicates_found_cluster += counts_cluster[i]
        
print(duplicates_found_cluster) #number of duplicates found after clustering #TP


#FP:
pairs = []
for i in final_final:
    a = list(itertools.combinations(i, 2))
    for j in a:
        pairs.append(list(j)) #convert the pairs into lists 

fp_counter = 0
for i in pairs:
    if data_dict[i[0]]["modelID"] != data_dict[i[1]]["modelID"]:
        fp_counter += 1

        

BIG = []

for ID in unique_modelID:
    this = []
    for product in data_dict:
        if data_dict[product]["modelID"] == ID:
            this.append(product)
    BIG.append(this)
            

for i in BIG:
    if len(i) == 1:
        BIG.remove(i)
          
    
pairsW = []
for i in BIG:
    a = list(itertools.combinations(i, 2))
    for j in a:
        pairsW.append(list(j)) #convert the pairs into lists 



to_delete = []

for i in pairs:
    for j in pairsW:
        if i == j:
            to_delete.append(i) 
            
for i in to_delete:
    pairsW.remove(i)
            
FN = len(pairsW)


        
P = duplicates_found_cluster/(fp_counter + duplicates_found_cluster)    
R = duplicates_found_cluster/(duplicates_found_cluster + FN )

f = 2*P*R/(P + R)
print(f"f: {f}")

#End Part 6
#################################################################################################


# In[ ]:





# In[82]:





# In[ ]:





# In[ ]:





# In[ ]:




