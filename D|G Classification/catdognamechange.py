#this script merges cats and dogs into one file and renames them
#to cat1.jpg, cat2.jpg, etc. and dog1.jpg, dog2.jpg, etc.

import os
import shutil

#make a new directory to hold the new files
os.mkdir('catsdogs')

#rename the files
def rename_imgs(path):
    #this takes in the path to the directory with the files which has cats and dogs subdirectories
    #and renames the files to cat1.jpg, cat2.jpg, etc. and dog1.jpg, dog2.jpg, etc.
    #and puts them in the catsdogs directory
    #path is a string

    #get the list of files in the cats directory
    cats = os.listdir(path + '/cats')
    #get the list of files in the dogs directory
    dogs = os.listdir(path + '/dogs')

    #rename the files
    for i in range(len(cats)):
        #get the file extension
        ext = cats[i][-4:]
        #copy the file to the new directory without changing the name
        shutil.copy(path + '/cats/' + cats[i], 'catsdogs/cat' + str(i+1) + ext)


    for i in range(len(dogs)):
        #get the file extension
        ext = dogs[i][-4:]
        #rename the fil
        shutil.copy(path + '/dogs/' + dogs[i], 'catsdogs/dog' + str(i+1) + ext)

#call the function
rename_imgs("/home/kalyan/DataSets/DogsandCats/training_set/training_set")
