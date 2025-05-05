import os
import sqlite3
import PIL.Image
import pydicom
import numpy as np
os.chdir(r'C:\\ML Data\\Chinesearchive\\TheChineseMammographyDatabase')
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

image_path_base = 'C:\\ML Data\\Chinesearchive\\TheChineseMammographyDatabase'
os.chdir(image_path_base)

text_path_base = 'C:/ML Data/Chinesearchive/TheChineseMammographyDatabase/cleanCMMD.csv' #Hardcoded path. Modify if needed

tempDataArray = []
tempImageArray = []

with open(text_path_base) as f:


  text_string = ""
  f.readline() #skip first line

  image_name_int = 0
  for line in f:
    splitLine = line.split(",")

    imagePath = splitLine[4].replace("\n", "").replace('"',"") #Removing newline and quotation marks

    for imagefile in os.listdir(imagePath):

      if imagefile.endswith('dcm'):

        img = os.path.join(imagePath, imagefile)

        dcm_image = pydicom.dcmread(img)

        nparrayImage = dcm_image.pixel_array
        
        classification = splitLine[3].replace('"', '') #Removing quotation marks

        cancertype = 0

        if classification == "Benign":
          cancertype = 0
        if classification == "Malignant":
          cancertype = 1


        if nparrayImage.dtype != np.uint8:
         # Normalize and convert to 8-bit
         nparrayImage = (nparrayImage - nparrayImage.min()) / (nparrayImage.max() - nparrayImage.min()) * 255
         nparrayImage = nparrayImage.astype(np.uint8)

        # Create a PIL Image from the pixel array
        # Assuming grayscale for simplicity. For color images, additional steps are needed.
        image = PIL.Image.fromarray(nparrayImage)

        imagePathappend = r'C:\ML Data\Chinesearchive\TheChineseMammographyDatabase\CMMD png 2\test' + str(image_name_int) + '.png'
        # Save the image as a PNG file
        image.save(imagePathappend)

        sql_insert_image = f"INSERT INTO images VALUES (?, ?,?,?)"
        cursor.execute(sql_insert_image, (image_name_int, cancertype, imagePathappend, 12349))
        conn.commit()
        image_name_int += 1

conn.close