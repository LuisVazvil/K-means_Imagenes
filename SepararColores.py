from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagen", required = True, help = "Ruta de la imagen")
ap.add_argument("-c", "--k", required = True, type = int,
	help = "Numero de K")
args = vars(ap.parse_args())

imagen = cv2.imread(args["imagen"])
(height, weight) = imagen.shape[:2]

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
 
imagen = imagen.reshape((height * weight, 3))


k = MiniBatchKMeans(n_clusters = args["k"])
labels = k.fit_predict(imagen)
puntos = k.cluster_centers_.astype("uint8")[labels]
 
puntos = puntos.reshape((height, weight, 3))
imagen = imagen.reshape((height, weight, 3))

 
puntos = cv2.cvtColor(puntos, cv2.COLOR_LAB2BGR)
imagen = cv2.cvtColor(imagen, cv2.COLOR_LAB2BGR)

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
puntos = cv2.cvtColor(puntos, cv2.COLOR_BGR2RGB)


plt.figure("Imagen Original")
plt.axis("off")
plt.imshow(imagen)

plt.figure("Imagen final")
plt.axis("off")
plt.imshow(puntos)
plt.show()