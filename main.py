# -*- encoding: utf-8 -*-
#

def train(dir, n_clusters, n_iter):
  from keras.applications.vgg16 import VGG16, preprocess_input
  from keras.preprocessing.image import load_img, img_to_array
  from keras.models import Model
  import numpy as np
  import keras.datasets.cifar10

  # build initial convnet for 50 classifier without weights preset
  model = VGG16(weights=None, include_top=True, classes=n_clusters)
  model.compile(loss="categorical_crossentropy", optimizer="RMSprop")

  files = _samples(dir)[:100]
  images = np.array([img_to_array(load_img(f, target_size=(224, 224), interpolation="bicubic")) for f in files])

  def _iterate(model):
    layer_name = "flatten"
    features_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features_model.compile(loss="categorical_crossentropy", optimizer="RMSprop")
    features = features_model.predict(x=images)
    clusters = _cluster(features, n_clusters)
    print(clusters)
  
  _iterate(model)

def _cluster(features, n_clusters):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=n_clusters)
  return kmeans.fit_predict(features)

def _samples(dir, files=[]):
  import os
  for f in os.listdir(dir):
    f = os.path.join(dir, f)
    if os.path.isfile(f) and f.endswith(".jpg"):
      files.append(f)
    elif os.path.isdir(f):
      _samples(f, files)
  return files

if __name__ == "__main__":
  import sys, keras, tensorflow
  print("Python %s" % sys.version)
  print("Keras %s" % keras.__version__)
  print("TesnforFlow %s" % tensorflow.__version__)

  import argparse
  parser = argparse.ArgumentParser(description="Unsupervised image clustering by DeepCluster")
  parser.add_argument("dir", help="images directory")
  parser.add_argument("--cluster", "-c", dest="n_cluster", type=int, default=1000, help="the number of cluster")
  parser.add_argument("--iteration", "-i", dest="n_iter", type=int, default=100, help="the number of iteration")
  args = parser.parse_args()

  train(args.dir, args.n_cluster, args.n_iter)
