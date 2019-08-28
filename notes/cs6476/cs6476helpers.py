from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cv2
import matplotlib.pyplot as plt
import numpy as np



def imshow(image,time:int=0,title:str='image'):
  """
  This function takes as arguments
  image: An image file loaded in via OpenCV into Numpy arrays
  time: An integer denoting the time the image should remain.
    The default, 0, waits for the user to press a key.
  title: A string setting the top menu bar name of the image.

  It displays an image for a user to see.
  """
  cv2.imshow(title,image)
  cv2.waitKey(time)
  cv2.destroyAllWindows()

def blend(im1:np.ndarray,im2:np.ndarray,a:float=0.5,type:str="float"):
  """
  This function takes as arguments
  im1: An image file loaded in via OpenCV into Numpy arrays
  im2: An image file loaded in via OpenCV into Numpy arrays
  a: The alpha parameter denoting the weight of images in the blend.
  type: A string denoting what the type of the output should be

  return: (a*im1 + (1.-a)*im2).astype(type)
  """
  return (a*im1 + (1.-a)*im2).astype(type)

def surf(img:np.ndarray):
  """
  This function takes as arguments
  img: An image file (2D) loaded in via OpenCV into Numpy arrays
  
  It displays a surface plot of a 2D image
  """
  assert len(img.shape) == 2, "Image must be 2D"
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X = np.arange(0,img.shape[1])
  Y = np.arange(0,img.shape[0])
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X,
                         Y,
                         img,
                         cmap=cm.coolwarm,
                         linewidth=0,
                         antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()

def noise(img:np.ndarray,type:str="salt and pepper",noise:int=100):
  """
  This noise function takes as inputs an image to operate on and
  the type of noise to apply.
  img: An image file loaded in via OpenCV into Numpy arrays
  type: The type of noise in the set
    ["salt and pepper"]
  noise: An integer to be used in determining the noise density.
    Lower produces a noisier image.

  This returns a noisy image; note that this makes *no* assumptions
  about the structure of your original image! The image returned is
  the same size and dtype as the original image.
  """
  types = ["salt and pepper"]
  assert type in types, "You must select from {}".format(", ".join(types))
  # How much of the image will be modified?
  noisy = np.random.randint(pad,size=img.shape)
  if type == "salt and pepper":
    # What are the maximum and minimum values in the image?
    salt = np.amin(img)
    pepper = np.amax(img)
    # Wherever noise is 0 create 'salt' noise
    img = np.where(noisy==0, salt, img)
    # Where noise is 'noise-1' create pepper noise
    img = np.where(noisy==noisy-1, pepper, img)
  return(img)