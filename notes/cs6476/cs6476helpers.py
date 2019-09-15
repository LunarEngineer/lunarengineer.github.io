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
  cv2.imshow(title,norm(image))
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

def norm(img:np.ndarray):
  """
  This is just a shorthand function to help when plotting float images
  """
  return cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

def LoG(size, sigma):
  """
  This filter function takes as inputs a filter size and sigma and
  produces a laplacian of a gaussian function centered at zero.
  size: The kernel size
  sigma: The standard deviation of the Gaussian function

  This returns a filter which is the Laplacian of a Gaussian kernel
  This function was borrowed from the course notes.
  """
  # Set the range for x and y to be size units away from the center.
  x = y = np.linspace(-size, size, 2*size+1)
  # Create a 2D Numpy array from those arrays.
  x, y = np.meshgrid(x, y)
  # Laplacian creation
  f = (x**2 + y**2)/(2*sigma**2)
  k = -1./(np.pi * sigma**4) * (1 - f) * np.exp(-f)
  return k

def drawLines(img,lines):
  """
  Given lines from cv2 houghlines draws the lines.
  """
  lines = lines.reshape(-1,2)
  for rho,theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
  return(img)

def intersection(l1,l2):
  """
  Finds the intersection of two lines.
  See https://stackoverflow.com/a/383527/5087436
  """
  r1, t1 = line1
  r2, t2 = line2
  A = np.array([
      [np.cos(t1), np.sin(t1)],
      [np.cos(t2), np.sin(t2)]
  ])
  b = np.array([[rho1], [rho2]])
  x0, y0 = np.linalg.solve(A, b)
  x0, y0 = int(np.round(x0)), int(np.round(y0))
  return np.array([x0, y0])

def enhance(img_in,scale=24,ksize=11):
  img_blurred = np.copy(img_in)
  img_blurred = cv2.medianBlur(img_blurred,ksize)
  img_blurred = cv2.GaussianBlur(img_blurred,(3,3),4)
  sharp_kernel = (-1/9.)*np.array([[1,1,1],[1,-scale,1],[1,1,1]])
  img_blurred = cv2.filter2D(img_blurred,-1,sharp_kernel)
  return img_blurred

def mask_img(img_in,shape="rectangle",center=(0,0),majorAxis=0,minorAxis=0):
  """
  Used to mask an image. For rectangles majoraxis is width and minor is height.
  For circles majoraxis is radius.
  center is an (x,y) coordinate as used by cv2.
  """
  img_out = np.copy(img_in)
  center = [int(x) for x in center]
  if shape == "rectangle":
    maxX, minX = max(0,center[0] + majorAxis//2), max(0,center[0] - majorAxis//2)
    maxY, minY = max(0,center[1] + minorAxis//2), max(0,center[1] - minorAxis//2)
    img_out[minY:maxY,minX:maxX,:] = np.array([255,255,255])
  return img_out

def threshold(img,threshold=200):
  img = np.copy(img)
  blue = img[:,:,0]
  blue[blue<threshold] = 0
  blue[blue>threshold] = 255
  green = img[:,:,1]
  green[green<threshold] = 0
  green[green>threshold] = 255
  red = img[:,:,2]
  red[red<threshold] = 0
  red[red>threshold] = 255
  img_out = np.stack([blue,green,red],axis=2)
  return(img_out)
