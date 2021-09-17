import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import cv2
# For measuring the inference time.
import time


def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
#  response = urlopen(url)
#  image_data = response.read()
#  image_data = BytesIO(image_data)
  pil_image = Image.open(url)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
#  if display:
#    display_image(pil_image)
  return filename

def resize(img, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
#  response = urlopen(url)
#  image_data = response.read()
#  image_data = BytesIO(image_data)
  #pil_image = Image.open(url)
  pil_image = ImageOps.fit(img, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
#  if display:
#    display_image(pil_image)
  return filename

def resize_image(img, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  
#  response = urlopen(url)
#  image_data = response.read()
#  image_data = BytesIO(image_data)
  pil_image = Image.open(img)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
#  if display:
#    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin
def draw_boxes(image, boxes, class_names, scores, max_boxes=1, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
  coordinates = []
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(boxes.shape[0]):
    if class_names[i] == b'Glasses':
      a =[]
      if len(coordinates) == 1:
          continue
      if scores[i] >= min_score:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        a.append(ymin*500)
        a.append(ymax*500)
        a.append(xmin*500)
        a.append(xmax*500)
        coordinates.append(a)
        display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                      int(100 * scores[i]))
        color = colors[hash(class_names[i]) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        draw_bounding_box_on_image(
            image_pil,
            ymin,
            xmin,
            ymax,
            xmax,
            color,
            font,
            display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))
  ymin = coordinates[0][0]
  ymax = coordinates[0][1]
  xmin = coordinates[0][2]
  xmax = coordinates[0][3]
  length_glass = (ymax - ymin)
  breadth_glass = (xmax - xmin)
  return image,length_glass,breadth_glass

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def load_img2(path):
#  img = tf.Tensor(path)
  img = tf.convert_to_tensor(path,dtype=tf.float32)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes, length, height = draw_boxes(img.numpy(), result["detection_boxes"],
      
      result["detection_class_entities"], result["detection_scores"])

  #display_image(image_with_boxes)
  #print('Length of the Glasses are: ',length)
  #print('Height of the Glasses are: ',height)
  return length, height

def run_detector2(detector, img):
#  img = load_img2(img)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes, length, height = draw_boxes(img, result["detection_boxes"],
      
      result["detection_class_entities"], result["detection_scores"])

  display_image(image_with_boxes)
  #print('Length of the Glasses are: ',length)
  #print('Height of the Glasses are: ',height)
  return image_with_boxes,(length, height)





#app=Flask(__name__)
#Swagger(app)

#pickle_in = open("Glass-Size.pkl","rb")
#classifier=pickle.load(pickle_in)



#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def prediction(path):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    #prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    prediction = classifier(path)
    print(prediction)
    
    return prediction


def GlassSize(path):
  downloaded_image_path = download_and_resize_image(path, 500, 500, True)
  module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" # ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
  detector = hub.load(module_handle).signatures['default']
  l,h = run_detector(detector, downloaded_image_path)
  return l,h

def GlassSize2(image):
  module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" # ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
  detector = hub.load(module_handle).signatures['default']
  l,h = run_detector2(detector, image)
  return l,h

def GlassSize3(image):
  downloaded_image_path = resize(image, 500, 500, True)
  module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" # ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
  detector = hub.load(module_handle).signatures['default']
  l,h = run_detector(detector, downloaded_image_path)
  return l,h


def main():
    st.title("Glass-Size-Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Glass size Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    #variance = st.text_input("Variance","Type Here")
    #skewness = st.text_input("skewness","Type Here")
    #curtosis = st.text_input("curtosis","Type Here")
    #entropy = st.text_input("entropy","Type Here")
    path = st.text_input("Enter the Image Path","Type Here")
    result=""
    if st.button("Predict"):
        #result=predict_note_authentication(variance,skewness,curtosis,entropy)
        result = GlassSize(path)
    st.text("Please make sure to wear the glasses before opening the camera and put the camera in a place that the glasses are visible")
    if st.button("Open Camera"):
        print("Streaming started")
        video_capture = cv2.VideoCapture(0)
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame,result = GlassSize2(frame)
	#cv2.imshow("Frame",frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
        video_capture.release()
        cv2.destroyAllWindows()
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
    
if __name__=='__main__':
    main()