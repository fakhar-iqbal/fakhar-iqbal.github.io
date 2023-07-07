# Image Recognizers for Non-Image Tasks?

An image recognizer, as the name suggests, only recognizes images. But a lot of things can be represented as images, which means that an image recognizer can learn to complete many tasks.
      
One interesting thing to remember is that Machine Learning is mostly about creativity! The more creative you are, the more you know to play with Machine Learning. 				
Same is the case with this `Image Recognizer`.
Lets look at some of the fascinating non-image tasks it does:

##Spectrograms of Sounds
A sound can be converted to a spectrogram, which is a chart that shows the amount of each frequency at each time in an audio file. Keeping this in view, Fast.ai student Ethan Sutin used this approach to easily beat teh published accuracy of a state-of-the-art environmental sound detection model using a dataset of 8732 urban sounds. fastai's `show_batch` clearly shows how each sound has a quite distinctive spectrogram:
![](/images/sounds.jpeg "Spectrogram of sounds")

##Time Series Images
A time series can easily be converted into an image by simply plotting a time series on a graph. However, it is often good idea to try to represent your data in a way that makes it as easy as possible to pull out the most important components. In a time series, things like seasonality and anomalies are most likely to be of interest.
fast.ai student ignacio Oguiza created images from a time series dataset for olive oil classification, using a technique called `Gramian Angular Difference Field (GADF)`. He then fed those images to an image classification model. His results, despite having only 30 training set images, were well over 90% accurate, and close to the state-of-the-art. 

![](/images/time.jpg "Time series to image")

##Mouse movements and clicks
Another interesting fast.ai student project example comes from Gleb Esman. He was working on fraud detection at Splunk, using a dataset of user's mouse movements and mouse clicks. He turned those into pictures by drawing an image displaying the position, speed, and acceleration of the mouse pointer by using colored lines, and the clicks were displayed using small colored circles. He fed those into an image recognition model. It worked so well that it led to a patent for this approach to fraud analytics!

![](/images/mouse.jpg "Computer Mouse behaviour in to images")


##Takeaways
   You will find that a small number of general approaches in deep learning can go a long way, if you are a bit creative in how you represent your data! You can make bring changes if you are a master to play with and represent your data in an insightful way. 
Don't think of approaches like the ones described here as `"hacky workarounds"`, because they often (as here) beat previously state-of-the-art results. These really are the right ways to think about these problem domains. 
