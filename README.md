# DeepDenoiser
This project aims to introduce a deep learning based denoiser to Blender's render engine Cycles.

## Overview
The current goal is to get a flexible pipeline where ideas can relatively easily be tried out and the results can be thoroughly analyzed with appropriate visualizations.
A rendering script was implemented which generates examples. The neural network gets the noisy rendered images as input and is supposed to output the clean ones.

## Practical Relevance
Based on the existing research and commercial solutions, there are two obvious ways in which such a solution might be used in practice. 

### Preview Rendering and Denoising
The denoising is performed after very few samples per pixel have been rendered to allow short iteration times for artists. To get stable results, it is most likely needed to render the same image multiple times with different seeds. Further, it appears than only low dynamic range could be achieved with this approach so far.
Instead of using LSTMs, it might be more efficient to feed two or three image variants into the network at once. This should help with the training time.

### Production Rendering and Denoising
The denoising is performed after a certain amount of samples per pixel have been rendered, such that production quality renders can be achieved. It should be possible to render any kind of image with this solution.
Due to the complexity of this task to cover such a huge spectrum of cases, it should be considered to allow the artists to fine tune the denoiser. If they plan to render several images within the same style, they might render a few without the denoiser until they converge. Those could then be used to continue the training of the denoiser to fine tune it for that specific style.

## Data Set
Data is the main ingredient for the training of neural networks. The nature of this project is very open and so is the data set. It only uses open content and is also available as such. I am looking for a place where it can be hosted and easily updated. If you are interested to get the data set, feel free to contact me (deepblender@gmail.com). I am glad to share it. Contributions are also very welcome!

## Documentation
The documentation is severely lacking at this point. It is going to be added as this project stabilizes.

## Limitations
This solution currently only works for individual images and not for sequences. It is certainly planned to extend the project to cover sequences as well.