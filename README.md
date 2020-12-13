# Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging 

> The project is the official implementation of our *[IEEE TIP Journal](https://ieeexplore.ieee.org/Xplore/home.jsp), "Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging"*<br>  **&mdash; Mohit Lamba, Kranthi Kumar Rachavarapu, Kaushik Mitra**

***A single PDF of the paper and the supplementary is available at [arXiv.org](https://arxiv.org/abs/2003.02438).***

Light Field (LF) offers unique advantages such as post-capture refocusing and depth estimation, but low-light conditions, especially during night, severely limit these capabilities. We, therefore, propose a deep neural network architecture for ***Low-Light Light Field (L3F)*** restoration, which we call `L3Fnet`. The proposed L3Fnet not only performs the necessary visual enhancement of each LF view but also preserves the epipolar geometry across views. To facilitate learning-based solution for low-light LF imaging, we also collected a comprehensive LF dataset called `L3F-dataset`. Our code and the L3F dataset are now publicly available for [download](https://docs.google.com/document/d/1T6ct8PLkfm15LPRjRu--Nw7eoFJdvGjZHCI52hyFsEo/edit?usp=sharing). 

<details>
  <summary>Click to read full <i>Abstract</i> !</summary>
  
<p> Light Field (LF) offers unique advantages such as post-capture refocusing and depth estimation, but low-light conditions severely limit these capabilities.
To restore low-light LFs we should harness the geometric cues present in different LF views, which is not possible using single-frame low-light enhancement techniques. We, therefore, propose a deep neural network architecture for Low-Light Light Field (L3F) restoration, which we refer to as <code>L3Fnet</code>. The proposed L3Fnet not only performs the necessary visual enhancement of each LF view but also preserves the epipolar geometry across views. We achieve this by adopting a two-stage architecture for L3Fnet. Stage-I looks at all the LF views to encode the LF geometry. This encoded information is then used in Stage-II to reconstruct each LF view. <br>
To facilitate learning-based techniques for low-light LF imaging, we collected a comprehensive LF dataset of various scenes. For each scene, we captured four LFs, one with near-optimal exposure and ISO settings and the others at different levels of low-light conditions varying from low to extreme low-light settings. The effectiveness of the proposed L3Fnet is supported by both visual and numerical comparisons on this dataset. To further analyze the performance of low-light reconstruction methods, we also propose an <code>L3F-wild dataset</code> that contains LF captured late at night with almost zero lux values. No ground truth is available in this dataset. To perform well on the L3F-wild dataset, any method must adapt to the light level of the captured scene. To do this we use a pre-processing block that makes L3Fnet robust to various degrees of low-light conditions. Lastly, we show that L3Fnet can also be used for low-light enhancement of single-frame images, despite it being engineered for LF data. We do so by converting the single-frame DSLR image into a form suitable to L3Fnet, which we call as <code>pseudo-LF</code>. </p> 
 
</details>

# Sample Output
Low-light conditions are a big challenge to Light Field applications.
For example, the depth estimate of LF captured in low light is very poor. Our
proposed method not only visually restores each of the LF views but also
preserves the LF geometry for faithful depth estimation, as shown below (*Click to see full resolution image*).
<p align="center">
  <a href="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/title_fig.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/title_fig.png" alt="Click to expand full res image" height="432">
  </a>
</p> 

<details>
  <summary>Click to see more <i>Results</i> !</summary>

The proposed L3Fnet harnesses information form all the views to produce sharper and less noisy restorations. Compared to our restoration, the existing state-of-the-art methods exhibit considerable amount of noise and blurriness in their restorations. This is substantiated by both qualitative and PSNR/SSIM quantitative evaluations.

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/fig4.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/table.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/depth.jpg">
</p>

</details>

# The L3F Dataset

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/dataset.png" height="332">
</p>

Unlike most previous works on low-light enhancement we do not simulate low-light images using Gamma correction or modifying images in Adobe Photoshop. Rather we physically capture Light Field images when the light falling on camera lens is in between 0-20 lux.

### L3F-20, L3-50 and L3F-100 dataset
The L3F-dataset used for training and testing comprises of `27 scenes`. For each scene we capture one LF with large exposure which then serves as the well-lit GT image. We then capture 3 more LFs captured at 20th, 50th and 100th fraction of the exposure used for the GT image. A detailed descrition of the collected dataset can be found in ***Section III*** of the main paper.

The RAW format used by Lytro Illum is very large (400 - 500 MB) and requires several post-processing such as hexagonal to rectilinear transformation before it can be used by `L3Fnet`. We thus used JPEG compressed images for training and testing. Even after compression each LF is abour 50 MB is size; much larger than even raw DSLR image. Although we do not use raw LF images in this work, we also make the raw LF images public.

<details>
  <summary>Click here to see the central SAIs of all the <i>27 scenes ! </i> </summary>

The following scenes are used for TRAINING.

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/train.png">
</p>

The following scenes are used for TESTING.

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/test.png">
</p>

</details>

### L3F-wild Dataset
The Light Fields captured in this dataset were captured late in the night in almost ***0 lux*** conditions. These scenes were captured with normal ISO and exposure settings as if being captured in bright sunlight in the day. The scenes in the L3F-wild dataset are so dark that no GT was possible. Thus they cannot be used for quantitative evaluation but serves as a real-life qualitative check for methods which claim low-light enhancement.

<details>
  <summary>Click here to see the <i>SAIs restored by our L3Fnet ! </i> </summary>

<p align="center">
  <br> <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/wild.png">
</p>

</details>


# How to use the Code ?

The L3F code and dataset is arranged as below,

<div style="width:300px;overflow:auto;padding-left:200px;">
<pre>
  ./L3F
├── L3F-dataset
│   ├── jpeg
│   │   ├── test
│   │   │   ├── 1
│   │   │   ├── 1_100
│   │   │   ├── 1_20
│   │   │   └── 1_50
│   │   └── train
│   │       ├── 1
│   │       ├── 1_100
│   │       ├── 1_20
│   │       └── 1_50
│   └── raw
│       ├── test
│       │   ├── 1
│       │   ├── 1_100
│       │   ├── 1_20
│       │   └── 1_50
│       └── train
│           ├── 1
│           ├── 1_100
│           ├── 1_20
│           └── 1_50
├── L3Fnet
│   ├── expected_output_images
│   ├── demo_cum_test.py
│   ├── train.ipynb
│   └── weights
└── L3F-wild
    ├── jpeg
    └── raw
</pre>
</div>

The <code>jpeg</code> directory contains the decoded LF in `.jpeg` format and were used in this work. The well-illuminated GT images are present in the directory called <code>1</code> and the other folders, namely `1_20`, `1_50` and `1_100` contain the low-light LF images with reduced exposure. The original undecoded raw files captured by Lytro Illum can be found in the <code>raw</code> directory. The corresponding raw and .jpeg files have the same file names.

To find the well-lit and low-light image pairs, alphabetically sort the directories by file name. The images are then matched by the same serial number. An example for the <code>L3F-100</code> datset is shown below,

<div style="width:600px;overflow:auto;padding-left:50px;">
<pre>
  import os
  from PIL import Image
  
  GT_files = sorted(os.walk('L3F/L3F-dataset/jpeg/test/1'))
  LowLight_files = sorted(os.walk('L3F/L3F-dataset/jpeg/test/1_100'))
  
  GT_image = Image.open(GT_files[idx]).convert('RGB')
  LowLight_image = Image.open(LowLight_files[idx]).convert('RGB')
</pre>
</div>

The code for L3F-net can be found in `L3F-net` directory. Execute `demo_cum_test.py` file for test the code. The expected output images after excuting this file is are given in `expected_output_images` directory. 

# Cite us

<div style="width:600px;overflow:auto;padding-left:50px;">
<pre>
  @inproceedings{l3fnet,
  title={Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging},
  author={Lamba, Mohit and Kumar, Kranthi and Mitra, Kaushik},
  booktitle={IEEE Transactions on image processing},
  year={20xx}
}
</pre>
</div>




<!---
| |L3F-20 | L3F-50 | L3F-100 |
|:---:|:---:|:---:|:---:|
| LFBM5D | 24.48/0.79| 20.94/0.64| 18.14/0.46 |
| PBS | 20.80/0.68 | 16.48/0.53 | 13.94/0.38       |
| RetinexNet | 21.82/0.72| 18.98/0.59| 17.8/0.41 |
| DID | 24.09/0.78| 22.63/0.68| 20.68/0.61       |
| SGN | 24.10/0.76| 22.18/0.67| 20.70/0.59       |
| SID | 24.53/0.76| 22.87/0.66| 20.75/0.58       |
| Our L3Fnet| 25.25/0.82| 23.67/0.74| 22.61/0.70     |
--->





