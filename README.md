# Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging

> The project is the official implementation of our *IEEE TIP Journal, "Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging"*<br>  **&mdash; Mohit Lamba, Kranthi Kumar Rachavarapu, Kaushik Mitra**

***A single PDF of the paper and the supplementary is available at [arXiv.org](https://arxiv.org/abs/2003.02438).***

Light Field (LF) offers unique advantages such as post-capture refocusing and depth estimation, but low-light conditions, especially during night, severely limit these capabilities. We, therefore, propose a deep neural network architecture for *Low-Light Light Field (L3F)* restoration, which we refer to as ***L3Fnet***. The proposed L3Fnet not only performs the necessary visual enhancement of each LF view but also preserves the epipolar geometry across views. To facilitate this learning-based solution for low-light LF imaging, we collected a comprehensive LF dataset. Both our code and the L3F dataset are now publicly available and can be downloaded using the links provided on the left. 

<details>
  <summary>Click to read full <i>Abstract</i> !</summary>
  
<p>Light Field (LF) offers unique advantages such as post-capture refocusing and depth estimation, but low-light conditions severely limit these capabilities.
To restore low-light LFs we should harness the geometric cues present in different LF views, which is not possible using single-frame low-light enhancement techniques. We, therefore, propose a deep neural network architecture for Low-Light Light Field (L3F) restoration, which we refer to as <b>L3Fnet</b>. The proposed L3Fnet not only performs the necessary visual enhancement of each LF view but also preserves the epipolar geometry across views. We achieve this by adopting a two-stage architecture for L3Fnet. Stage-I looks at all the LF views to encode the LF geometry. This encoded information is then used in Stage-II to reconstruct each LF view.<br>

To facilitate learning-based techniques for low-light LF imaging, we collected a comprehensive LF dataset of various scenes. For each scene, we captured four LFs, one with near-optimal exposure and ISO settings and the others at different levels of low-light conditions varying from low to extreme low-light settings. The effectiveness of the proposed L3Fnet is supported by both visual and numerical comparisons on this dataset. To further analyze the performance of low-light reconstruction methods, we also propose an <b>L3F-wild dataset</b> that contains LF captured late at night with almost zero lux values. No ground truth is available in this dataset. To perform well on the L3F-wild dataset, any method must adapt to the light level of the captured scene. To do this we use a pre-processing block that makes L3Fnet robust to various degrees of low-light conditions. Lastly, we show that L3Fnet can also be used for low-light enhancement of single-frame images, despite it being engineered for LF data. We do so by converting the single-frame DSLR image into a form suitable to L3Fnet, which we call as pseudo-LF.</p>
  
 
</details>

# Sample Output
Low-light is a severe bottleneck to Light Field applications.
For example, the depth estimate of LF captured in low light is very poor. Our
proposed method not only visually restores each of the LF views but also
preserves the LF geometry for faithful depth estimation, as shown below (*Click to see full resolution image*).
<p align="center">
  <a href="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/title_fig.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/title_fig.png" alt="Click to expand full res image" height="432">
  </a>
</p> 

The proposed L3Fnet harnesses information form all the views to produce sharper and less noisy restorations. Compared to our restoration, the existing state-of-the-art methods exhibit considerable amount of noise and blurriness in their restorations. This is substantiated by both qualitative and PSNR/SSIM quantitative evaluations.

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/fig4.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/table.png">
  <img src="https://raw.githubusercontent.com/MohitLamba94/L3Fnet/main/imgs/depth.jpg">
</p>


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





