# SSL_Micro-Movement

Self-Supervised Approach by Learning Spatio-Temporal Features in Micro-Movement

## How to run the code

<b>Step 1)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 2) Downstream Task Training and Evaluation

&nbsp; <b>2a.</b> Micro-Expression Recognition

&nbsp; ``` python me_main.py ```

&nbsp; <b>2b.</b> Micro-Gesture Recognition

&nbsp; ``` python mg_main.py ```

#### &nbsp;&nbsp; Note for parameter settings <br>
&nbsp;&nbsp;&nbsp; --train_model (True/False)

## If you wish to re-train the self-supervised approach
<b>Step 1)</b> Download pre-train dataset from : (Please request from the author at the moment)
  
<!--
https://drive.google.com/file/d/13MKvf6q3Yq1dq7OnyYBZOAaM5R1sKznk/view?usp=sharing
-->
  
<b>Step 2)</b> Place the folder (CASME_sq) accordingly:
  
>├─CASME_sq <br>
>├─ME_Recog <br>
>├─ME_Recog_Weights <br>
>├─MG_Recog <br>
>└─......
  
<b>Step 3)</b> SSL Training

``` python ssl_main.py ```

## Additional Notes

If you have issue installing torch, run this: <br>
``` pip install torch===1.5.0 torchvision===0.6.0 torchsummary==1.5.1 -f https://download.pytorch.org/whl/torch_stable.html ```

If you have issues downloading the files Composite_dataset.pkl (~166MB) and imigue_dataset.pkl (~660MB) due to large file size, you can download the files here: <br>
Composite_dataset.pkl : https://drive.google.com/file/d/1apjtx2hNdBiRuOXAiJDs7uCVcB2GUT9a/view?usp=sharing <br>
imigue_dataset.pkl : https://drive.google.com/file/d/1_iBbrc19fd4-UWlOpVpQzXGRcAI_f2z0/view?usp=sharing
  
##### Please email me at genbing67@gmail.com if you have any inquiries or issues.
