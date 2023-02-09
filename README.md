## Self-Supervised Micro-Expression Analysis with Facial Micro-Movement

## How to run the code

<b>Step 1)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 2) Downstream Task Training and Evaluation

&nbsp; <b>2a.</b> Micro-Expression Recognition

&nbsp; ``` python me_recog_main.py ```

#### &nbsp;&nbsp;&nbsp; For parameter settings <br>
&nbsp;&nbsp;&nbsp;&nbsp; --dataset_name (Composite/MMEW) <br>
&nbsp;&nbsp;&nbsp;&nbsp; --train_model (True/False)
  
&nbsp; <b>2b.</b> Micro-Gesture Recognition

&nbsp; ``` python mg_recog_main.py ```

#### &nbsp;&nbsp;&nbsp; For parameter settings <br>
&nbsp;&nbsp;&nbsp;&nbsp; --train_model (True/False)

&nbsp; <b>2c.</b> Micro-Expression Spotting

&nbsp;&nbsp;&nbsp; Due to huge storage required, please download the processed optical flow of the datasets from the following links: <br>
&nbsp;&nbsp;&nbsp; CASME_sq_dataset.pkl (5.6 GB): https://drive.google.com/file/d/13wtd9OBJ_FV4jXq5iWhFg4v8XYLlR_NX/view?usp=sharing <br>
&nbsp;&nbsp;&nbsp; SAMMLV_dataset.pkl (21.8 GB): https://drive.google.com/file/d/1KLu0NrJ_sf9dz81ANA-ChrEszxRaIz9D/view?usp=sharing <br>
  
&nbsp;&nbsp;&nbsp; Place the files CASME_sq_dataset.pkl and SAMMLV_dataset.pkl under the folder dataset <br>
  
&nbsp; ``` python me_spot_main.py ```

#### &nbsp;&nbsp;&nbsp; For parameter settings <br>
&nbsp;&nbsp;&nbsp;&nbsp; --dataset_name (CASME_sq/SAMMLV) <br>
&nbsp;&nbsp;&nbsp;&nbsp; --train_model (True/False)
  
## If you wish to re-train the self-supervised approach
<b>Step 1)</b> Download pre-train dataset from : (please request from the author at the moment)
  
<!--
https://drive.google.com/file/d/13MKvf6q3Yq1dq7OnyYBZOAaM5R1sKznk/view?usp=sharing
-->
  
<b>Step 2)</b> Place the folder (CASME_sq) accordingly: <br>
>├─CASME_sq <br>
>├─ME_Recog <br>
>├─MG_Recog <br>
>└─......
  
<b>Step 3)</b> SSL Training

``` python ssl_main.py ```

## Additional Notes

<!--
If you have issue installing torch, run this: <br>
``` pip install torch===1.5.0 torchvision===0.6.0 torchsummary==1.5.1 -f https://download.pytorch.org/whl/torch_stable.html ```
-->

If you have issues downloading the files Composite_dataset.pkl (~166MB) and imigue_dataset.pkl (~660MB) due to large file size, you can download the files here: <br>
Composite_dataset.pkl : https://drive.google.com/file/d/1apjtx2hNdBiRuOXAiJDs7uCVcB2GUT9a/view?usp=sharing <br>
imigue_dataset.pkl : https://drive.google.com/file/d/1_iBbrc19fd4-UWlOpVpQzXGRcAI_f2z0/view?usp=sharing
  
The weights for all networks are hidden (please request from the author at the moment)

<!--
Download from the following link: <br>
https://drive.google.com/file/d/1NJ8szyeG5pVRg1ab_OtFYTWPX-aCv3-z/view?usp=sharing <br>

Place the folders accordingly: <br>
>├─CASME_sq <br>
>├─ME_Recog <br>
>├─ME_Recog_Weights <br>
>├─MG_Recog <br>
>├─MG_Recog_Weights <br>
>└─......
-->
  
##### Please email me at genbing67@gmail.com if you have any inquiries or issues.
