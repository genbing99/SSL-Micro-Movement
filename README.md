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
<b>Step 1)</b> Download pre-train dataset from : https://drive.google.com/file/d/13MKvf6q3Yq1dq7OnyYBZOAaM5R1sKznk/view?usp=sharing

<b>Step 2)</b> Place the folder (CASME_sq) accordingly:
  
>├─CASME_sq <br>
>├─ME_Recog <br>
>├─ME_Recog_Weights <br>
>├─MG_Recog <br>
>└─......
  
<b>Step 3)</b> SSL Training

``` python ssl_main.py ```

## Additional Notes

##### Please email me at genbing67@gmail.com if you have any inquiries or issues.
