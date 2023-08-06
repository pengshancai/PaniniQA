# PaniniQA
Repo for the TACL 2023 paper "PaniniQA: Enhancing Patient Education Through Interactive Question Answering"


## Dataset 

We open source two datasets:

1. **Dataset 1** - 456 annotated discharge instructions from MIMIC-III Clinical Database
2. **Dataset 2** - 100 synthesized discharge instructions generated by pre-trained neural models)

### Detailed instructions on Dataset 1
The 456 discharge instructions in Dataset 1 are from the MIMIC-III Clinical Database, a large freely-available database comprising deidentified health-related data associated with patients who stayed in critical care units of the Beth Israel Deaconess Medical Center. 
To acquire these discharge instructions, please first obtain the credential from [this website](https://link-url-here.org).
After acquiring the credential, please visit [this link](https://physionet.org/content/mimiciii/1.4/NOTEEVENTS.csv.gz) to download the file **NOTEEVENTS.csv.gz**.

Please run the following command to extract the clinical instructions from the downloaded file:

