# DeepMASS_Web

DeepMASS_Web is an online platform developed from DeepMASS2, which enables deep-learning based metabolite annotation
 via semantic similarity analysis of mass spectral language. This approach enables the prediction 
 of structurally related metabolites for the unknown compounds. By considering the chemical space, these 
 structurally related metabolites provide valuable information about the potential location of the unknown 
 metabolites and assist in ranking candidates obtained from molecular structure databases. 

## News

[10/2024] Using DeepMASS2, we made a web UI interface, check it out! [Website](http://218.245.102.112/)

## Installation

If you want to build this website by yourself, please follow the installation steps below:

1. Install [Anaconda](https://www.anaconda.com/)  or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

2. Clone the repository and navigate into it:

   ```bash
   git clone https://github.com/hcji/DeepMASS_Web.git
   cd DeepMASS_Web/
   ```

3. For the installation of dependencies

   Use the following step for installation:

   ```bash
   conda env create -f environment.yml
   conda activate deepmass
   ```

   Or follow the steps below:

    (1) Create a new conda environment and activate:

   ```bash
   conda create -n deepmass python=3.8.13
   conda activate deepmass
   ```

    (2) Install dependency (note, for *MacOS* some dependency may install with conda manually):

   ```bash
   pip install -r requirements.txt
   ```

4. Download the [dependent data](https://github.com/hcji/DeepMASS2_GUI/releases/tag/v0.99.1).    

   1. put the following files into *data* folder:

      ```bash
      DeepMassStructureDB-v1.1.csv
      references_index_negative_spec2vec.bin
      references_index_positive_spec2vec.bin
      references_spectrums_negative.pickle
      references_spectrums_positive.pickle
      ```

   2. put the following files into *model* folder:

      ```bash
      Ms2Vec_allGNPSnegative.hdf5
      Ms2Vec_allGNPSnegative.hdf5.syn1neg.npy
      Ms2Vec_allGNPSnegative.hdf5.wv.vectors.npy
      Ms2Vec_allGNPSpositive.hdf5
      Ms2Vec_allGNPSpositive.hdf5.syn1neg.npy
      Ms2Vec_allGNPSpositive.hdf5.wv.vectors.npy
      ```

5. Create a new file named `config.yaml` in the `DeepMASS_Web/backend/config` directory. Insert the following code into the file, replace **XX** with the actual values as needed, and save the file.

   ```bash
   email:
       mail_user: XX
       mail_pwd: XX
       mail_sender: XX
       port: 25
       host: XX
   register:
       captcha_expire_time:
   identification:
       max_spectrum: 1000
       max_files: 1000
       max_file_size: 1024*1024*3
       plot:
           dpi: 900
           width: 2
           length: 1
   ```

6. Create a `User_Information.db` database by yourself and place it in the `DeepMASS_Web/backend/sqlite` directory.

7. Replace all instances of the IP address `218.245.102.112` in the files with the IP address of your own host.

8. Run DeepMASS_Web.

   ```bash
   sh run_replace.sh 
   ```

## Citation

In preparation



## Contact

Ji Hongchao   

E-mail: ji.hongchao@foxmail.com    

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-7364-0741" href="https://orcid.org/0000-0002-7364-0741" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-7364-0741</a></div>

WeChat public account: Chemocoder    

<img align="center" src="https://github.com/hcji/hcji/blob/main/img/qrcode.jpg" width="20%"/>

