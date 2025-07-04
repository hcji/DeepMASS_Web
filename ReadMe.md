# DeepMASS_Web

DeepMASS_Web is an online platform developed from DeepMASS2, which enables deep-learning based metabolite annotation
 via semantic similarity analysis of mass spectral language. This approach enables the prediction 
 of structurally related metabolites for the unknown compounds. By considering the chemical space, these 
 structurally related metabolites provide valuable information about the potential location of the unknown 
 metabolites and assist in ranking candidates obtained from molecular structure databases. 

## News

[10/2024] Using DeepMASS2, we made a web UI interface, check it out! [Website](http://deepmass.cn/)

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
       mail_user: XX(eg:deepmass@sina.cn)
       mail_pwd: XX(eg:16c6533b293a1b75)
       mail_sender: XX(eg:deepmass@sina.cn)
       port: 25
       host: XX(eg:smtp.sina.com)
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

6. Using the `schema.sql` file located in `DeepMASS_Web/backend/sqlite`, create your empty database there with:

   ```bash
    sqlite3 ./backend/sqlite/User_Information.db < ./backend/sqlite/schema.sql
   ```

7. Replace all instances of the IP address `deepmass.cn` in the files with the IP address of your own host.

8. Comprehensive Guide to Deploying the DeepMASS Frontend on Linux with Nginx

   1. Install Nginx  

      ```bash
      # Install Nginx
      sudo apt install -y nginx
      
      # Enable & start Nginx on boot
      sudo systemctl enable nginx
      sudo systemctl start nginx
      ```

   2. Copy the frontend files out of `/root` to `/var/www`  

      ```bash
      # Create target directory
      sudo mkdir -p /var/www/DeepMASS_Web
      
      # Copy the frontend folder
      sudo cp -r /root/DeepMASS_Web/frontend /var/www/DeepMASS_Web/
      
      # Set ownership & permissions for Nginx (www-data)
      sudo chown -R www-data:www-data /var/www/DeepMASS_Web
      sudo chmod -R 755             /var/www/DeepMASS_Web
      
      ```

   3. Edit `/etc/nginx/nginx.conf` and inside the `http { ... }` block add the WebSocket `map`

      ```bash
      http {
          # ... existing settings ...
      
          # Enable proper handling of Upgrade/Connection headers
          map $http_upgrade $connection_upgrade {
              default   upgrade;
              ""        close;
          }
      
          include /etc/nginx/sites-enabled/*.conf;
      }
      ```

   4. Configure the site virtual host (`/etc/nginx/sites-available/default`)  

      ```bash
      server {
          listen 80 default_server;
          listen [::]:80 default_server;
          server_name www.deepmass.cn deepmass.cn; # 换成你的域名或 IP
      
          # Serve frontend static files
          root  /var/www/DeepMASS_Web/frontend;
          index index.html;
      
          # Single-page app routing support
          location / {
              try_files $uri $uri/ /index.html;
          }
      
          # Proxy /anal_sear/ to port 5578
          location = /anal_sear {
              return 302 /anal_sear/;
          }
          location /anal_sear/ {
              rewrite ^/anal_sear/(.*)$ /$1 break;
              proxy_pass         http://127.0.0.1:5578/;
              proxy_http_version 1.1;
              proxy_set_header   Upgrade    $http_upgrade;
              proxy_set_header   Connection $connection_upgrade;
              proxy_set_header   Host       $host;
          }
      
          # Proxy /comp_ident/ to port 12341
          location = /comp_ident {
              return 302 /comp_ident/;
          }
          location /comp_ident/ {
              rewrite ^/comp_ident/(.*)$ /$1 break;
              proxy_pass         http://127.0.0.1:12341/;
              proxy_http_version 1.1;
              proxy_set_header   Upgrade    $http_upgrade;
              proxy_set_header   Connection $connection_upgrade;
              proxy_set_header   Host       $host;
          }
      
          # Proxy API calls to FastAPI on port 8000
          location /api/ {
              rewrite ^/api/(.*)$ /$1 break;
              proxy_pass http://127.0.0.1:8000/;
          }
      }
      
      ```

   5. Enable and reload Nginx 

      ```bash
      # If not already enabled:
      sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/
      
      # Test configuration syntax
      sudo nginx -t
      
      # Reload Nginx to apply changes
      sudo systemctl reload nginx
      
      ```

9. Run DeepMASS_Web.

   ```bash
   sh run_replace.sh 
   ```

10. Browser test.

    Flush your browser cache or launch a private/incognito window, then visit `http://YOUR_SERVER_IP_OR_DOMAIN/` (replace with your own server IP or domain) to verify the updated frontend is loading correctly.

## Citation

In preparation



## Contact

Ji Hongchao   

E-mail: ji.hongchao@foxmail.com    

<div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-7364-0741" href="https://orcid.org/0000-0002-7364-0741" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-7364-0741</a></div>

WeChat public account: Chemocoder    

<img align="center" src="https://github.com/hcji/hcji/blob/main/img/qrcode.jpg" width="20%"/>