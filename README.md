# DSCoV materials for deep learning

## Files -- email drew_linsley@brown.edu w/ questions

### DNN_scrape_and_finetune_python3.ipynb <a href="https://colab.research.google.com/github/dscov-tutorials/deep_learning/blob/master/DNN_scrape_and_finetune_python3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
An introduction to developing deep learning models and using them to recognize images. Load pre-trained DNNs, scrape images from the web, and "finetune" the DNNs to recognize these new images.

### DNN_scrape_and_finetune_script.py
Adataptation of the notebook to be run as a script for cloud computing tutorials. Changes correspond to removing Google specific code and saving images instead of interactive plotting.

* To run in aws with an ubuntu-deep-learning image

  ```bash
  source activate tensorflow_p27
  pip install google_images_download
  python DNN_scrape_and_finetune_script.py
  ```

