
# **Anime Face Generator using GANs**

This project demonstrates how to generate anime-style facial images using **Generative Adversarial Networks (GANs)** in a **Jupyter Notebook** environment. The implementation uses **PyTorch** and trains a **DCGAN** model on the Anime Face Dataset.  

---

## **Features**
- End-to-end implementation of DCGAN in a Jupyter Notebook.
- Uses **Anime Face Dataset** for training.
- GPU support for efficient training.
- Visualizations of generated images during training.
- Video generation from saved outputs to showcase training progress.

---

## **Requirements**

### **1. Environment**
Ensure you have Python 3.8+ and Jupyter Notebook installed.

### **2. Install Dependencies**
Install the required Python libraries:  
```bash
pip install torch torchvision opendatasets matplotlib tqdm opencv-python
```

---

## **Steps to Run the Project**

### **1. Open the Notebook**
Download and open the Jupyter Notebook containing the code in your preferred environment:  
```bash
jupyter notebook anime_face_gan.ipynb
```

### **2. Download the Dataset**
Use the `opendatasets` library to download the Anime Face Dataset from Kaggle. Add the following code in a notebook cell:
```python
import opendatasets as od

download_url = "https://www.kaggle.com/datasets/splcher/animefacedataset"
od.download(download_url)
```

This will download the dataset to a folder named `animefacedataset`.

### **3. Configure the Dataset**
Ensure the dataset directory structure is as follows:
```
animefacedataset/
  images/
    image1.jpg
    image2.jpg
    ...
```

### **4. Train the Model**
Run each cell of the notebook in sequence to:
1. Load and preprocess the dataset.
2. Define the **Discriminator** and **Generator** architectures.
3. Train the GAN using the provided training loop.

### **5. Generate Images**
After training, use the pre-trained generator to create anime faces:
```python
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(epoch, fixed_latent)
```
Generated images will be saved in the `generated/` folder.

---

## **Visualizing Results**
- View training progress with saved images in the `generated/` folder.
- Plot the loss curves for both **discriminator** and **generator** to analyze model performance:
```python
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')
```

### **Generate a Training Animation**
Combine the saved images into a video:
```python
import cv2
vid_fname = 'gans_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*'MP4V'), 1, (530, 530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()
```

---

## **Key Components**
- **Discriminator**: Identifies real vs. generated images using convolutional layers.
- **Generator**: Creates realistic anime-style faces from random noise using transposed convolution layers.
- **Loss Function**: Binary Cross-Entropy (BCE) for both networks.
- **Optimizer**: Adam with learning rate `0.0002`.

---

## **Future Enhancements**
- Explore advanced GAN architectures like **StyleGAN**.
- Add conditional generation for controlling attributes (e.g., gender, hair color).
- Train with a larger dataset for more detailed outputs.

---

## **Acknowledgments**
- **Dataset**: [Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **GAN Architecture**: Based on [DCGAN Paper (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)

---

Let me know if you need more tweaks or assistance!
